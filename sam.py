import os
import glob
import itertools
import collections

import utils
import logging
import time
import re


logger = logging.getLogger(__name__)

GalaxyData = collections.namedtuple('Galaxy', 'stellar_mas bulge_mass bh_mass hi_mass')

def _submit_jobs(commands):
    for command in commands:
        out, err, code = utils.exec_command(command)
        if code != 0:
            logger.error("Error when submitting job with %s (exit code %d): stdout: %s, stderr: %s",
                         command, code, out, err)

def run_jobs(commands, max_concurrent, count_jobs, interval):

    logger.info('Scheduling %d jobs with maximum concurrenty %d', len(commands), max_concurrent)

    # Based on max_concurrent and the current number of jobs
    # calculate on each iteration how many commands can we submit
    while commands:
        avail_slots = max_concurrent - count_jobs()
        to_submit = commands[:avail_slots]
        commands = commands[avail_slots:]
        logger.info("Submitting %d jobs, %d remaining", len(to_submit), len(commands))

        # There's stuff to do
        if to_submit:
            _submit_jobs(to_submit)
        # Wait a bit before checking again
        if commands:
            time.sleep(interval)

    # Everything has been submitted, now let's wait
    while count_jobs() > 0:
        time.sleep(interval)

class BaseSAMRunner(object):
    """Base class for all SAM runners. It currently only holds common information"""

    def __init__(self, scripts_dir, obs_dir, sim_dir, sam_dir, output_dir, pkfile,
                 cluster, box_size, swarm_size, n_boxes, constrains):
        self.scripts_dir = utils.fully_normalized(scripts_dir)
        self.obs_dir = utils.fully_normalized(obs_dir)
        self.sim_dir = utils.fully_normalized(sim_dir)
        self.sam_dir = utils.fully_normalized(sam_dir)
        self.output_dir = utils.fully_normalized(output_dir)
        self.pkfile = utils.fully_normalized(pkfile)
        self.cluster = cluster.lower()
        self.box_size = box_size
        self.swarm = list(range(swarm_size))
        self.boxes = list(range(n_boxes))
        self.constrains = constrains

class GalformRunner(BaseSAMRunner):
    """A class that runs the Galform SAM within the context of a PSO loop"""

    nouts = 2
    omegaM = 0.275
    omegaB = 0.046
    omegaL = 0.725
    hubble = 0.701
    sigma8 = 0.816
    min_mhalo = 1e10
    snapshots = (176, 171);
    zout = (0.0, 0.1);

    def __init__(self, *args, **kwargs):
        super(GalformRunner, self).__init__(*args, **kwargs)
        self.jobs_name = 'Galform_MBII'

        # TODO: prepare scripts that will be used to submit jobs later on

    def count_jobs(self):
        """Returns how many jobs with self.jobs_name are currently queued or running"""

        try:
            out, err, code = utils.exec_command("qstat")
        except OSError:
            raise RuntimeError("Couldn't run qstat, is it installed?")

        if code:
            raise RuntimeError("qstat failed with code %d: stdout: %s, stderr: %s" % (code, out, err))

        lines_with_jobname = [l for l in out.splitlines() if self.jobs_name in l]
        return len(lines_with_jobname)

    def prepare_inputs(self):

        input_base = os.path.join(self.sam_dir, "Galform.input");
        with open(input_base, 'rt') as f:
            original_config = f.read()

        for swarm_element in self.swarm:

            # TODO: translate PSO particles into parameters, currently all zeros
            params = [0] * 9

            for box in self.boxes:

                # Produce a copy from the original configuration
                # We put it in a dictionary we can modify it easily within set_var
                x = {'config': original_config + ''}

                # Replace variables in configuration so they contain specific values
                def set_var(name, value):
                    pattern = r'^%s\s*=.*$' % name
                    replacement = '%s = %s' % (name, str(value))
                    x['config'] = re.sub(pattern, replacement, x['config'], flags=re.MULTILINE)

                # Global, particle-independent values
                set_var('aquarius_tree_file', "%s/treedir_%03d/tree_%03d" % (self.sim_dir, self.snapshots[0], self.snapshots[0]))
                set_var("PKfile", self.pkfile)
                set_var("min_halo_mass", "%e" % self.min_mhalo)
                set_var("omega0", self.omegaM);
                set_var("lambda0", self.omegaL);
                set_var("omegab", self.omegaB);
                set_var('h0', self.hubble)
                set_var('sigma8', self.sigma8)
                set_var('volume', self.box_size**3)
                set_var('nout', len(self.zout))
                set_var('zout', '[' + ','.join(['%f' % z for z in self.zout]) + ']')

                # The actual parameters to be calibrated
                set_var('alphahot', params[0])
                set_var('vhotdisk', params[1])
                set_var('vhotburst', params[2])
                set_var('alpha_cool', params[3])
                set_var('F_SMBH', params[4])
                set_var('nu_sf', params[5])
                set_var("stabledisk", params[6]);
                set_var("f_dyn", params[7]);
                set_var("tau_star_min", params[8]);

                # Write final configuration file with updated values
                inputfile = os.path.join(self.sam_dir, "Galform_%03d.N%d.input" % (box, swarm_element));
                with open(inputfile, 'wt') as f:
                    f.write(x['config'])

    def run(self, max_jobs, wait_interval):

        # Prepare the input files for Galform
        self.prepare_inputs()

        # Calculate all commands that need to be run
        commands = []
        for swarm_element, box in itertools.product(self.swarm, self.boxes):
            script = os.path.join(self.scripts_dir, "run_galform_%03d.N%d.csh" % (box, swarm_element))
            if 'geryon' == self.cluster:
                command = ['qsub', '-hard', script]
            elif 'geryon2' == self.cluster:
                command = ['qsub', script]
            commands.append(command)

        # Go, go, go!
        run_jobs(commands, max_jobs, self.count_jobs, wait_interval)

    def get_probabilities(self, swarm_element):

        # Read data for two different redshifts
        z0_gals = self.read_galaxies(swarm_element, self.zout[0]);
        z01_gals = self.read_galaxies(swarm_element, self.zout[1]);

        # Calculate probabilities
        probs = []
        if 'bh_bulge' in self.constrains:
            probs.append(BlackHoleBulge(z0_gals, self.obs_dir))
        if 'hi_massfunc' in self.constrains:
            probs.append(HIMassFunc(z0_gals, self.obs_dir))
        if 'st_massfunc' in self.constrains:
            probs.append(MassFunc(z01_gals))
        return probs

    def read_galaxies(self, swarm_element, z):

        all_galaxies = []
        for box in self.boxes:
            fname = os.path.join(self.output_dir, 'Galaxies', 'BOX_%03d' % box, str(swarm_element), 'calibration_z%3.1f.cat' % z)
            with open(fname, 'rt') as f:
                # Skip the header, then read all the data
                f.readline()
                galaxies = [GalaxyData([float(x) for x in line.split()]) for line in f]
            all_galaxies.extend(galaxies)

        return all_galaxies

    def remove_files(self):

        # Collect all filename first
        all_files = []
        for swarm_element, box in itertools.product(self.swarm, self.boxes):
            fname_pattern = os.path.join(self.output_dir, 'Galaxies', 'BOX_%03d' % box,  str(swarm_element), '*')
            files = glob.glob(fname_pattern)
            all_files.extend(files)

        # Now remove them all
        for f in all_files:
            os.remove(f)

class SharkRunner(object):
    pass