"""
n.d.padilla@gmail.com
CDPLagos
andres.ruiz@unc.edu.ar
"""
import argparse

import sam
import logging
import sys


logger = logging.getLogger(__name__)

def pso(sam_runner, num_steps, opts):

    for step in range(num_steps):

        logger.info("Starting PSO step %d/%d", step + 1, num_steps)

        # Execute the SAM (i.e., send it to the queue and wait for all
        # its tasks to finish)
        sam_runner.run(opts.max_jobs, opts.wait_time);

        # Collect probabilities form all swam elements
        all_probs = []
        for swarm_element in range(opts.swarm_size):
            probs = sam_runner.get_probabilities(swarm_element)
            all_probs.append(probs)

        # TODO: turn SAM-specific values into particle's data
        # TODO: Calculate next PSO step

        # Leave a clean slate for the next round
        sam_runner.remove_files()

def run(opts):

    # Choose which runner we're going to use
    if opts.sam == 'galform':
        logger.info("Running PSO for galform")
        sam_runner = sam.GalformRunner(opts.scripts_dir, opts.obs_dir, opts.sim_dir,
                                       opts.sam_dir, opts.output_dir_sam, opts.pkfile,
                                       opts.cluster, opts.box_size, opts.swarm_size, opts.num_boxes, opts.constrains)
    elif opts.sam == 'shark':
        logger.info("Running PSO for shark")
        sam_runner = sam.SharkRunner(opts.scripts_dir, opts.obs_dir, opts.sim_dir,
                                     opts.sam_dir, opts.output_dir_sam, opts.pkfile,
                                     opts.cluster, opts.box_size, opts.swarm_size, opts.num_boxes, opts.constrains)
    else:
        raise ValueError("Unsupported sam flavour: %s, Supported values are 'galform' and 'shark'" % opts.sam)

    pso(sam_runner, opts.num_steps, opts)

def main():

    parser = argparse.ArgumentParser()
    swarm_opts = parser.add_argument_group('Swarm options')
    swarm_opts.add_argument('-s', '--swarm-size', type=int, help="Size of the swarm, defaults to 50", default=50)
    swarm_opts.add_argument('-n', '--num-steps', type=int, help="Number of iterations to execute, defaults to 100", default=100)
    swarm_opts.add_argument('-i', '--inertia-weight', type=float, help="Inertia scaling factor, defaults to 0.72", default=0.72)
    swarm_opts.add_argument('-b', '--best-weight', type=float, help="Individual best scaling factor, defaults to 1.193", default=1.193)
    swarm_opts.add_argument('-g', '--global-weight', type=float, help="Global best scaling factor, defaults to 1.193", default=1.193)

    sam_opts = parser.add_argument_group('SAM options')
    sam_opts.add_argument('-S', '--sam', help='Which semianalytic model software is being run, defaults to galform', choices=['galform', 'shark'], default='galform')
    sam_opts.add_argument('-d', '--sim-dir', help='Directory with simulation output', default='~/pso-sam/sim-output')
    sam_opts.add_argument('-D', '--sam-dir', help='Directory where the SAM software can be found', default='~/pso-sam/sam')
    sam_opts.add_argument('-N', '--num-boxes', type=int, help="Number of boxes to evaluate on each step, defaults to 4", default=4)
    sam_opts.add_argument('-B', '--box-size', type=float, help="The box size, in XXXXXX units. Defaults to 39.6850263", default=39.6850263)
    sam_opts.add_argument('-O', '--output-dir-sam', help="Output directory of the semianalytic model", default='~/pso-sam/sam-output')
    sam_opts.add_argument('-C', '--constrains', help="Cluster where this tool is running, defaults to geryon", default='geryon')

    queue_opts = parser.add_argument_group("Queue system options")
    queue_opts.add_argument('-j', '--max-jobs', type=int, help="Maximum number of jobs to run concurrently, defaults to 100", default=100)
    queue_opts.add_argument('-w', '--wait-time', type=int, help="Waiting time between job queue checks in seconds, defaults to 10", default=10)

    general_opts = parser.add_argument_group("General options")
    general_opts.add_argument('-c', '--cluster', help="Cluster where this tool is running, defaults to geryon", default='geryon')
    general_opts.add_argument('-o', '--obs-dir', help="Directory where observational data is found, defaults to ~/pso-sam/obsdata", default='~/pso-sam/obsdata')
    general_opts.add_argument('-p', '--pkfile', help="Path to power spectrum file, defaults to ./pso-sam/Power_Spec/pk_MassiveBlack2_norm.dat", default='~/pso-sam/Power_Spec/pk_MassiveBlack2_norm.dat')
    general_opts.add_argument('--scripts-dir', help="Directory where submission scripts will be written into, defaults to ~/pso-sam/scripts", default="~/pso-sam/scripts")

    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run(opts)

if __name__ == '__main__':
    main()