import subprocess
import os


def fully_normalized(path):
    return os.path.abspath(os.path.normpath(os.path.expanduser(path)))

def exec_command(cmd, shell=False, **kwargs):
    """Executes `cmd` and returns the stdout, stderr and exit code"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         shell=shell, **kwargs)
    out, err = p.communicate()
    return out, err, p.poll()