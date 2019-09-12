import os
import sys
import glob

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

print(module_path)
print(sys.path)

import argparse
import logging
from subprocess import call

from functools import partial
from multiprocessing.dummy import Pool
from ruamel.yaml import YAML

# from experiments.dispatch.ray_dispatcher import generate_commands_from_yaml
from experiments.dispatch.dispatch_utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--num-cpus", dest="num_cpus", default=1, type=int)
argparser.add_argument("--cpus-per-command", dest="cpu_per_com", default=1, type=int)
argparser.add_argument("-t", "--test", dest="test", action="store_true")
argparser.add_argument('--exp-file', nargs='+')
args = argparser.parse_args()

yaml = YAML()



def dispatch_command_strings(commands, num_cpus, cpus_per_command=1):
    at_once = int(float(num_cpus) / cpus_per_command)
    logger.info(f"Running {at_once} commands at a time")

    pool = Pool(at_once)  # two concurrent commands at a time
    for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
        if returncode != 0:
            print(f"{i} command failed: {returncode}")

if __name__ == "__main__":
    logger.info("Using Simple Dispatcher")

    exp_files = []
    for exp_path in args.exp_file:
        f = glob.glob(exp_path)
        if f is not None:
            exp_files.extend(f)
        else:
            raise FileNotFoundError("Could not find exp file")

    print(exp_files)

    try:
        commands = []
        for exp_file in exp_files:
            commands.extend(generate_commands_from_yaml(args, exp_file))
    except FileNotFoundError:
        logger.error("Could not find experiment dispatcher yaml file!")

    print('Running:')
    print(commands)
    dispatch_command_strings(commands, num_cpus=args.num_cpus, cpus_per_command=args.cpu_per_com)
