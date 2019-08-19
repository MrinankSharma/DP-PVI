import argparse
import itertools
import logging
import os
import subprocess
import time

import ray
from ruamel.yaml import YAML

from experiments.dispatch.ray_dispatcher import generate_commands_from_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--num-cpus", dest="num_cpus", default=1, type=int)
argparser.add_argument("--cpus-per-command", dest="cpu_per_com", default=1, type=int)
argparser.add_argument("--exp-file", action="append", dest="exp_file", required=False, type=str)
argparser.add_argument("-t", "--test", dest="test", action="store_true")
args = argparser.parse_args()

yaml = YAML()


def dispatch_command_strings(commands, num_cpus, cpus_per_command=1):
    at_once = int(float(num_cpus) / cpus_per_command)
    logger.info(f"Running {at_once} commands at a time")
    subcommand_lists = [commands[i:i + at_once] for i in range(0, len(commands), at_once)]

    for subcommands in subcommand_lists:
        processes = [
            subprocess.call(subcommand + f" ray_cfg.num_cpus={cpus_per_command}", shell=True)
            for subcommand in subcommands]

        print(processes)

        for p in processes:
            p.wait()


if __name__ == "__main__":
    logger.info("Using Simple Dispatcher")

    try:
        commands = []
        for exp_file in args.exp_file:
            commands.extend(generate_commands_from_yaml(exp_file))
    except FileNotFoundError:
        logger.error("Could not find experiment dispatcher yaml file!")

    dispatch_command_strings(commands, num_cpus=args.num_cpus, cpus_per_command=args.cpu_per_com)
