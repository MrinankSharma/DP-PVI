import itertools
import logging
import subprocess
import time
import argparse

from ruamel.yaml import YAML

import ray

logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--cpus", dest="num_cpus", default=None, type=int)
argparser.add_argument("--gpus", dest="num_gpus", default=None, type=int)
argparser.add_argument("--exp-file", dest="exp_file", required=True, type=str)
args = argparser.parse_args()

yaml = YAML()


def dispatch_command_strings(commands, cpus_per_command=1, gpus_per_command=0, pause_time=0.0):
    if not ray.is_initialized():
        raise Exception("Ray must be initialised to dispatch commands")

    while len(commands) > 0:
        resources = ray.available_resources()
        if "CPU" in resources and resources["CPU"] >= cpus_per_command:
            if gpus_per_command == 0 or ("GPU" in resources and resources["GPU"] >= gpus_per_command):
                command = commands.pop() + f" -r localhost:9003 &"
                logger.info("Starting", command)
                subprocess.call(command, shell=True)
                time.sleep(pause_time)

        time.sleep(0.5)


def nested_dict_to_option_strings(d, base_key=None):
    ret = []
    base_key_to_use = "" if base_key is None else f"{base_key}."
    for k, v in d.items():
        # print(f"{k} has value {v}")
        if isinstance(v, list):
            list_values = []
            for l in v:
                list_values.append((base_key_to_use + k, l))
            ret.append(list_values)
        elif isinstance(v, dict):
            ret.extend(nested_dict_to_option_strings(v, base_key_to_use + k))
        else:
            ret.append([(base_key_to_use + k, v)])
    return ret


def generate_commands_from_yaml(yaml_filepath):
    with open(yaml_filepath, "r") as yaml_file:
        exp_config = yaml.load(yaml_file.read())

    exp_file = exp_config.pop("experiment_file")
    seed_values = [("seed", i) for i in range(exp_config.pop("num_seeds"))]
    all_options = nested_dict_to_option_strings(exp_config)
    all_options.append(seed_values)
    product = itertools.product(*all_options)
    command_strings = []
    for p in product:
        # the -r flag indicates that this is a proper run!
        command_str = f"python {exp_file} --run with"
        for option in p:
            command_str = f"{command_str} {option[0]}={option[1]}"
        command_strings.append(command_str)

    return command_strings


if __name__ == "__main__":
    ray.init(redis_address="localhost:9003", num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    logger.info("Nodes", ray.nodes())
    logger.info("Tasks", ray.tasks())
    logger.info("Total Resources", ray.cluster_resources())
    logger.info("Available Resources", ray.available_resources())

    try:
        commands = generate_commands_from_yaml(args.exp_file)
    except FileNotFoundError:
        logger.error("Could not find experiment dispatcher yaml file!")

    dispatch_command_strings(commands, cpus_per_command=1, pause_time=1.)
