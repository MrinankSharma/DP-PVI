import argparse
import itertools
import logging
import os
import subprocess
import time

import ray
from ruamel.yaml import YAML

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--num-cpus", dest="num_cpus", default=1, type=int)
argparser.add_argument("--cpus-per-command", dest="cpu_per_com", default=1, type=int)
argparser.add_argument("--num-gpus", dest="num_gpus", default=0, type=int)
argparser.add_argument("--exp-file", action="append", dest="exp_file", required=False, type=str)
argparser.add_argument("-t", "--test", dest="test", action="store_true")
args = argparser.parse_args()

yaml = YAML()


def dispatch_command_strings(commands, cpus_per_command=1, gpus_per_command=0, pause_time=10.0):
    if not ray.is_initialized():
        raise Exception("Ray must be initialised to dispatch commands")

    while len(commands) > 0:
        resources = ray.available_resources()
        logger.info(f"Currently Available {resources}")
        if "CPU" in resources and resources["CPU"] >= cpus_per_command:
            if gpus_per_command == 0 or ("GPU" in resources and resources["GPU"] >= gpus_per_command):
                command = commands.pop() + f" ray_cfg.redis_address=localhost:9002 ray_cfg.num_cpus={cpus_per_command} ray_cfg.num_gpus={gpus_per_command} &"
                logger.info(f"Starting  {command}")
                subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)
                time.sleep(pause_time)
        else:
            # wait a minute to see if more resources are free
            time.sleep(60.0)


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
    if 'collection' in exp_config:
        collection = exp_config.pop("collection")
    else:
        collection = None

    seed_values = [("seed", i) for i in range(1, exp_config.pop("num_seeds") + 1)]
    all_options = nested_dict_to_option_strings(exp_config)
    all_options.append(seed_values)
    product = itertools.product(*all_options)

    if collection is not None:
        run_flag = f'--collection {collection}'
    else:
        run_flag = "--test" if args.test else "--experiment"

    command_strings = []
    for p in product:
        # the -r flag indicates that this is a proper run!
        command_str = f"python {exp_file} {run_flag} with"
        for option in p:
            command_str = f"{command_str} {option[0]}={option[1]}"
        command_strings.append(command_str)

    return command_strings


if __name__ == "__main__":
    logger.info("Creating  Ray Server")
    # for some reason this don't work on Hinton
    # subprocess.call(
    #     f"ray stop; ray start --head --redis-port 9002 --num-cpus {args.num_cpus} --num-gpus {args.num_gpus}",
    #     shell=True)
    ray.init(redis_address="localhost:9002")
    logger.info("Ray Server Created")
    # logger.info(f"Nodes: {ray.nodes()}")
    # logger.info(f"Tasks {ray.tasks()}")
    init_resources = ray.cluster_resources()
    logger.info(f"Total Resources {init_resources}")
    # logger.info(f"Available Resources {ray.available_resources()}")

    logger.info(f"Subprocess console output stored in {os.devnull}")

    try:
        commands = []
        for exp_file in args.exp_file:
            commands.extend(generate_commands_from_yaml(exp_file))
    except FileNotFoundError:
        logger.error("Could not find experiment dispatcher yaml file!")

    dispatch_command_strings(commands, cpus_per_command=args.cpu_per_com, pause_time=2.0)
    # if we have unavailable resources - right at the end (e.g. imagine sending off two commands - we don't want ray to
    # shutdown immediately!)
    while not init_resources == ray.available_resources():
        time.sleep(60)
    logger.info("All resources available again - taking down server")
    ray.shutdown()
    subprocess.call("ray stop", shell=True)
