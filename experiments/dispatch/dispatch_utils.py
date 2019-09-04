import yaml
import itertools

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


def generate_commands_from_yaml(args, yaml_filepath):
    with open(yaml_filepath, "r") as yaml_file:
        exp_config = yaml.load(yaml_file.read())

    exp_file = exp_config.pop("experiment_file")
    if 'database' in exp_config:
        database = exp_config.pop("database")
    else:
        database = None

    seed_values = [("seed", i) for i in range(1, exp_config.pop("num_seeds") + 1)]
    all_options = nested_dict_to_option_strings(exp_config)
    all_options.append(seed_values)
    product = itertools.product(*all_options)

    if database is not None:
        run_flag = f'--database {database}'
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


