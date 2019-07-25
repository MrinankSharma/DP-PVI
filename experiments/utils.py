import datetime
import json
import os


def save_log(log_dict, exp_name, base_output_dir, test):
    if test:
        log_dir_modifier = "test"
    else:
        log_dir_modifier = "runs"

    log_dir = os.path.join(base_output_dir, exp_name, log_dir_modifier, datetime.datetime.now())
    dump_file = os.path.join(log_dir, 'results.json')

    os.makedirs(log_dir)

    with open(dump_file, 'w') as f:
        json.dump(log_dict, f, indent=4)

    return dump_file
