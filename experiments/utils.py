import datetime
import json
import os
import pickle


def save_log(log_dict, log_name, exp_name, exp_tag, base_output_dir, test, timestamp):
    if test:
        log_dir_modifier = "test"
    else:
        log_dir_modifier = "runs"

    log_dir = os.path.join(base_output_dir, exp_name, exp_tag, log_dir_modifier,
                           f"{timestamp.strftime('%d-%m-%yT%H:%M:%S')}")
    dump_file = os.path.join(log_dir, f'{log_name}.json')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(dump_file, 'w') as f:
        json.dump(log_dict, f, indent=4)

    return dump_file

def save_pickle(object, file_name, exp_name, exp_tag, base_output_dir, test, timestamp):
    if test:
        log_dir_modifier = "test"
    else:
        log_dir_modifier = "runs"

    log_dir = os.path.join(base_output_dir, exp_name, exp_tag, log_dir_modifier,
                           f"{timestamp.strftime('%d-%m-%yT%H:%M:%S')}")
    dump_file = os.path.join(log_dir, f'{file_name}.pkl')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(dump_file, 'wb') as f:
        pickle.dump(object, f)

    return dump_file