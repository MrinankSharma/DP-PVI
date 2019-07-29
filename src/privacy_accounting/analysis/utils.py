import os
import pickle

accountant_tables_dir = "accountant_tables"

from ray.services import logger


def set_accountant_tables_dir(new_dir):
    global accountant_tables_dir
    accountant_tables_dir = new_dir


def grab_pickled_accountant_results(filename):
    try:
        filepath = os.path.join(accountant_tables_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as file:
                return True, pickle.load(file), filepath
        else:
            return False, None, filepath
    except (IOError, FileNotFoundError):
        logger.error("Issue with opening Pickled Accountant")
