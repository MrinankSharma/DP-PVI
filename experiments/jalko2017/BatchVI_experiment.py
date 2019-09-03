import os
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

# set numpy environment variables
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6

import datetime
import logging
import time

from copy import deepcopy

import numpy as np
import ray
# ray must be imported before pyarrow
import pyarrow

import torch
from sacred import Experiment

import src.privacy.analysis.moment_accountant as moment_accountant
import src.utils.numpy_nest_utils as numpy_nest
# noinspection PyUnresolvedReferences
from experiments.jalko2017.MongoDBOption import TestOption, ExperimentOption, DatabaseOption
from experiments.jalko2017.ingredients.data_distribution import dataset_dist_ingred, generate_dataset_distribution_func
from experiments.jalko2017.ingredients.dataset_ingredient import dataset_ingredient, load_data
from experiments.jalko2017.measure_performance import compute_prediction_accuracy, compute_log_likelihood
from experiments.utils import save_log, save_pickle
from src.client.client import StandardClient, ensure_positive_t_i_factory
from src.model.logistic_regression_models import MeanFieldMultiDimensionalLogisticRegression
from src.privacy.dp_query import NumpyGaussianDPQuery, NumpyNoDPSumQuery
from src.privacy.optimizer import StandardOptimizer
from src.server import SyncronousPVIParameterServer
from src.utils.yaml_string_dumper import YAMLStringDumper
import src.utils.numpy_utils as B
import src.utils.numpy_nest_utils as np_nest

ex = Experiment("BatchVI_test", [dataset_ingredient, dataset_dist_ingred])
logger = logging.getLogger(__name__)
pretty_dump = YAMLStringDumper()


@ex.config
def default_config(dataset, dataset_dist):
    # adapt settings based on the dataset ingredient
    dataset.name = "adult"

    dataset_dist.rho = 380

    optimisation_settings = {
        "lr": 0.1,
        "N_steps": 1,
        "lr_decay": 0,
        "L": 380
    }

    N_iterations = 1000

    logging_base_directory = "logs"

    prior_pres = 1.0
    N_samples = 50

    experiment_tag = "client_bad_q_protection"

    slack_json_file = "slack.json"

    save_q = False

    log_level = 'info'

    ray_cfg = {
        "redis_address": None,
        "num_cpus": 1,
        "num_gpus": 0,
    }

    prediction = {
        "interval": 10,
        "type": "prohibit"
    }


@ex.automain
def run_experiment(ray_cfg,
                   prior_pres,
                   optimisation_settings,
                   N_samples,
                   N_iterations,
                   prediction,
                   experiment_tag,
                   logging_base_directory,
                   save_q,
                   log_level,
                   _run,
                   _config,
                   seed):
    torch.set_num_threads(int(ray_cfg["num_cpus"]))
    np.random.seed(seed)
    torch.manual_seed(seed)

    if log_level == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif log_level == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    try:

        training_set, test_set, d_in = load_data()
        clients_data, nis, prop_positive, M = generate_dataset_distribution_func()(training_set["x"], training_set["y"])

        # _run.info = {
        #     **_run.info,
        #     "prop_positive": prop_positive,
        #     "n_is": nis,
        # }

        logger.info(f'N_i\'s {nis}')
        logger.info(f'Class ratios\'s {prop_positive}')

        # time.sleep(np.random.uniform(0, 10))

        if ray_cfg["redis_address"] == None:
            logger.info("Running Locally")
            ray.init(num_cpus=ray_cfg["num_cpus"], num_gpus=ray_cfg["num_gpus"], logging_level=logging.INFO,
                     local_mode=True)
        else:
            logger.info("Connecting to existing server")
            ray.init(redis_address=ray_cfg["redis_address"], logging_level=logging.INFO)

        prior_params = {
            "w_nat_mean": np.zeros(d_in, dtype=np.float32),
            "w_pres": prior_pres * np.ones(d_in, dtype=np.float32)
        }

        logger.debug(f"Prior Parameters:\n\n{pretty_dump.dump(prior_params)}\n")

        # create the model to optimise in batch VI fashion

        N_full = np.sum(nis)

        model = MeanFieldMultiDimensionalLogisticRegression(
            parameters=np_nest.map_structure(np.add, prior_params, prior_params),
            hyperparameters={
                "base_optimizer_class": torch.optim.Adagrad,
                "wrapped_optimizer_class": StandardOptimizer,
                "base_optimizer_parameters": {"lr": optimisation_settings["lr"],
                                              "lr_decay": optimisation_settings["lr_decay"]},
                "wrapped_optimizer_parameters": {},
                "N_steps": optimisation_settings["N_steps"],
                "N_samples": N_samples,
                "n_in": d_in,
                "batch_size": optimisation_settings["L"],
                "N_full": N_full
            }
        )

        parameters = prior_params

        client_probs = 1 / np.array([data['x'].shape[0] for data in clients_data])
        client_probs = client_probs / client_probs.sum()


        for epoch in range(N_iterations):
            # dispatch work to ray and grab the log
            st_tick = time.time()

            # fit the model to each batch of data
            for i in range(len(clients_data)):

                client_index = int(np.random.choice(len(clients_data), 1, replace=False, p=client_probs))

                parameters = model.fit(clients_data[client_index], parameters)
                parameters = np_nest.map_structure(np.subtract, parameters, prior_params)

            st_log = time.time()
            sacred_log = {}

            sacred_log = numpy_nest.flatten(sacred_log, sep=".")

            st_pred = time.time()
            # predict every interval, and also for the last "interval" runs.
            if (epoch % prediction["interval"] == 0):
                y_pred_train = model.predict(training_set['x'])
                y_pred_test = model.predict(test_set['x'])
                sacred_log["train_all"] = compute_log_likelihood(y_pred_train, training_set["y"])
                sacred_log["train_accuracy"] = compute_prediction_accuracy(y_pred_train, training_set["y"])
                sacred_log["test_all"] = compute_log_likelihood(y_pred_test, test_set["y"])
                test_acc = compute_prediction_accuracy(y_pred_test, test_set["y"])
                sacred_log["test_accuracy"] = test_acc
            end_pred = time.time()

            for k, v in sacred_log.items():
                _run.log_scalar(k, v, epoch)
            end = time.time()

            logger.info(f"Server Ticket Complete\n"
                        f"Server Timings:\n"
                        f"  Server Tick: {st_log - st_tick:.2f}s\n"
                        f"  Predictions: {end_pred - st_pred:.2f}s\n"
                        f"  Logging:     {end - end_pred + st_pred - st_log:.2f}s\n\n"
                        f"Iteration Number:{epoch}\n")
            logger.debug(f"Parameters:\n"
                         f" {pretty_dump.dump(parameters)}\n")
        t = datetime.datetime.now()

        # time.sleep(np.random.uniform(0, 100))

        if save_q:
            ex.add_artifact(save_pickle(
                parameters, 't_is', ex.get_experiment_info()["name"], experiment_tag, logging_base_directory,
                _run.info["test"], t
            ), 't_is.pkl')

    except pyarrow.lib.ArrowIOError:
        raise Exception("Experiment Terminated - was this you?")

    return test_acc
