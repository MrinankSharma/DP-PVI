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
from src.client.client import DPClient, ensure_positive_t_i_factory
from src.model.logistic_regression_models import MeanFieldMultiDimensionalLogisticRegression
from src.privacy.dp_query import GaussianDPQuery
from src.privacy.optimizer import DPPercentileClippingGaussianOptimizer
from src.server import SyncronousPVIParameterServer
from src.utils.yaml_string_dumper import YAMLStringDumper

ex = Experiment('jalko2017_percentile_clip_datapoint_exp', [dataset_ingredient, dataset_dist_ingred])
logger = logging.getLogger(__name__)
pretty_dump = YAMLStringDumper()


@ex.config
def default_config(dataset, dataset_dist):
    # adapt settings based on the dataset ingredient
    dataset.name = "adult"

    PVI_settings = {
        'damping_factor': 0.1,
    }

    privacy_settings = {
        "L": 40,
        "C_percentile": 50,
        "target_delta": 1e-4,
        "sigma_relative": 1.22
    }

    optimisation_settings = {
        "lr": 0.25,
        "N_steps": 10,
        "lr_decay": 0,
    }

    N_iterations = 100

    logging_base_directory = "/scratch/DP-PVI/logs"

    ray_cfg = {
        "redis_address": None,
        "num_cpus": 1,
        "num_gpus": 0,
    }

    prior_pres = 1.0
    N_samples = 50

    prediction = {
        "interval": 1,
        "type": "prohibit"
    }
    experiment_tag = "datapoint_level_protection_percentile_clipping"

    slack_json_file = "/scratch/DP-PVI/DP-PVI/slack.json"

    save_t_is = False

    log_level = 'info'


@ex.automain
def run_experiment(ray_cfg, prior_pres, privacy_settings, optimisation_settings, PVI_settings, N_samples, N_iterations, prediction,
                   experiment_tag, logging_base_directory, log_level, save_t_is,
                   _run, _config, seed):
    if log_level == 'info':
        logger.setLevel(logging.INFO)
    elif log_level == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    torch.set_num_threads(int(ray_cfg["num_cpus"]))
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        training_set, test_set, d_in = load_data()
        clients_data, nis, prop_positive, M = generate_dataset_distribution_func()(training_set["x"], training_set["y"])

        if ray_cfg["redis_address"] is None:
            logger.info("Running locally")
            ray.init(num_cpus=ray_cfg["num_cpus"], num_gpus=ray_cfg["num_gpus"], logging_level=logging.DEBUG,
                     local_mode=True)
        else:
            logger.info("Connecting to existing server")
            ray.init(redis_address=ray_cfg["redis_address"], logging_level=logging.INFO)

        prior_params = {
            "w_nat_mean": np.zeros(d_in, dtype=np.float32),
            "w_pres": prior_pres * np.ones(d_in, dtype=np.float32)
        }

        logger.info(f"Prior Parameters:\n\n{pretty_dump.dump(prior_params)}\n")

        # client factories for each client - this avoids pickling of the client object for ray internals
        client_factories = [DPClient.create_factory(
            model_class=MeanFieldMultiDimensionalLogisticRegression,
            dp_query_class=GaussianDPQuery,
            data=clients_data[i],
            accounting_dict={
                'MomentAccountant': {
                    'accountancy_update_method': moment_accountant.compute_online_privacy_from_ledger,
                    'accountancy_parameters': {
                        'target_delta': privacy_settings["target_delta"]
                    }
                }
            },
            model_parameters=prior_params,
            model_hyperparameters={
                "base_optimizer_class": torch.optim.Adagrad,
                "wrapped_optimizer_class": DPPercentileClippingGaussianOptimizer,
                "base_optimizer_parameters": {'lr': optimisation_settings["lr"],
                                              'lr_decay': optimisation_settings["lr_decay"]},
                "wrapped_optimizer_parameters": {
                    "percentile": privacy_settings["C_percentile"],
                    "noise_multiplier": privacy_settings["sigma_relative"],
                },
                "N_steps": optimisation_settings["N_steps"],
                "N_samples": N_samples,
                "n_in": d_in,
                "batch_size": privacy_settings["L"],
            },
            hyperparameters={
                # note that these don't matter anymore
                'dp_query_parameters': {
                    'l2_norm_clip': 1,
                    'noise_stddev': 1
                },
                't_i_init_function': lambda x: np.zeros(x.shape),
                't_i_postprocess_function': ensure_positive_t_i_factory("w_pres"),
                "damping_factor": PVI_settings['damping_factor'],
            }
        ) for i in range(M)]

        # custom decorator based on passed in resources!
        remote_decorator = ray.remote(num_cpus=int(ray_cfg["num_cpus"]), num_gpus=int(ray_cfg["num_gpus"]))
        server = remote_decorator(SyncronousPVIParameterServer).remote(
            model_class=MeanFieldMultiDimensionalLogisticRegression,
            model_parameters=prior_params,
            model_hyperparameters={
                "prediction": prediction["type"],
            },
            hyperparameters={
                "damping_factor": PVI_settings["damping_factor"],
                "damping_decay": PVI_settings['damping_decay']
            },
            prior=prior_params,
            client_factories=client_factories,
            max_iterations=N_iterations
        )

        while not ray.get(server.should_stop.remote()):
            # dispatch work to ray and grab the log
            st_tick = time.time()
            ray.get(server.tick.remote())
            num_iterations = ray.get(server.get_num_iterations.remote())

            st_log = time.time()
            sacred_log = {}
            sacred_log['server'], _ = ray.get(server.log_sacred.remote())
            params = ray.get(server.get_parameters.remote())
            client_sacred_logs = ray.get(server.get_client_sacred_logs.remote())
            for i, log in enumerate(client_sacred_logs):
                sacred_log['client_' + str(i)] = log[0]
            sacred_log = numpy_nest.flatten(sacred_log, sep='.')

            st_pred = time.time()
            # predict every interval, and also for the last 'interval' runs.
            if ((num_iterations - 1) % prediction["interval"] == 0) or (
                    N_iterations - num_iterations < prediction["interval"]):
                y_pred_train = ray.get(server.get_model_predictions.remote(training_set))
                y_pred_test = ray.get(server.get_model_predictions.remote(test_set))
                sacred_log["train_all"] = compute_log_likelihood(y_pred_train, training_set["y"])
                sacred_log["train_accuracy"] = compute_prediction_accuracy(y_pred_train, training_set["y"])
                sacred_log["test_all"] = compute_log_likelihood(y_pred_test, test_set["y"])
                test_acc = compute_prediction_accuracy(y_pred_test, test_set["y"])
                sacred_log["test_accuracy"] = test_acc
            end_pred = time.time()

            for k, v in sacred_log.items():
                _run.log_scalar(k, v, num_iterations)
            end = time.time()

            logger.info(f"Server Ticket Complete\n"
                        f"Server Timings:\n"
                        f"  Server Tick: {st_log - st_tick:.2f}s\n"
                        f"  Predictions: {end_pred - st_pred:.2f}s\n"
                        f"  Logging:     {end - end_pred + st_pred - st_log:.2f}s\n\n"
                        f"Parameters:\n"
                        f" {pretty_dump.dump(params)}\n"
                        f"Iteration Number:{num_iterations}\n")

        final_log = ray.get(server.get_compiled_log.remote())
        final_log["N_i"] = nis
        final_log["Proportion_positive"] = prop_positive
        t = datetime.datetime.now()

        ex.add_artifact(
            save_log(final_log, "full_log", ex.get_experiment_info()["name"], experiment_tag, logging_base_directory,
                     _run.info["test"], t), 'full_log.json')

        if save_t_is:
            t_is = [client.t_i for client in ray.get(server.get_clients.remote())]
            ex.add_artifact(save_pickle(
                t_is, 't_is', ex.get_experiment_info()["name"], experiment_tag, logging_base_directory,
                _run.info["test"], t
            ), 't_is.pkl')

    except pyarrow.lib.ArrowIOError:
        raise Exception("Experiment Terminated - was this you?")

    return "done"
