import os
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

# set numpy environment variables
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import datetime
import logging
import time

import numpy as np
import ray
# ray must be imported before pyarrow
import pyarrow

import torch
from sacred import Experiment

import src.utils.numpy_nest_utils as numpy_nest
# noinspection PyUnresolvedReferences
from experiments.jalko2017.MongoDBOption import TestOption, ExperimentOption, DatabaseOption
from experiments.jalko2017.ingredients.data_distribution import dataset_dist_ingred, generate_dataset_distribution_func
from experiments.jalko2017.ingredients.dataset_ingredient import dataset_ingredient, load_data
from experiments.jalko2017.measure_performance import compute_prediction_accuracy, compute_log_likelihood
from experiments.utils import save_log, save_pickle
from src.client.client import StandardClient, ensure_positive_t_i_factory
from src.model.logistic_regression_models import MeanFieldMultiDimensionalLogisticRegression
from src.privacy.optimizer import StandardOptimizer
from src.server import SyncronousPVIParameterServer, AsyncronousPVIParameterServer
from src.utils.yaml_string_dumper import YAMLStringDumper
import src.utils.numpy_nest_utils as np_nest

ex = Experiment("jalko2017_pure_PVI", [dataset_ingredient, dataset_dist_ingred])
logger = logging.getLogger(__name__)
pretty_dump = YAMLStringDumper()


@ex.config
def default_config(dataset, dataset_dist):
    # adapt settings based on the dataset ingredient
    dataset.name = "adult"

    dataset_dist.rho = 380

    PVI_settings = {
        'damping_factor': 1.,
        'damping_decay': 0.0,
        'async': False
    }

    optimisation_settings = {
        "lr": 0.2,
        "N_steps": 200,
        "lr_decay": 0,
        "L": 380
    }

    N_iterations = 30

    logging_base_directory = "logs"

    ray_cfg = {
        "redis_address": "None",
        "num_cpus": 1,
        "num_gpus": 0,
    }

    prior_pres = 1.0
    N_samples = 50

    experiment_tag = "client_bad_q_protection"

    slack_json_file = "slack.json"

    save_t_is = False

    log_level = 'info'

    ray_cfg = {
        "redis_address": None,
        "num_cpus": 1,
        "num_gpus": 0,
    }

    prediction = {
        "interval": 1,
        "type": "prohibit"
    }

    dataset_seed = 1


@ex.automain
def run_experiment(ray_cfg,
                   prior_pres,
                   PVI_settings,
                   optimisation_settings,
                   N_samples,
                   N_iterations,
                   prediction,
                   experiment_tag,
                   logging_base_directory,
                   save_t_is,
                   log_level,
                   _run,
                   _config,
                   seed):
    if log_level == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif log_level == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    torch.set_num_threads(int(ray_cfg["num_cpus"]))
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:

        training_set, test_set, d_in = load_data()
        clients_data, nis, prop_positive, M = generate_dataset_distribution_func()(training_set["x"], training_set["y"])

        _run.info = {
            **_run.info,
            "prop_positive": prop_positive,
            "n_is": nis,
        }

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

        def param_postprocess_function(delta_param, all_params, c):
            delta_ti = np_nest.apply_to_structure(lambda x: np.divide(x, len(c)), delta_param)

            ti_updates = []
            for client_index in c:
                new_client_params = np_nest.map_structure(np.add, all_params[client_index], delta_ti)
                precisions = new_client_params['w_pres']
                precisions[precisions < 0] = 1e-5
                new_client_params['w_pres'] = precisions
                ti_updates.append(np_nest.map_structure(np.subtract, new_client_params, all_params[client_index]))
                logger.debug()

            return ti_updates

        param_postprocess_handle = lambda delta, all_params, c: param_postprocess_function(delta, all_params, c)

        # client factories for each client - this avoids pickling of the client object for ray internals
        client_factories = [StandardClient.create_factory(
            model_class=MeanFieldMultiDimensionalLogisticRegression,
            data=clients_data[i],
            model_parameters=prior_params,
            model_hyperparameters={
                "base_optimizer_class": torch.optim.Adagrad,
                "wrapped_optimizer_class": StandardOptimizer,
                "base_optimizer_parameters": {"lr": optimisation_settings["lr"],
                                              "lr_decay": optimisation_settings["lr_decay"]},
                "wrapped_optimizer_parameters": {},
                "N_steps": optimisation_settings["N_steps"],
                "N_samples": N_samples,
                "n_in": d_in,
                "batch_size": optimisation_settings["L"],
            },
            hyperparameters={
                "t_i_init_function": lambda x: np.zeros(x.shape),
                "t_i_postprocess_function": ensure_positive_t_i_factory("w_pres"),
            },
            metadata={
                'client_index': i,
                'test_self': {
                    'accuracy': compute_prediction_accuracy,
                    'log_lik': compute_log_likelihood
                }
            }
        ) for i in range(M)]

        logger.info(f"Making M={M} Clients")

        # custom decorator based on passed in resources!
        remote_decorator = ray.remote(num_cpus=int(ray_cfg["num_cpus"]), num_gpus=int(ray_cfg["num_gpus"]))
        if PVI_settings['async']:
            server = remote_decorator(AsyncronousPVIParameterServer).remote(
                model_class=MeanFieldMultiDimensionalLogisticRegression,
                model_parameters=prior_params,
                hyperparameters={
                    "lambda_postprocess_func": param_postprocess_handle,
                    "damping_factor": PVI_settings['damping_factor'],
                    "damping_decay": PVI_settings['damping_decay']
                },
                max_iterations=N_iterations,
                client_factories=client_factories,
                prior=prior_params,
            )
        else:
            server = remote_decorator(SyncronousPVIParameterServer).remote(
                model_class=MeanFieldMultiDimensionalLogisticRegression,
                model_parameters=prior_params,
                hyperparameters={
                    "lambda_postprocess_func": param_postprocess_handle,
                    "damping_factor": PVI_settings['damping_factor'],
                    "damping_decay": PVI_settings['damping_decay']
                },
                max_iterations=N_iterations,
                client_factories=client_factories,
                prior=prior_params,
            )

        while not ray.get(server.should_stop.remote()):
            # dispatch work to ray and grab the log
            st_tick = time.time()
            ray.get(server.tick.remote())
            num_iterations = ray.get(server.get_num_iterations.remote())

            st_log = time.time()
            sacred_log = {}
            sacred_log["server"], _ = ray.get(server.log_sacred.remote())
            params = ray.get(server.get_parameters.remote())
            client_sacred_logs = ray.get(server.get_client_sacred_logs.remote())
            for i, log in enumerate(client_sacred_logs):
                sacred_log["client_" + str(i)] = log[0]
            sacred_log = numpy_nest.flatten(sacred_log, sep=".")

            st_pred = time.time()
            # predict every interval, and also for the last "interval" runs.
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
                     _run.info["test"], t), "full_log.json")

        if save_t_is:
            t_is = [client.t_i for client in ray.get(server.get_clients.remote())]
            ex.add_artifact(save_pickle(
                t_is, 't_is',  ex.get_experiment_info()["name"], experiment_tag, logging_base_directory, _run.info["test"], t
            ), 't_is.pkl')

    except pyarrow.lib.ArrowIOError:
        raise Exception("Experiment Terminated - was this you?")


    time.sleep(30)

    return test_acc
