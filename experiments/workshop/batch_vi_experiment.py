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

import src.utils.numpy_nest_utils as numpy_nest
# noinspection PyUnresolvedReferences
from experiments.workshop.ingredients.data_distribution import dataset_dist_ingred, generate_dataset_distribution_func
from experiments.workshop.ingredients.dataset_ingredient import dataset_ingredient, load_data
from experiments.workshop.measure_performance import compute_prediction_accuracy, compute_log_likelihood
from experiments.utils import save_log, save_pickle
from src.client.client import GradientVIClient
from src.model.logistic_regression_models import MeanFieldMultiDimensionalLogisticRegression
from src.privacy.optimizer import StandardOptimizer
from src.server import AsynchronousParameterServer
from src.utils.yaml_string_dumper import YAMLStringDumper

ex = Experiment("batch_vi", [dataset_ingredient, dataset_dist_ingred])
logger = logging.getLogger(__name__)
pretty_dump = YAMLStringDumper()


@ex.config
def default_config(dataset, dataset_dist):
    experiment_tag = 'batch_vi'

    dataset = {
        'name': 'adult',
        'scaled': True,
        'ordinal_cat_encoding': False,
        'train_proportion': 0.8,
        'data_base_dir': 'data',
    }

    dataset_dist = {
        'M': 10,
        'client_size_factor': 0,
        'class_balance_factor': 0,
        'dataset_seed': None,
    }

    optimisation_settings = {
        'lr': 0.2,
        'N_steps': 1,
        'lr_decay': 0,
        'L': 100,
    }

    ray_cfg = {
        'redis_address': None,
        'num_cpus': 1,
        'num_gpus': 0,
    }

    prediction = {
        'interval': 25,
        'type': 'prohibit',
    }

    N_iterations = 500
    prior_pres = 1.0
    N_samples = 50

    log_level = 'info'
    save_q = False
    logging_base_directory = 'logs'
    slack_json_file = 'slack.json'


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

        client_factories = [GradientVIClient.create_factory(
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
                "N_full": N_full
            },
            hyperparameters={
                'prior': prior_params
            },
            metadata={
                'client_index': i,
                'test_self': {
                    'accuracy': compute_prediction_accuracy,
                    'log_lik': compute_log_likelihood
                }
            }
        ) for i in range(M)]

        remote_decorator = ray.remote(num_cpus=int(ray_cfg["num_cpus"]), num_gpus=int(ray_cfg["num_gpus"]))
        server = remote_decorator(AsynchronousParameterServer).remote(
            model_class=MeanFieldMultiDimensionalLogisticRegression,
            model_parameters=prior_params,
            hyperparameters={
                "lambda_postprocess_func": lambda x: x,
                "damping_factor": 1.0,
                "damping_decay": 0.0
            },
            max_iterations=N_iterations,
            client_factories=client_factories,
            prior=prior_params,
        )

        total_communications = 0

        while not ray.get(server.should_stop.remote()):
            # dispatch work to ray and grab the log
            st_tick = time.time()
            communications_this_round = ray.get(server.tick.remote())
            total_communications += communications_this_round
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
                # y_pred_train = ray.get(server.get_model_predictions.remote(training_set))
                y_pred_test = ray.get(server.get_model_predictions.remote(test_set))
                # sacred_log["train_all"] = compute_log_likelihood(y_pred_train, training_set["y"])
                # sacred_log["train_accuracy"] = compute_prediction_accuracy(y_pred_train, training_set["y"])
                sacred_log["test_all"] = compute_log_likelihood(y_pred_test, test_set["y"])
                test_acc = compute_prediction_accuracy(y_pred_test, test_set["y"])
                sacred_log["test_accuracy"] = test_acc
            end_pred = time.time()

            for k, v in sacred_log.items():
                _run.log_scalar(k + '_time', v, num_iterations)
                _run.log_scalar(k + '_communications', v, total_communications)
            end = time.time()

            logger.info(f"Server Ticket Complete\n"
                        f"Server Timings:\n"
                        f"  Server Tick: {st_log - st_tick:.2f}s\n"
                        f"  Predictions: {end_pred - st_pred:.2f}s\n"
                        f"  Logging:     {end - end_pred + st_pred - st_log:.2f}s\n\n"
                        f"Iteration Number:{num_iterations}\n")
            logger.debug(f"Parameters:\n"
                         f" {pretty_dump.dump(params)}\n")

        final_log = ray.get(server.get_compiled_log.remote())
        final_log["N_i"] = nis
        final_log["Proportion_positive"] = prop_positive
        t = datetime.datetime.now()

        ex.add_artifact(
            save_log(final_log, "full_log", ex.get_experiment_info()["name"], experiment_tag, logging_base_directory,
                     _run.info["test"], t), 'full_log.json')

        if save_q:
            ex.add_artifact(save_pickle(
                ray.get(server.parameters.remote()), 't_is', ex.get_experiment_info()["name"], experiment_tag, logging_base_directory,
                _run.info["test"], t
            ), 't_is.pkl')

        return test_acc

    except pyarrow.lib.ArrowIOError:
        raise Exception("Experiment Terminated - was this you?")
