import logging

import numpy as np
import ray
import torch
from sacred import Experiment

import src.privacy_accounting.analysis.moment_accountant as moment_accountant
import src.privacy_accounting.analysis.pld_accountant as pld_accountant
import src.utils.numpy_nest_utils as numpy_nest
# noinspection PyUnresolvedReferences
from experiments.jalko2017.MongoDBOption import TestOption, ExperimentOption
from experiments.jalko2017.ingredients.dataset_ingredient import dataset_ingredient, load_data
from experiments.jalko2017.measure_performance import compute_prediction_accuracy, compute_log_likelihood
from experiments.utils import save_log
from src.client.client import DPClient
from src.model.logistic_regression_models import MeanFieldMultiDimensionalLogisticRegression
from src.privacy_accounting.dp_query import GaussianDPQuery
from src.privacy_accounting.optimizer import DPOptimizer
from src.server import SyncronousPVIParameterServer
from src.utils.yaml_string_dumper import YAMLStringDumper

ex = Experiment('jalko2017', [dataset_ingredient])
logger = logging.getLogger(__name__)
pretty_dump = YAMLStringDumper()


@ex.config
def default_config(dataset):
    # adapt settings based on the dataset ingredient
    if dataset["name"] == "abalone":
        # settings from antiis paper
        privacy_settings = {
            "L": 167,
            "C": 5,
            "sigma_relative": 1.22,
            "target_delta": 1e-3
        }

        N_iterations = 1000

    elif dataset["name"] == "adult":
        privacy_settings = {
            "L": 195,
            "C": 75,
            "target_delta": 1e-3
        }

        N_iterations = 2000

    logging_base_directory = "/scratch/DP-PVI/logs"

    ray_cfg = {
        "redis_address": "None",
        "num_cpus": 1,
        "num_gpus": 0,
    }

    optimisation_settings = {
        "lr": 0.1,
        "N_steps": 1,
    }

    prior_pres = 1.0
    N_samples = 100

    prediction_type = "laplace"

    experiment_tag = "test"


@ex.automain
def run_experiment(privacy_settings, optimisation_settings, logging_base_directory, N_samples, N_iterations, prior_pres,
                   ray_cfg, prediction_type, _run):
    if ray_cfg["redis_address"] == "None":
        logger.info("Creating new ray server")
        ray.init(num_cpus=ray_cfg["num_cpus"], num_gpus=ray_cfg["num_gpus"], logging_level=logging.INFO)
    else:
        logger.info("Connecting to existing server")
        ray.init(redis_address=ray_cfg["redis_address"], logging_level=logging.INFO)

    training_set, test_set, d_in = load_data()

    prior_params = {
        "w_nat_mean": np.zeros(d_in, dtype=np.float32),
        "w_pres": prior_pres * np.ones(d_in, dtype=np.float32)
    }

    logger.info(f"Prior Parameters:\n\n{pretty_dump.dump(prior_params)}\n")

    # client factories for each client - this avoids pickling of the client object for ray internals
    client_factories = [DPClient.create_factory(
        model_class=MeanFieldMultiDimensionalLogisticRegression,
        dp_query_class=GaussianDPQuery,
        data=training_set,
        accounting_dict={
            'MomentAccountant': {
                'accountancy_update_method': moment_accountant.compute_online_privacy_from_ledger,
                'accountancy_parameters': {
                    'target_delta': privacy_settings["target_delta"]
                }
            },
            'PLDAccountant': {
                'accountancy_update_method': pld_accountant.compute_online_privacy_from_ledger,
                'accountancy_parameters': {
                    'target_delta': privacy_settings["target_delta"],
                    'L': 50
                }
            }
        },
        model_parameters=prior_params,
        model_hyperparameters={
            "base_optimizer_class": torch.optim.Adagrad,
            "wrapped_optimizer_class": DPOptimizer,
            "base_optimizer_parameters": {'lr': optimisation_settings["lr"]},
            "wrapped_optimizer_parameters": {},
            "N_steps": optimisation_settings["N_steps"],
            "N_samples": N_samples,
            "n_in": d_in,
            "batch_size": privacy_settings["L"],
        },
        hyperparameters={
            'dp_query_parameters': {
                'l2_norm_clip': privacy_settings["C"],
                'noise_stddev': privacy_settings["C"] * privacy_settings["sigma_relative"]
            },
            't_i_init_function': lambda x: np.zeros(x.shape)
        }
    )]

    # custom decorator based on passed in resources!
    remote_decorator = ray.remote(num_cpus=int(ray_cfg["num_cpus"]), num_gpus=int(ray_cfg["num_gpus"]))
    server = remote_decorator(SyncronousPVIParameterServer).remote(
        model_class=MeanFieldMultiDimensionalLogisticRegression,
        model_parameters=prior_params,
        model_hyperparameters={
            "prediction": prediction_type,
        },
        prior=prior_params,
        clients_factories=client_factories,
        max_iterations=N_iterations
    )

    while not ray.get(server.should_stop.remote()):
        # dispatch work to ray and grab the log
        logger.info("Server Tick")
        ray.get(server.tick.remote())
        num_iterations = ray.get(server.get_num_iterations.remote())
        sacred_log = {}
        sacred_log['server'], _ = ray.get(server.log_sacred.remote())
        params = ray.get(server.get_parameters.remote())
        client_sacred_logs = ray.get(server.get_client_sacred_logs.remote())
        for i, log in enumerate(client_sacred_logs):
            sacred_log['client_' + str(i)] = log[0]

        sacred_log = numpy_nest.flatten(sacred_log, sep='.')
        logger.info(f"Completed Iteration {num_iterations} Parameters:\n {pretty_dump.dump(params)}")

        # compute predictive performance
        y_pred_train = ray.get(server.get_model_predictions.remote(training_set))
        y_pred_test = ray.get(server.get_model_predictions.remote(test_set))
        sacred_log["train_all"] = compute_log_likelihood(y_pred_train, training_set["y"])
        sacred_log["train_accuracy"] = compute_prediction_accuracy(y_pred_train, training_set["y"])
        sacred_log["test_all"] = compute_log_likelihood(y_pred_test, test_set["y"])
        sacred_log["test_accuracy"] = compute_prediction_accuracy(y_pred_test, test_set["y"])

        for k, v in sacred_log.items():
            _run.log_scalar(k, v, num_iterations)

    final_log = ray.get(server.get_compiled_log.remote())
    ex.add_artifact(save_log(final_log, ex.get_experiment_info()["name"], logging_base_directory, _run.info["test"]),
                    'full_log')
