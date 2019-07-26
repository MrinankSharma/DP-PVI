import logging

import numpy as np
import ray
import torch
from sacred import Experiment

import src.privacy_accounting.analysis.moment_accountant as moment_accountant
import src.privacy_accounting.analysis.pld_accountant as pld_accountant
import src.utils.numpy_nest_utils as numpy_nest
# noinspection PyUnresolvedReferences
from experiments.jalko2017.MongoDBOption import TestMongoDbOption, ExperimentMongoDbOption
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
        privacy_settings = {
            "L": 10,
            "C": 5,
            "sigma_relative": 1,
            "target_delta": 1e-5
        }

        optimisation_settings = {
            "lr": 0.01,
            "N_steps": 10,
        }

    elif dataset["name"] == "adult":
        privacy_settings = {
            "L": 100,
            "C": 5,
            "target_delta": 1e-5
        }

        optimisation_settings = {
            "lr": 0.01
        }

    logging_base_directory = "/scratch/DP-PVI/logs"

    ray_cfg = {
        "redis_address": "None",
        "num_cpus": 1,
        "num_gpus": 0,
    }

    prior_pres = 1.0 / 10
    N_samples = 50
    N_iterations = 100


# @ray.remote
def perform_iteration(server):
    logger.info("Server Tick")
    server.tick()
    sacred_log = {}
    sacred_log['server'], _ = server.log_sacred()
    client_sacred_logs = [client.log_sacred() for client in server.clients]
    for i, log in enumerate(client_sacred_logs):
        sacred_log['client_' + str(i)] = log[0]

    sacred_log = numpy_nest.flatten(sacred_log, sep='.')
    return sacred_log


@ex.automain
def run_experiment(privacy_settings, optimisation_settings, logging_base_directory, N_samples, N_iterations, prior_pres,
                   ray_cfg, _run):
    if ray_cfg["redis_address"] == "None":
        ray.init(num_cpus=ray_cfg["num_cpus"], num_gpus=ray_cfg["num_gpus"])
    else:
        ray.init(redis_address=ray_cfg["redis_address"], num_cpus=ray_cfg["num_cpus"], num_gpus=ray_cfg["num_gpus"])

    training_set, test_set, d_in = load_data()

    prior_params = {
        "w_nat_mean": np.zeros(d_in, dtype=np.float32),
        "w_pres": prior_pres * np.ones(d_in, dtype=np.float32)
    }

    logger.info(f"Prior Parameters:\n\n{pretty_dump.dump(prior_params)}\n")

    clients = [DPClient(model_class=MeanFieldMultiDimensionalLogisticRegression,
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
                        )
               ]

    server = SyncronousPVIParameterServer(
        model_class=MeanFieldMultiDimensionalLogisticRegression,
        model_parameters=prior_params,
        model_hyperparameters={
            "prediction": "laplace"
        },
        prior=prior_params,
        clients=clients,
        max_iterations=N_iterations
    )

    while not server.should_stop():
        # dispatch work to ray and grab the log
        sacred_log = perform_iteration(server)

        # compute predictive performance
        y_pred_train = server.model.predict(training_set["x"])
        y_pred_test = server.model.predict(test_set["x"])
        sacred_log["train_ll"] = compute_log_likelihood(y_pred_train, training_set["y"])
        sacred_log["train_accuracy"] = compute_prediction_accuracy(y_pred_train, training_set["y"])
        sacred_log["test_ll"] = compute_log_likelihood(y_pred_test, test_set["y"])
        sacred_log["test_accuracy"] = compute_prediction_accuracy(y_pred_test, test_set["y"])

        # sacred_log = ray.get(perform_iteration.remote(server))
        for k, v in sacred_log.items():
            _run.log_scalar(k, v, server.iterations)

    final_log = server.get_compiled_log()
    ex.add_artifact(save_log(final_log, ex.get_experiment_info()["name"], logging_base_directory, _run.info["test"]),
                    'full_log')
