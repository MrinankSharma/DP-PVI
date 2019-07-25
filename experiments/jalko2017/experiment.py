import logging

import numpy as np
import ray
import torch
from sacred import Experiment

import src.privacy_accounting.analysis.moment_accountant as moment_accountant
import src.privacy_accounting.analysis.pld_accountant as pld_accountant
# noinspection PyUnresolvedReferences
from experiments.jalko2017.MongoDBOption import TestMongoDbOption, ExperimentMongoDbOption
from experiments.jalko2017.ingredients.dataset_ingredient import dataset_ingredient, load_data
from src.client.client import DPClient
from src.model.logistic_regression_models import MeanFieldMultiDimensionalLogisticRegression
from src.privacy_accounting.dp_query import GaussianDPQuery
from src.privacy_accounting.optimizer import DPOptimizer
from src.utils.yaml_string_dumper import YAMLStringDumper

ex = Experiment('jalko2017', [dataset_ingredient])
logger = logging.getLogger(__name__)
pretty_dump = YAMLStringDumper()


@ex.config
def default_config(dataset):
    # adapt settings based on the dataset ingredient
    if dataset["name"] == "abalone":
        privacy_settings = {
            "L": 100,
            "C": 5,
            "target_delta": 1e-5
        }

        optimisation_settings = {
            "lr": 0.01,
            "N_steps": 50,
        }

    elif dataset["name"] == "adult":
        privacy_settings = {
            "L": 100,
            "C": 5,
            "target_delta": 1e-5
        }

        optimisation_settings = {
            "base_optimiser_settings": {
                "lr": 0.01
            },
            "wrapped_optimiser_settings": {

            }
        }

    logging_base_directory = "/scratch/DP-PVI/logs/"

    ray_cfg = {
        "redis_address": "None",
        "num_cpus": 1,
        "num_gpus": 0,
    }

    prior_pres = 1.0 / 10


@ex.automain
def run_experiment(privacy_settings, optimisation_settings, logging_base_directory, prior_pres, ray_cfg, _run):
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
                 model_parameters=prior_params,
                 model_hyperparameters={
                     "base_optimizer_class": torch.optim.SGD,
                     "wrapped_optimizer_class": DPOptimizer,
                     "base_optimizer_parameters": {'lr': 0.02},
                     "wrapped_optimizer_parameters": {},
                     "N_steps": 10,
                     "N_samples": 50,
                     "n_in": d_in,
                     "prediction_integration_limit": 50,
                     "batch_size": x.shape[0],
                 },
                 hyperparameters={
                     'dp_query_parameters': {
                         'l2_norm_clip': 5,
                         'noise_stddev': 4
                     },
                     't_i_init_function': lambda x: np.zeros(x.shape)
                 }
                 )
    ]
