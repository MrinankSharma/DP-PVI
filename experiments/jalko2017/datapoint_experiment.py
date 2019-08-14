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
import src.privacy.analysis.pld_accountant as pld_accountant
import src.utils.numpy_nest_utils as numpy_nest
# noinspection PyUnresolvedReferences
from experiments.jalko2017.MongoDBOption import TestOption, ExperimentOption
from experiments.jalko2017.ingredients.dataset_ingredient import dataset_ingredient, load_data
from experiments.jalko2017.ingredients.data_distribution import dataset_dist_ingred, generate_dataset_distribution_func
from experiments.jalko2017.measure_performance import compute_prediction_accuracy, compute_log_likelihood
from experiments.utils import save_log
from src.client.client import DPClient, ensure_positive_t_i_factory
from src.model.logistic_regression_models import MeanFieldMultiDimensionalLogisticRegression
from src.privacy.dp_query import GaussianDPQuery
from src.privacy.optimizer import DPOptimizer
from src.server import SyncronousPVIParameterServer
from src.utils.yaml_string_dumper import YAMLStringDumper

ex = Experiment('jalko2017_datapoint_exp', [dataset_ingredient, dataset_dist_ingred])
logger = logging.getLogger(__name__)
pretty_dump = YAMLStringDumper()


@ex.config
def default_config(dataset, dataset_dist):
    print(dataset)
    print(dataset_dist)

@ex.automain
def run_experiment(_run, _config, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    training_set, test_set, d_in = load_data()
    func = generate_dataset_distribution_func()
    data, nis, prop_positive = func(training_set['x'], training_set['y'])
    for (x, y) in data:
        print(f"x shape {x.shape} y shape {y.shape}")

    print(prop_positive)

    return "done"
