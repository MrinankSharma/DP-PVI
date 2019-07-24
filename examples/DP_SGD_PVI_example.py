import datetime
import json
import os

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver, SlackObserver

import src.privacy_accounting.analysis.moment_accountant as moment_accountant
import src.privacy_accounting.analysis.pld_accountant as pld_accountant
import src.utils.numpy_nest_utils as numpy_nest
from src.client import DPClient
from src.model.logistic_regression_models import MeanFieldMultiDimensionalLogisticRegression
from src.privacy_accounting.dp_query import GaussianDPQuery
from src.privacy_accounting.optimizer import DPOptimizer
from src.server import SyncronousPVIParameterServer

ex = Experiment('Full Experiment')

ex.observers.append(
    MongoObserver.create(
        url='localhost:9001',
        db_name='sacred'
    )
)

ex.observers.append(
    SlackObserver.from_config('../slack.json')
)


@ex.config
def cfg():
    N = 1000
    batch_size = 10
    learning_rate = 0.001
    epochs = 100
    privacy = {
        "l2_norm_clip": 5,
        "noise_multiplier": 4,
        "max_delta": 0.00001,
        "max_lambda": 32
    }


ex.add_config('test_config.yaml')


@ex.automain
def main(N, batch_size, learning_rate, epochs, privacy, _run):
    x = np.array([[2, 2], [1, 1], [0, 1], [1, 0], [-0.5, 0.1],
                  [-1, -1], [-2, -2], [0, -1], [-1, 0],
                  [0.5, 0.1]], dtype=np.float32)
    y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=np.float32)

    dataset = {
        'x': x,
        'y': y,
    }

    target_delta = 0.0001

    prior_params = {
        "w_nat_mean": np.array([0.0, 0.0], dtype=np.float64),
        "w_pres": np.array([0.1, 0.1], dtype=np.float64)
    }

    clients = [
        DPClient(model_class=MeanFieldMultiDimensionalLogisticRegression,
                 dp_query_class=GaussianDPQuery,
                 accounting_dict={
                     'MomentAccountant': {
                         'accountancy_update_method': moment_accountant.compute_online_privacy_from_ledger,
                         'accountancy_parameters': {
                             'target_delta': target_delta
                         }
                     },
                     'PLDAccountant': {
                         'accountancy_update_method': pld_accountant.compute_online_privacy_from_ledger,
                         'accountancy_parameters': {
                             'target_delta': target_delta,
                             'L': 50
                         }
                     }
                 },
                 data=dataset,
                 model_parameters=prior_params,
                 model_hyperparameters={
                     "base_optimizer_class": torch.optim.SGD,
                     "wrapped_optimizer_class": DPOptimizer,
                     "base_optimizer_parameters": {'lr': 0.02},
                     "wrapped_optimizer_parameters": {},
                     "N_steps": 10,
                     "N_samples": 50,
                     "n_in": 2,
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

    server = SyncronousPVIParameterServer(
        model_class=MeanFieldMultiDimensionalLogisticRegression,
        model_parameters=prior_params,
        model_hyperparameters={
            "base_optimizer_class": torch.optim.SGD,
            "wrapped_optimizer_class": None,
            "base_optimizer_parameters": {'lr': 0.02},
            "wrapped_optimizer_parameters": {},
            "N_steps": 500,
            "N_samples": 50,
            "n_in": 2,
            "prediction_integration_limit": 50,
            "batch_size": x.shape[0],
        },
        prior=prior_params,
        clients=clients,
        max_iterations=100
    )

    while not server.should_stop():
        server.tick()

        sacred_log = {}
        sacred_log['server'], _ = server.log_sacred()
        client_sacred_logs = [client.log_sacred() for client in server.clients]
        for i, log in enumerate(client_sacred_logs):
            sacred_log['client_' + str(i)] = log[0]

        sacred_log = numpy_nest.flatten(sacred_log, sep='.')

        for k, v in sacred_log.items():
            _run.log_scalar(k, v, server.iterations)

    final_log = {}
    final_log['server'] = server.get_log()
    client_logs = [client.get_log() for client in clients]
    for i, log in enumerate(client_logs):
        final_log['client_' + str(i)] = log

    log_dir = '../logs/tests'

    log_dir = os.path.join(log_dir, f'{datetime.datetime.now()}')
    dump_file = os.path.join(log_dir, 'results.json')

    os.makedirs(log_dir)

    with open(dump_file, 'w') as f:
        json.dump(final_log, f, indent=4)

    ex.add_artifact(dump_file, 'full_log')
