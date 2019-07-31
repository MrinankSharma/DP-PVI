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
from src.client import DPClient, StandardClient
from src.model.linear_regression_models import LinearRegressionMultiDimSGD
from src.privacy_accounting.dp_query import GaussianDPQuery
from src.privacy_accounting.optimizer import DPOptimizer, StandardOptimizer
from src.server import SyncronousPVIParameterServer

ex = Experiment('Multi-dim Linear Regression Experiment')

#ex.observers.append(
#    MongoObserver.create(
#        url='localhost:9001',
#        db_name='sacred'
#    )
#)

#ex.observers.append(
#    SlackObserver.from_config('../slack.json')
#)


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


#ex.add_config('test_config.yaml')


@ex.automain
def main(N, batch_size, learning_rate, epochs, privacy, _run):
    x = np.array([[1, 0.29449245], [1, -0.14604409], [1, -0.21363078], [1, 2.50904502], 
                  [1, -0.0729865], [1, 0.74002993], [1, -0.62756792], [1, -0.19469873], 
                  [1, 0.19276911], [1, 0.68966338]], dtype=np.float32)
#    y = np.array([1.29449245, 0.85395591, 0.78636922, 3.50904502, 0.9270135,
#                  1.74002993, 0.37243208, 0.80530127, 1.19276911, 1.68966338], dtype=np.float32)
    y = np.array([0.70853103, 0.240928, -1.41032053, 2.88666036, 2.0095466,
                  3.53902563, 1.18043607, -1.05195666, 0.55742282, 1.7068966], dtype=np.float32)

    dataset = {
        'x': x,
        'y': y,
    }

    target_delta = 0.0001

    prior_params = {
        "w_nat_mean": np.array([0.0, 0.0], dtype=np.float64),
        "w_pres": np.array([1.0, 1.0], dtype=np.float64)
    }

    clients = [
        DPClient.create_factory(
            model_class=LinearRegressionMultiDimSGD,
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
                "n_in": 2,
                "batch_size": x.shape[0],
                "model_noise": 1
            },
            hyperparameters={
                'dp_query_parameters': {
                    'l2_norm_clip': 5,
                    'noise_stddev': 4
                },
                't_i_init_function': lambda x: np.zeros(x.shape)
            }
        )
#        StandardClient.create_factory(
#            model_class=LinearRegressionMultiDimSGD,
#            data=dataset,
#            model_parameters=prior_params,
#            model_hyperparameters={
#                "base_optimizer_class": torch.optim.SGD,
#                "wrapped_optimizer_class": StandardOptimizer,
#                "base_optimizer_parameters": {'lr': 0.02},
#                "wrapped_optimizer_parameters": {},
#                "N_steps": 10,
#                "n_in": 2,
#                "batch_size": x.shape[0],
#                "model_noise": 1
#            },
#            hyperparameters={
#                't_i_init_function': lambda x: np.zeros(x.shape)
#            }
#        )
    ]

    server = SyncronousPVIParameterServer(
        model_class=LinearRegressionMultiDimSGD,
        model_parameters=prior_params,
        model_hyperparameters={
            "base_optimizer_class": torch.optim.SGD,
            "wrapped_optimizer_class": None,
            "base_optimizer_parameters": {'lr': 0.01},
            "wrapped_optimizer_parameters": {},
            "N_steps": 500,
            "n_in": 2,
            "batch_size": x.shape[0],
            "model_noise": 1
        },
        prior=prior_params,
        clients_factories=clients,
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

    final_log = server.get_compiled_log()

    log_dir = '../logs/tests'

    log_dir = os.path.join(log_dir, f'{datetime.datetime.now()}')
    dump_file = os.path.join(log_dir, 'results.json')

    os.makedirs(log_dir)

    with open(dump_file, 'w') as f:
        json.dump(final_log, f, indent=4)

    ex.add_artifact(dump_file, 'full_log')
