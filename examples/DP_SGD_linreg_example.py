import datetime
import json
import os

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver, SlackObserver

import src.privacy.analysis.moment_accountant as moment_accountant
import src.utils.numpy_nest_utils as numpy_nest
from src.client import DPClient, StandardClient
from src.model.linear_regression_models import LinearRegressionMultiDimSGD
from src.privacy.dp_query import GaussianDPQuery
from src.privacy.optimizer import DPOptimizer, StandardOptimizer
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
    batch_size = 30
    learning_rate = 0.03
    N_steps = 100
    epochs = 1000
    model_noise = 3
    privacy = {
        "l2_norm_clip": 5,
        "noise_stddev": 5
    }


@ex.automain
def main(N, batch_size, learning_rate, N_steps, epochs, model_noise, privacy, _run):
    true_params = np.array([5, -2], dtype=np.float32)
    x = np.ones((N, 2))
    x[:, 1] = np.random.normal(0, 1, N)
    y = x @ true_params + np.random.normal(0, model_noise, N)

    dataset = {
        'x': x,
        'y': y
    }

    target_delta = 1e-4

    prior_params = {
        "w_nat_mean": np.array([0.0, 0.0], dtype=np.float64),
        "w_pres": np.array([1.0, 1.0], dtype=np.float64)
    }

    prior_sigma = 1
    exact_infer_var = np.linalg.inv((x.T @ x) / model_noise ** 2 + np.identity(2) / prior_sigma ** 2)
    exact_infer_mean = (exact_infer_var @ x.T @ y) / model_noise ** 2
    exact_infer_params = {
        'mean': exact_infer_mean,
        'var': exact_infer_var
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
                }
            },
            data=dataset,
            model_parameters=prior_params,
            model_hyperparameters={
                "base_optimizer_class": torch.optim.SGD,
                "wrapped_optimizer_class": DPOptimizer,
                "base_optimizer_parameters": {'lr': learning_rate},
                "wrapped_optimizer_parameters": {},
                "N_steps": N_steps,
                "n_in": 2,
                "batch_size": batch_size,
                "model_noise": model_noise
            },
            hyperparameters={
                'dp_query_parameters': {
                    'l2_norm_clip': privacy['l2_norm_clip'],
                    'noise_stddev': privacy['noise_stddev']
                },
                't_i_init_function': lambda x: np.zeros(x.shape)
            }
        )
        # StandardClient.create_factory(
        #     model_class=LinearRegressionMultiDimSGD,
        #     data=dataset,
        #     model_parameters=prior_params,
        #     model_hyperparameters={
        #        "base_optimizer_class": torch.optim.SGD,
        #        "wrapped_optimizer_class": StandardOptimizer,
        #        "base_optimizer_parameters": {'lr': learning_rate},
        #        "wrapped_optimizer_parameters": {},
        #        "N_steps": N_steps,
        #        "n_in": 2,
        #        "batch_size": batch_size,
        #        "model_noise": model_noise
        #     },
        #     hyperparameters={
        #        't_i_init_function': lambda x: np.zeros(x.shape)
        #     }
        # )
    ]

    server = SyncronousPVIParameterServer(
        model_class=LinearRegressionMultiDimSGD,
        model_parameters=prior_params,
        model_hyperparameters={
            "base_optimizer_class": torch.optim.SGD,
            "wrapped_optimizer_class": None,
            "base_optimizer_parameters": {'lr': learning_rate},
            "wrapped_optimizer_parameters": {},
            "N_steps": 0,
            "n_in": 2,
            "batch_size": batch_size,
            "model_noise": model_noise
        },
        prior=prior_params,
        clients_factories=clients,
        max_iterations=epochs
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

        if sacred_log['client_0.MomentAccountant.epsilon'] > 100:
            break

    final_log = server.get_compiled_log()
    final_log['exact_inference'] = numpy_nest.structured_ndarrays_to_lists(exact_infer_params)

    log_dir = '../logs/tests'

    log_dir = os.path.join(log_dir, f'{datetime.datetime.now()}')
    dump_file = os.path.join(log_dir, 'results.json')

    os.makedirs(log_dir)

    with open(dump_file, 'w') as f:
        json.dump(final_log, f, indent=4)

    ex.add_artifact(dump_file, 'full_log')
