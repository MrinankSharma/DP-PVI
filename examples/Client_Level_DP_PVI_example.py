import datetime
import json
import os

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

import src.privacy.analysis.moment_accountant as moment_accountant
import src.utils.numpy_nest_utils as numpy_nest
from src.client import StandardClient
from src.model.linear_regression_models import LinearRegression1DAnalyticNumpy
from src.privacy.dp_query import NumpyGaussianDPQuery
from src.server import DPSequentialIndividualPVIParameterServer

ex = Experiment('Client Level DP-PVI Experiment')

ex.observers.append(
    MongoObserver.create(
        url='localhost:9001',
        db_name='test'
    )
)

# ex.observers.append(
#     SlackObserver.from_config('../slack.json')
# )

@ex.config
def cfg():
    N = 10
    noise_scale = 0.5
    M = 20
    L = 10


def gen_client_dataset(N, noise_scale):
    x = np.random.normal(0, 1, N)
    theta = 2
    y = x * theta
    return {"x": x, "y": y}


@ex.automain
def main(N, M, L, noise_scale, _run):
    clients_data = [gen_client_dataset(N, noise_scale) for i in range(M)]

    target_delta = 0.0001

    prior_params = {
        'slope_eta_1': np.array([0]),
        'slope_eta_2': np.array([1])
    }

    client_factories = [
        StandardClient.create_factory(
            model_class=LinearRegression1DAnalyticNumpy,
            data=clients_data[i],
            model_parameters=prior_params,
            model_hyperparameters={
                'model_noise': noise_scale
            },
            hyperparameters={
                't_i_init_function': lambda x: np.zeros(x.shape),
                'privacy_function': lambda x: x
            },
            metadata={
                "client_index": i,
            }
        ) for i in range(M)
    ]

    server = DPSequentialIndividualPVIParameterServer(
        model_class=LinearRegression1DAnalyticNumpy,
        dp_query_class=NumpyGaussianDPQuery,
        hyperparameters={
            "L": L,
            'dp_query_parameters': {
                'l2_norm_clip': 5000,
                'noise_stddev': 1e-10
            },
        },
        max_iterations=50,
        client_factories=client_factories,
        prior=prior_params,
        accounting_dict={
            'MomentAccountant': {
                'accountancy_update_method': moment_accountant.compute_online_privacy_from_ledger,
                'accountancy_parameters': {
                    'target_delta': target_delta
                }
            }
        }
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

    log_dir = '../logs/examples/client_example'

    log_dir = os.path.join(log_dir, f'{datetime.datetime.now()}')
    dump_file = os.path.join(log_dir, 'results.json')

    os.makedirs(log_dir)

    with open(dump_file, 'w') as f:
        json.dump(final_log, f, indent=4)

    ex.add_artifact(dump_file, 'full_log')
