import datetime
import json
import numbers
import os

import numpy as np
import ray
from sacred import Experiment
from sacred.observers import MongoObserver, SlackObserver

import src.utils.numpy_nest_utils as nest_utils
from src.client.client import DPClient
from src.model.linear_regression_models import LinearRegression1DAnalyticNumpy
from src.server.server import SynchronousParameterServer

ex = Experiment('PVI Test Experiment')

ex.observers.append(
    MongoObserver.create(
        url='localhost:9001',
        db_name='sacred'
    ))
ex.observers.append(
    SlackObserver.from_config('../slack.json')
)


def whiten_data(data_shards):
    x_mu = 0
    x_var = 0
    y_mu = 0
    y_var = 0
    N = 0
    for data in data_shards:
        x_mu += data['x'].sum(axis=0)
        y_mu += data['y'].sum(axis=0)
        N += data['x'].shape[0]

    x_mu = x_mu / N
    y_mu = y_mu / N

    for data in data_shards:
        x_var += ((data['x'] - x_mu) ** 2).sum(axis=0)
        y_var += ((data['x'] - y_mu) ** 2).sum(axis=0)

    x_sigma = np.sqrt(x_var / N)
    y_sigma = np.sqrt(y_var / N)

    for data in data_shards:
        data['x'] = (data['x'] - x_mu) / x_sigma
        data['y'] = (data['y'] - y_mu) / y_sigma

    return data_shards, [x_mu, y_mu, x_sigma, y_sigma]


def zero_init_function(parameter):
    return np.zeros(parameter.shape)


def no_privacy_function(update):
    return update, []


@ex.config
def config():
    num_clients = 10
    log_base_dir = '../logs'


@ex.automain
def main(
        log_base_dir,
        num_clients,
        _run,
):
    ray.init()

    data_gen_model = LinearRegression1DAnalyticNumpy(
        parameters={
            'slope_eta_1': np.array([5]),
            'slope_eta_2': np.array([-0.5])
        }
    )

    data_shards = []
    for i in range(num_clients):
        x = np.atleast_2d(np.random.normal(0, 1, 100)).T
        y = data_gen_model.sample(x)
        data_shards.append({'x': x,
                            'y': y})

    data_shards, stats = whiten_data(data_shards)

    all_x = np.concatenate([data['x'] for data in data_shards], axis=0)
    all_y = np.concatenate([data['y'] for data in data_shards], axis=0)

    exact_inference_parameters = {
        'slope_eta_1': (all_x.T @ all_x) / (1 / stats[3]),
        'slope_eta_2': (all_x.T @ all_y) / (1 / stats[3])
    }

    prior = {
        'slope_eta_1': np.array([0]),
        'slope_eta_2': np.array([-0.5])
    }

    clients = [
        DPClient.remote(
            model_class=LinearRegression1DAnalyticNumpy,
            data=data_shards[i],
            model_parameters=prior,
            model_hyperparameters={
                'model_noise': 1 / stats[3]
            },
            hyperparameters={
                't_i_init_function': zero_init_function,
                'privacy_function': no_privacy_function
            }
        )
        for i in range(num_clients)
    ]

    for client in clients: client.set_metadata.remote({'log_params': True})

    server = SynchronousParameterServer(
        model_class=LinearRegression1DAnalyticNumpy,
        prior=prior,
        clients=clients,
        max_iterations=20
    )

    while not server.should_stop():
        server.tick()

        iteration = server.iterations

        if iteration % 4 == 0:
            log = {}
            server_log = server.log_sacred()
            log['server'] = server_log
            client_logs = ray.get([client.log_sacred.remote() for client in clients])
            for i, client_log in enumerate(client_logs):
                log[f'client_{i}'] = client_log[0]

            log = nest_utils.flatten(log)

            for k, v in log.items():
                if isinstance(v, numbers.Number):
                    _run.log_scalar(k, v, iteration)

    log = {}
    server_log = server.get_log()
    log['server'] = server_log
    client_logs = ray.get([client.get_log.remote() for client in clients])
    log['clients'] = client_logs

    log_dir = os.path.join(log_base_dir, 'tests', str(datetime.datetime.now()))
    os.makedirs(log_dir)
    dump_file = os.path.join(log_dir, 'full_log.json')
    with open(dump_file, 'w') as file:
        json.dump(log, file)
    ex.add_artifact(dump_file, 'full_log')
