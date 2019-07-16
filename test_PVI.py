import ray
import numpy as np

from src.server.server import SyncronousPVIParameterServer
from src.client.client import DPClient
from src.model.linear_regression_models import LinearRegression1DAnalyticNumpy

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


if __name__ == '__main__':
    ray.init()

    num_clients = 10

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
        'slope_eta_1': (all_x.T @ all_x) / (1/stats[3]),
        'slope_eta_2': (all_x.T @ all_y) / (1/stats[3])
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
                'model_noise': 1/stats[3]
            },
            hyperparameters={
                't_i_init_function': zero_init_function,
                'privacy_function': no_privacy_function
            }
        )
        for i in range(num_clients)
    ]

    server = SyncronousPVIParameterServer(
        model_class=LinearRegression1DAnalyticNumpy,
        prior=prior,
        clients=clients,
        max_iterations=3
    )

    while not server.should_stop():
        print(f'Iteration {server.iterations}')
        print(f'Exact inference values \t {exact_inference_parameters}')
        print(f'PVI Values \t {server.parameters}')

        server.tick()