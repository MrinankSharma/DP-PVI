import numpy as np
import numpy.random as random

import src.utils.numpy_utils as B
from src.model.model import Model


class LinearRegression1DAnalyticNumpy(Model):

    def get_default_parameters(self):
        return {
            'w_nat_mean': np.zeros(1),    # natural mean = mu / sigma squared
            'w_precision': np.zeros(1)    # precision = -1 / (2 * sigma squared)
        }

    def get_default_hyperparameters(self):
        return {
            'model_noise': 0    # model_noise is standard deviation
        }

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)
        self.noise = self.hyperparameters['model_noise']

    def fit(self, data, t_i, parameters=None, hyperparameters=None):
        super().fit(data, t_i, parameters, hyperparameters)

        x = data['x']
        y = data['y']

        return B.add_parameters(
            B.subtract_params(
                self.parameters,
                t_i),
            {
                'w_nat_mean': x.T @ y / (self.noise ** 2),
                'w_precision': x.T @ x / (-2 * self.noise ** 2)
            }
        )

    def predict(self, x, parameters=None, hyperparameters=None):
        super().predict(x, parameters, hyperparameters)

    def sample(self, x, parameters=None, hyperparameters=None):
        eta_1 = self.parameters['w_nat_mean']
        eta_2 = self.parameters['w_precision']

        mu = -eta_1 / (2 * eta_2)
        sigma_squared = -1 / (2 * eta_2)

        return x * mu + random.normal(0, self.noise, x.shape)

    def get_incremental_sacred_record(self):
        return {}

    def get_incremental_log_record(self):
        return {}


class LinearRegressionMultiDimAnalyticNumpy(Model):

    def get_default_parameters(self):
        return {
            'w_nat_mean': np.zeros((2, 1)),    # natural mean = sigma inverse * mu
            'w_precision': np.zeros((2, 2))    # precision = sigma inverse / -2
        }

    def get_default_hyperparameters(self):
        return {
            'dimension': 2,
            'model_noise': 0    # model_noise is standard deviation
        }

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)
        self.dim = self.hyperparameters['dimension']
        self.noise = self.hyperparameters['model_noise']

    def fit(self, data, t_i, parameters=None, hyperparameters=None):
        super().fit(data, t_i, parameters, hyperparameters)

        x = data['x']
        y = data['y']

        return B.add_parameters(
            B.subtract_params(
                self.parameters,
                t_i),
            {
                'w_nat_mean': x.T @ y / (self.noise ** 2),
                'w_precision': x.T @ x / (-2 * self.noise ** 2)
            }
        )

    def predict(self, x, parameters=None, hyperparameters=None):
        super().predict(x, parameters, hyperparameters)

    def sample(self, x, parameters=None, hyperparameters=None):
        eta_1 = self.parameters['w_nat_mean']
        eta_2 = self.parameters['w_precision']

        mu = np.linalg.inv(eta_2) @ eta_1 / -2
        sigma = np.linalg.inv(eta_2) / -2

        return x @ mu + random.normal(0, self.noise, (x.shape[0], 1))

    def get_incremental_sacred_record(self):
        return {}

    def get_incremental_log_record(self):
        return {}
