import numpy as np
import numpy.random as random

import src.utils.numpy_utils as B
from src.model.model import Model


class LinearRegression1DAnalyticNumpy(Model):

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
                'slope_eta_1': (x.T @ x) / self.noise,
                'slope_eta_2': (x.T @ y) / self.noise
            }
        )

    def predict(self, x, parameters=None, hyperparameters=None):
        super().predict(x, parameters, hyperparameters)

    def sample(self, x, parameters=None, hyperparameters=None):
        eta_1 = self.parameters['slope_eta_1']
        eta_2 = self.parameters['slope_eta_2']

        mu = -eta_1 / (2 * eta_2)
        sigma = 1 / (2 * eta_2)

        return x * mu + random.normal(0, self.noise, x.shape)

    @staticmethod
    def get_default_parameters():
        return {
            'slope_eta_1': np.zeros([1]),
            'slope_eta_2': np.zeros([1])
        }

    @staticmethod
    def get_default_hyperparameters():
        return {
            'model_noise': 0
        }

    def get_incremental_sacred_record(self):
        return {}

    def get_incremental_log_record(self):
        return {}
