import logging

import numpy as np
import numpy.random as random

import src.utils.numpy_utils as B
from src.model.model import Model

logger = logging.getLogger(__name__)


class LinearRegression1DAnalyticNumpy(Model):

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)
        self.noise = self.hyperparameters['model_noise']

    def fit(self, data, t_i, parameters=None, hyperparameters=None):
        super().fit(data, t_i, parameters, hyperparameters)

        x = data['x']
        y = data['y']

        logger.debug(f"Computing Update:"
                    f"  t_i: {t_i}\n"
                    f"  lambda_old: {parameters}\n"
                    f"  slope_eta_1: {(x.T @ x) / (self.noise ** 2)}\n"
                    f"  slope_eta_2: {(x.T @ y) / (self.noise ** 2)}\n\n")

        return B.add_parameters(
            B.subtract_params(
                self.parameters,
                t_i),
            {
                'slope_eta_1': (x.T @ x) / (self.noise ** 2),
                'slope_eta_2': (x.T @ y) / (self.noise ** 2)
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

    @classmethod
    def get_default_parameters(cls):
        return {
            'slope_eta_1': np.zeros([1]),
            'slope_eta_2': np.zeros([1])
        }

    @classmethod
    def get_default_hyperparameters(cls):
        return {
            'model_noise': 0
        }

    def get_incremental_sacred_record(self):
        return {}

    def get_incremental_log_record(self):
        return {}
