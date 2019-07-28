import numpy as np
import numpy.random as random
import torch
import torch.nn as nn

import src.utils.numpy_utils as B
from src.model.model import Model
from src.model.logistic_regression_models import params_to_nat_params, nat_params_to_params, nat_params_to_params_dict, params_to_nat_params_dict


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
                'w_precision': np.diag(np.diag(x.T @ x)) / (-2 * self.noise ** 2)
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


class LinearRegressionTorchModule(nn.Module):

    def __init__(self, parameters, hyperparameters):
        super(LinearRegressionTorchModule, self).__init__()

        self.n_in = hyperparameters['n_in']
        self.noise = self.hyperparameters['model_noise']

        self.w_mu = nn.Parameter(torch.zeros(self.n_in, dtype=torch.float32))
        self.w_log_var = nn.Parameter(torch.zeros(self.n_in, dtype=torch.float32))

        # we make the assumption that the input parameters are actually a numpy array!
        self.set_parameters_from_numpy(parameters)

    def set_parameters(self, parameters):
        if parameters is not None:
            self.w_mu.data = parameters["w_mu"]
            self.w_log_var.data = parameters["w_log_var"]

    def set_parameters_from_numpy(self, parameters_numpy):
        if parameters_numpy is not None:
            self.w_mu.data = torch.tensor(parameters_numpy["w_mu"], dtype=torch.float32)
            self.w_log_var.data = torch.tensor(parameters_numpy["w_log_var"], dtype=torch.float32)

    def set_prior_parameters_from_numpy(self, prior_nat_params_numpy):
        if prior_nat_params_numpy is not None:
            prior_params = nat_params_to_params_dict(prior_nat_params_numpy)
            self.prior_mu = torch.tensor(prior_params["w_mu"], dtype=torch.float32)
            self.prior_log_var = torch.tensor(prior_params["w_log_var"], dtype=torch.float32)

    def predict(self, x, integration_limit=500, parameters=None):
        """
        Bayesian prediction on probability at certain point, performed using numerical integration
        so this is the 'fully Bayesian' probability

        :param x: data to predict on
        :param integration_limit: how much to truncate numerical integration for the predictive distribution
        :param parameters: logistic regression parameters
        :param hyperparameters: logistic regression hyperparams
        :return: probability of each point in x being +1
        """

        self.set_parameters(parameters)
        # all per point
        mean_1d = torch.mv(x, self.w_mu)
        cov_mat = torch.diag(torch.exp(self.w_log_var))
        cov_list = torch.diag(torch.mm(x, torch.mm(cov_mat, x.t())))
        mean_vars = list(zip(mean_1d, cov_list))
        p_vals = torch.Tensor(len(mean_vars))

        for ind, mean_var in enumerate(mean_vars):
            p_val, err = quad(prediction_function, -integration_limit, integration_limit, mean_var)
            p_vals[ind] = p_val

        return p_vals

    def forward(self, x, parameters=None):
        """
        Apply forward pass, getting sampled activations and probabilities

        :param x: data tensor
        :param parameters: model parameters
        :param hyperparameters: model hyperparameters
        :return: Y: probability matrix, activation_mat: matrix of activation values (makes ELBO loss calculation easier)
        """
        self.set_parameters(parameters)

        return x

    def compute_loss_per_point(self, x, y, parameters=None):
        """
        Compute the ELBO loss per training datapoint, given the activation matrix produced by the forward pass

        :param activation_mat: Matrix of activation input. Each row is a data-point, each column is a sample
        :param Y_true: True labels
        :param parameters: Model parameters
        :param hyperparameters: Model hyperparameters
        :return: ELBO per point
        """
        self.set_parameters(parameters)

        # compute the differential entropy term
        diff_entropy = torch.log(torch.prod(torch.exp(self.w_log_var))) / 2

        # compute the likelihood term
        likelihood_term = []

        for i in x.shape[0]:
            likelihood_term.append((x[i] ** 2) @ torch.exp(self.w_log_var) + (x[i] @ self.w_mu) ** 2 - 2 * y[i] * x[i] @ self.w_mu)

        likelihood_term = torch.cat(likelihood_term) / (-2 * self.noise ** 2)
        
        # compute the prior term
        prior_var_inv = 1 / torch.exp(self.prior_log_var)
        prior_term = (prior_var_inv @ torch.exp(self.w_log_var) + (self.w_mu ** 2) @ prior_var_inv - 2 * (prior_mu * prior_var_inv) @ self.w_mu) / -2

        # compute loss per point
        loss_per_point = likelihood_term + (diff_entropy + prior_term) / x.shape[0]

        return -loss_per_point

    def sample(self, x, parameters):
        """
        Create some linear regression data. *** NOTE: This ignores the precisions of each of the values of w, and
        simply assuming the true (unknown) weight is w; this is different to finding the predictive distribution!! ***

        :param x: input values to predict at
        :param parameters: model parameters (not will not update)
        :param hyperparameters: model hyperparameters (will also not update)
        :return: y: tensor of predictions
        """
        w_mu = parameters["w_mu"]
        
        return torch.mv(x, w_mu) + random.normal(0, self.noise, x.shape[0])
