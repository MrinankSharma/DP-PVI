from src.model.model import Model

import torch
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from scipy.integrate import quad

def prediction_function(a, mean, var):
    return ((1 + np.exp(-a))**-1) * ((2*np.pi*var)**-0.5) * np.exp(-0.5*(a-mean)**2/var)

class MeanFieldMultiDimensionalLogisticRegression(nn.Module, Model):

    def __init__(self, parameters, hyperparameters, n_in):
        super(MeanFieldMultiDimensionalLogisticRegression, self).__init__()
        self.n_in = n_in
        self.prior_mean = hyperparameters["prior_mu"]
        self.prior_log_var = hyperparameters["prior_log_var"]

        # learnable params
        # initalise in random in box
        l = torch.nn.Linear(n_in, 1)
        self.w_mu = nn.Parameter(torch.Tensor(self.n_in).uniform_(-0.1, 0.1))
        self.w_log_var = nn.Parameter(torch.Tensor(self.n_in).uniform_(-0.1, 0.1))

        # single linear layer + sigmoidal function --> logistic regression
        self.act = nn.Sigmoid()

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)
        self.prior_mean = hyperparameters["prior_mu"]
        self.prior_log_var = hyperparameters["prior_log_var"]

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.w_mu = parameters["w_mu"]
        self.w_log_var = parameters["w_log_var"]

    def sample(self, x, parameters, hyperparameters):
        """
        Create some logistic regression data. *** NOTE: This ignores the precisions of each of the values of w, and
        simply assuming the true (unknown) weight is w; this is different to finding the predictive distribution!! ***

        :param x: input values to predict at
        :param parameters: model parameters (not will not update)
        :param hyperparameters: model hyperparameters (will also not update)
        :return: y: tensor of parameter labels
        """
        w_nat_means = parameters["w_nat_means"]

        z = torch.mv(x, w_nat_means)
        sigmoid = torch.nn.Sigmoid()
        p = sigmoid(z)

        dist = Bernoulli(p)
        y = dist.sample()
        return y

    def predict(self, x, integration_limit=500, parameters=None, hyperparameters=None):
        """
        Bayesian prediction on probability at certain point, performed using numerical integration
        so this is the 'fully Bayesian' probability

        :param x: data to predict on
        :param integration_limit: how much to truncate numerical integration for the predictive distribution
        :param parameters: logistic regression parameters
        :param hyperparameters: logistic regression hyperparams
        :return: probability of each point in x being +1
        """
        super().predict(x, parameters, hyperparameters)

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

    def get_default_parameters(self):
        return {
            'w_mu': torch.tensor([0]),
            'w_log_var': torch.tensor([0])
        }
