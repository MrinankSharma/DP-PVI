from src.model.model import Model

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from scipy.integrate import quad


def prediction_function(a, mean, var):
    return ((1 + np.exp(-a)) ** -1) * ((2 * np.pi * var) ** -0.5) * np.exp(-0.5 * (a - mean) ** 2 / var)


def compute_KL_qp(q_mean, q_var, p_mean, p_var):
    k = q_mean.shape[0]
    p_inv = torch.inverse(p_var)
    m1_m2 = p_mean - q_mean
    KL = 0.5 * (torch.trace(torch.mm(p_inv, q_var)) + torch.dot(m1_m2, torch.mv(p_inv, m1_m2)) - k + torch.log(
        torch.det(p_var)) - torch.log(torch.det(q_var)))
    return KL


class MeanFieldMultiDimensionalLogisticRegression(nn.Module, Model):

    def __init__(self, parameters, hyperparameters, n_in):
        super(MeanFieldMultiDimensionalLogisticRegression, self).__init__()
        self.n_in = n_in
        self.prior_mean = hyperparameters["prior_mu"]
        self.prior_log_var = hyperparameters["prior_log_var"]

        # learnable params
        # initalise in random in box
        self.w_mu = nn.Parameter(torch.Tensor(self.n_in).uniform_(-0.1, 0.1))
        self.w_log_var = nn.Parameter(torch.Tensor(self.n_in).uniform_(-0.1, 0.1))

        # single linear layer + sigmoidal function --> logistic regression
        self.act = nn.Sigmoid()

        # standard normal
        self.normal_dist = Normal(loc=torch.tensor([0], dtype=torch.float64),
                                  scale=torch.tensor([1], dtype=torch.float64)

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)

        if hyperparameters is not None:
            self.prior_mean = hyperparameters["prior_mu"]
            # note that the prior is a full matrix
            self.prior_log_var_matrix = hyperparameters["prior_log_var"]

    def set_parameters(self, parameters):
        super().set_parameters(parameters)

        if parameters is not None:
            self.w_mu = parameters["w_mu"]
            self.w_log_var = parameters["w_log_var"]

    def forward(self, X, N_samples, parameters=None, hyperparameters=None):
        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

        # we will apply the local re-parameterisation trick - we need samples of w^T x which has mean mu_w^T x and
        # covariance x^T sigma_w x
        mean_i = torch.mv(X, self.w_mu)
        cov_mat = torch.diag(torch.exp(self.w_log_var))
        std_i = torch.pow(torch.diag(torch.mm(X, torch.mm(cov_mat, X.t()))), 0.5)

        # z = w^T x
        z_i = torch.flatten(self.normal_dist.sample([N_samples, ]))

        # each row corresponds to a input datapoint, each column is a sample
        activation_mat = torch.einsum('i,j->ij', std_i, z_i) + mean_i.repeat(N_samples, 1).t()
        Y = self.act(activation_mat)

        return Y, activation_mat

    def compute_ELBO_loss_per_point(self, activation_mat, Y_true, parameters=None, hyperparameters=None):
        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

        N_data = activation_mat.size[0]
        N_samples = activation_mat.size[1]

        # compute the KL term
        KL = 1/N_data * compute_KL_qp(self.w_mu, torch.diag(torch.exp(self.w_log_var)), self.prior_mean,
                           torch.exp(self.prior_log_var_matrix))

        likelihood = self.act(Y_true.repeat(N_samples, 1).t() * activation_mat)
        average_log_likelihood = 1/N_samples * torch.einsum('ij->i', torch.log(likelihood)) + KL

        return average_log_likelihood


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
