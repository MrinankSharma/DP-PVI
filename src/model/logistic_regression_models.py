from src.model.model import Model

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

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

def params_to_nat_params(mean, log_var):
    pres = 1/torch.exp(log_var)
    nat_mean = mean * pres
    return nat_mean, pres

def nat_params_to_params(nat_mean, pres):
    log_var = torch.log(1/pres)
    mean = nat_mean / pres
    return mean, log_var

class MeanFieldMultiDimensionalLogisticRegression(nn.Module, Model):

    def __init__(self, parameters, hyperparameters, n_in):
        super(MeanFieldMultiDimensionalLogisticRegression, self).__init__()
        self.n_in = n_in
        self.prior_mean = hyperparameters["prior_mu"]
        self.prior_log_var = hyperparameters["prior_log_var"]

        # learnable params
        # initalise in random in box
        self.w_mu = nn.Parameter(torch.zeros(self.n_in, dtype=torch.float64).uniform_(-0.1, 0.1))
        self.w_log_var = nn.Parameter(torch.zeros(self.n_in, dtype=torch.float64).uniform_(-0.1, 0.1))

        # single linear layer + sigmoidal function --> logistic regression
        self.act = nn.Sigmoid()

        # standard normal
        self.normal_dist = Normal(loc=torch.tensor([0], dtype=torch.float64),
                                  scale=torch.tensor([1], dtype=torch.float64))

        self.learning_rate = 1
        self.N_samples = 50
        self.N_steps = 1000

    def set_hyperparameters(self, hyperparameters):
        if hyperparameters is not None:
            self.prior_mean = hyperparameters["prior_mu"]
            # note that the prior is a full matrix
            self.prior_log_var = hyperparameters["prior_log_var"]
            self.learning_rate = hyperparameters["learning_rate"]
            self.N_samples = hyperparameters["N_samples"]
            self.N_steps = hyperparameters["N_steps"]

    def set_parameters(self, parameters):
        if parameters is not None:
            self.w_mu = parameters["w_mu"]
            self.w_log_var = parameters["w_log_var"]

    def forward(self, X, N_samples, parameters=None, hyperparameters=None):
        """
        Apply forward pass, getting sampled activations and probabilities

        :param X: data tensor
        :param N_samples: number of samples to generate
        :param parameters: model parameters
        :param hyperparameters: model hyperparameters
        :return: Y: probability matrix, activation_mat: matrix of activation values (makes ELBO loss calculation easier)
        """
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
        """
        Compute the ELBO loss per training datapoint, given the activation matrix produced by the forward pass

        :param activation_mat: Matrix of activation functions. Each row is a data-point, each column is a sample
        :param Y_true: True labels
        :param parameters: Model parameters
        :param hyperparameters: Model hyperparameters
        :return: ELBO per point
        """
        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

        N_data = activation_mat.shape[0]
        N_samples = activation_mat.shape[1]

        # compute the KL term
        KL = compute_KL_qp(self.w_mu, torch.diag(torch.exp(self.w_log_var)), self.prior_mean,
                           torch.diag(torch.exp(self.prior_log_var)))

        likelihood = self.act(Y_true.repeat(N_samples, 1).t() * activation_mat)

        ELBO_per_point = 1 / N_samples * torch.einsum('ij->i', torch.log(likelihood)) - 1/N_data * KL

        return ELBO_per_point

    def compute_aggregated_ELBO(self, activation_mat, Y_true, parameters=None, hyperparameters=None):
        """
        Compute the ELBO loss, aggregated across all of the datapoints.

        :param activation_mat: Matrix of activation functions. Each row is a data-point, each column is a sample
        :param Y_true: True labels
        :param parameters: Model parameters
        :param hyperparameters: Model hyperparameters
        :return: ELBO
        """
        ELBO_pp = self.compute_ELBO_loss_per_point(activation_mat, Y_true, parameters, hyperparameters)
        ELBO = torch.sum(ELBO_pp)
        return ELBO

    def sample(self, x, parameters, hyperparameters):
        """
        Create some logistic regression data. *** NOTE: This ignores the precisions of each of the values of w, and
        simply assuming the true (unknown) weight is w; this is different to finding the predictive distribution!! ***

        :param x: input values to predict at
        :param parameters: model parameters (not will not update)
        :param hyperparameters: model hyperparameters (will also not update)
        :return: y: tensor of parameter labels
        """
        w_nat_means = parameters["w_mu"]

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

    def update_hyperparameters(self, updated_dict):
        for key, value in updated_dict.items():
            self.__dict__[key] = value

    def fit(self, data, t_i, parameters=None, hyperparameters=None):
        super().fit(data, t_i, parameters, hyperparameters)
        X = data["X"]
        y_true = data["y"]

        # compute the cavity distribution using natural parameters, and then convert back
        current_nat_mean, current_pres = params_to_nat_params(self.get_parameters()['w_mu'], self.get_parameters()['w_log_var'])
        t_i_nat_mean, t_i_pres = params_to_nat_params(t_i['w_mu'], t_i['w_log_var'])
        cav_mean, cav_log_var = nat_params_to_params(t_i_nat_mean + current_nat_mean, t_i_pres + current_pres)

        self.update_hyperparameters({
            "prior_mu": cav_mean,
            "prior_log_var": cav_log_var
        })

        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        # lets just do this for the time being
        for i in range(self.N_steps):
            optimizer.zero_grad()  # zero the gradient buffers
            _, act_mat = self.forward(X, self.N_samples)
            # we maximise the ELBO, so the loss is the negative of the ELBO
            loss = -self.compute_aggregated_ELBO(act_mat, y_true)
            # for some reason, it complains when we don't retain the graph. I don't understand why ...
            loss.backward(retain_graph=True)
            optimizer.step()  # Does the update

        return self.get_parameters()

    def get_parameters(self):
        return {
            'w_mu': self.w_mu,
            'w_log_var': self.w_log_var,
        }