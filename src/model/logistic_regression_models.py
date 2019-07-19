import logging

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch.nn as nn

import numpy as np

from scipy.integrate import quad

from src.model.model import Model
import src.utils.numpy_backend as B

logger = logging.getLogger(__name__)


# note that these functions are all using NUMPY variables
def prediction_function(a, mean, var):
    return ((1 + np.exp(-a)) ** -1) * ((2 * np.pi * var) ** -0.5) * np.exp(-0.5 * (a - mean) ** 2 / var)


def params_to_nat_params(mean, log_var):
    pres = 1 / np.exp(log_var)
    nat_mean = mean * pres
    return nat_mean, pres


def nat_params_to_params(nat_mean, pres):
    log_var = np.log(1 / pres)
    mean = nat_mean / pres
    return mean, log_var


def nat_params_to_params_dict(natural_params_dict):
    mean, log_var = nat_params_to_params(natural_params_dict['w_nat_mean'], natural_params_dict['w_pres'])
    return {
        "w_mu": mean,
        "w_log_var": log_var
    }


def params_to_nat_params_dict(params_dict):
    nat_mean, pres = params_to_nat_params(params_dict["w_mu"], params_dict["w_log_var"])
    return {
        "w_nat_mean": nat_mean,
        "w_pres": pres,
    }


class LogisticRegressionTorchModule(nn.Module):

    def __init__(self, parameters, hyperparameters):
        super(LogisticRegressionTorchModule, self).__init__()

        self.n_in = hyperparameters['n_in']

        self.w_mu = nn.Parameter(torch.zeros(self.n_in, dtype=torch.float64))
        self.w_log_var = nn.Parameter(torch.zeros(self.n_in, dtype=torch.float64))

        # we make the assumption that the input parameters are actually a numpy array!
        self.set_parameters_from_numpy(parameters)
        self.act = nn.Sigmoid()

        self.N_samples = hyperparameters["N_samples"]

        # standard normal, for sampling required
        self.normal_dist = Normal(loc=torch.tensor([0], dtype=torch.float64),
                                  scale=torch.tensor([1], dtype=torch.float64))

    def set_parameters(self, parameters):
        if parameters is not None:
            self.w_mu.data = parameters["w_mu"]
            self.w_log_var.data = parameters["w_log_var"]

    def set_parameters_from_numpy(self, parameters_numpy):
        if parameters_numpy is not None:
            self.w_mu.data = torch.tensor(parameters_numpy["w_mu"])
            self.w_log_var.data = torch.tensor(parameters_numpy["w_log_var"])

    def set_prior_parameters_from_numpy(self, prior_nat_params_numpy):
        if prior_nat_params_numpy is not None:
            prior_params = nat_params_to_params_dict(prior_nat_params_numpy)
            self.prior_mu = torch.tensor(prior_params["w_mu"])
            self.prior_log_var = torch.tensor(prior_params["w_log_var"])

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

    def forward(self, X, parameters=None):
        """
        Apply forward pass, getting sampled activations and probabilities

        :param X: data tensor
        :param N_samples: number of samples to generate
        :param parameters: model parameters
        :param hyperparameters: model hyperparameters
        :return: Y: probability matrix, activation_mat: matrix of activation values (makes ELBO loss calculation easier)
        """
        self.set_parameters(parameters)

        # we will apply the local re-parameterisation trick - we need samples of w^T x which has mean mu_w^T x and
        # covariance x^T sigma_w x
        mean_i = torch.mv(X, self.w_mu)
        cov_mat = torch.diag(torch.exp(self.w_log_var))
        std_i = torch.pow(torch.diag(torch.mm(X, torch.mm(cov_mat, X.t()))), 0.5)

        # z = w^T x
        z_i = torch.flatten(self.normal_dist.sample([self.N_samples, ]))

        # each row corresponds to a input datapoint, each column is a sample
        activation_mat = torch.einsum('i,j->ij', std_i, z_i) + mean_i.repeat(self.N_samples, 1).t()
        Y = self.act(activation_mat)

        return Y

    def compute_ELBO_loss_per_point(self, Y_pred, Y_true, parameters=None):
        """
        Compute the ELBO loss per training datapoint, given the activation matrix produced by the forward pass

        :param activation_mat: Matrix of activation functions. Each row is a data-point, each column is a sample
        :param Y_true: True labels
        :param parameters: Model parameters
        :param hyperparameters: Model hyperparameters
        :return: ELBO per point
        """
        self.set_parameters(parameters)

        def compute_KL_qp(q_mean, q_var, p_mean, p_var):
            k = q_mean.shape[0]
            p_inv = torch.inverse(p_var)
            m1_m2 = p_mean - q_mean
            KL = 0.5 * (torch.trace(torch.mm(p_inv, q_var)) + torch.dot(m1_m2, torch.mv(p_inv, m1_m2)) - k + np.log(
                torch.det(p_var)) - torch.log(torch.det(q_var)))
            return KL

        activation_mat = torch.log(Y_pred / (1 - Y_pred))

        N_data = activation_mat.shape[0]
        N_samples = activation_mat.shape[1]

        # compute the KL term
        KL_term = -1 / N_data * compute_KL_qp(self.w_mu, torch.diag(torch.exp(self.w_log_var)), self.prior_mu,
                                              torch.diag(torch.exp(self.prior_log_var)))

        likelihood = self.act(Y_true.repeat(N_samples, 1).t() * activation_mat)

        likelihood_term = 1 / N_samples * torch.einsum('ij->i', torch.log(likelihood))
        ELBO_per_point = likelihood_term + KL_term

        # we call the ELBO loss the negative of the elbo (we maximise the ELBO)
        return -ELBO_per_point

    def sample(self, x, parameters):
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
        p = self.act(z)

        output_dist = Bernoulli(p)
        y = output_dist.sample()
        return y


class MeanFieldMultiDimensionalLogisticRegression(Model):

    def __init__(self, parameters, hyperparameters):
        # initialise torch module first
        self.torch_module = LogisticRegressionTorchModule(nat_params_to_params_dict(parameters), hyperparameters)
        super(MeanFieldMultiDimensionalLogisticRegression, self).__init__(parameters, hyperparameters)

        base_optimizer = self.hyperparameters['base_optimizer_class'](self.torch_module.parameters(),
                                                                      **hyperparameters['base_optimizer_parameters'])

        self.wrapped_optimizer = self.hyperparameters['wrapped_optimizer_class'](optimizer=base_optimizer,
                                                                                 model=self.torch_module,
                                                                                 loss_per_example=self.torch_module.compute_ELBO_loss_per_point,
                                                                                 **hyperparameters[
                                                                                     'wrapped_optimizer_parameters'])

    def set_hyperparameters(self, hyperparameters):
        if hyperparameters is not None:
            self.hyperparameters = hyperparameters

    def set_parameters(self, nat_parameters):
        if nat_parameters is not None:
            parameters = nat_params_to_params_dict(nat_parameters)
            self.torch_module.set_parameters_from_numpy(parameters)

    def get_parameters(self):
        return params_to_nat_params_dict({
            'w_mu': self.torch_module.w_mu.detach().numpy(),
            'w_log_var': self.torch_module.w_log_var.detach().numpy()
        })

    def get_default_parameters(self):
        return {
            'w_pres': 1,
            'w_nat_mean': 0
        }

    def get_default_hyperparameters(self):
        return {
            "base_optimizer_class": None,
            "wrapped_optimizer_class": None,
            "base_optimizer_params": {},
            "wrapped_optimizer_params": {},
            "N_steps": 1,
            "N_samples": 50,
            "n_in": 2,
            "prediction_integration_limit": 50,
        }

    def sample(self, x, parameters):
        """
        Create some logistic regression data. *** NOTE: This ignores the precisions of each of the values of w, and
        simply assuming the true (unknown) weight is w; this is different to finding the predictive distribution!! ***

        :param x: input values to predict at
        :param parameters: model parameters (not will not update)
        :param hyperparameters: model hyperparameters (will also not update)
        :return: y: tensor of parameter labels
        """
        return self.torch_module.sample(x, parameters).numpy()

    def predict(self, x, parameters=None, hyperparameters=None):
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

        return self.torch_module.predict(x, self.hyperparameters["prediction_integration_limit"], parameters).numpy()

    def fit(self, data, t_i, parameters=None, hyperparameters=None):
        """
        Note that this method will start the gradient optimisation from the current parameter values

        :param data:
        :param t_i:
        :param parameters:
        :param hyperparameters:
        :return: optimum params
        """
        super().fit(data, t_i, parameters, hyperparameters)

        # convert data into a tensor
        X = torch.tensor(data["X"], dtype=torch.float64)
        y_true = torch.tensor(data["y"], dtype=torch.float64)

        cav_nat_params = B.subtract_params(self.get_parameters(), t_i)
        # numpy dict for the effective prior
        self.torch_module.set_prior_parameters_from_numpy(cav_nat_params)

        print_interval = np.ceil(self.hyperparameters['N_steps'] / 20)

        training_array = np.empty(self.hyperparameters['N_steps'])
        # lets just do this for the time being
        for i in range(self.hyperparameters['N_steps']):
            current_loss = self.wrapped_optimizer.fit_batch(X, y_true)
            training_array[i] = current_loss
            if i % print_interval == 0:
                print("Loss: {:.3f} after {} steps".format(current_loss, i))

        return self.get_parameters(), training_array