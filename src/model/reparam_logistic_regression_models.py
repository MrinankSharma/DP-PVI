import logging

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

import src.utils.numpy_utils as B
from src.model.model import Model

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# note that these functions are all using NUMPY variables
def prediction_function(a, mean, var):
    return ((1 + np.exp(-a)) ** -1) * ((2 * np.pi * var) ** -0.5) * np.exp(-0.5 * (a - mean) ** 2 / var)


def params_to_nat_params(mean, stored_log_var):
    pres = 1 / np.exp(stored_log_var * 10)
    nat_mean = mean * pres
    return nat_mean, pres


def nat_params_to_params(nat_mean, pres):
    true_log_var = np.log(1 / pres)
    stored_log_var = true_log_var / 10
    mean = nat_mean / pres
    return mean, stored_log_var


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

class ReparamLogisticRegressionTorchModule(nn.Module):

    def __init__(self, parameters, hyperparameters):
        super(ReparamLogisticRegressionTorchModule, self).__init__()

        self.set_hyperparameters(hyperparameters)

        self.w_mu = nn.Parameter(torch.zeros(self.n_in, dtype=torch.float32))
        self.w_log_var = nn.Parameter(torch.zeros(self.n_in, dtype=torch.float32))

        # we make the assumption that the input parameters are actually a numpy array!
        self.set_parameters_from_numpy(parameters)
        self.act = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()

        # standard normal, for sampling required
        self.normal_dist = Normal(loc=torch.tensor([0], dtype=torch.float32),
                                  scale=torch.tensor([1], dtype=torch.float32))

        # dummy value to begin with!
        self.N_full = torch.tensor(1.0)
        self.prediction_type = "prohibit"

    def set_N_full(self, N_full):
        """
        Note - input assumed to be a numpy type
        :param N_full: int numpy
        """
        self.N_full = torch.tensor(N_full, dtype=torch.float32)

    def set_parameters(self, parameters):
        if parameters is not None:
            self.w_mu.data = parameters["w_mu"]
            self.w_log_var.data = parameters["w_log_var"]

    def set_parameters_from_numpy(self, parameters_numpy):
        if parameters_numpy is not None:
            self.w_mu.data = torch.tensor(parameters_numpy["w_mu"], dtype=torch.float32)
            self.w_log_var.data = torch.tensor(parameters_numpy["w_log_var"], dtype=torch.float32)

    def set_prior_parameters_from_numpy(self, prior_nat_params_numpy):
        logger.debug("setting prior params")
        if prior_nat_params_numpy is not None:
            prior_params = nat_params_to_params_dict(prior_nat_params_numpy)
            self.prior_mu = torch.tensor(prior_params["w_mu"], dtype=torch.float32)
            self.prior_log_var = torch.tensor(prior_params["w_log_var"], dtype=torch.float32)

    def set_hyperparameters(self, hyperparameters):
        """
        Convert the hyperparam dict into internal things
        :param hyperparameters:
        :return:
        """
        if hyperparameters is not None:
            self.n_in = hyperparameters['n_in']
            self.N_samples = hyperparameters["N_samples"]
            if hyperparameters["prediction"] == "prohibit" or hyperparameters["prediction"] == "exact":
                self.prediction_type = hyperparameters["prediction"]
            else:
                logger.error(f"{self.hyperparameters['prediction']} \
                is an invalid prediction setting! Please either use laplace or exact")
                raise ValueError("Invalid logistic regression prediction option")

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

        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

        # all per point
        mean_1d = torch.mv(x, self.w_mu)
        cov_mat = torch.diag(torch.exp(10 * self.w_log_var))
        cov_list = torch.diag(torch.mm(x, torch.mm(cov_mat, x.t())))
        mean_vars = list(zip(mean_1d, cov_list))
        p_vals = torch.empty(len(mean_vars))

        if self.prediction_type == "exact":
            for ind, mean_var in enumerate(mean_vars):
                mu = mean_var[0].detach().numpy()
                var = mean_var[1].detach().numpy()
                p_val = torch.tensor(self._prediction_func(mu, var), dtype=torch.float32)
                p_vals[ind] = p_val
        elif self.prediction_type == "prohibit":
            logger.debug("Using prohibit debug")
            p_vals = self.sigmoid(mean_1d * (1 + cov_list / 8 * np.pi) ** -0.5).detach()
        else:
            raise ValueError("Invalid prediction type supplied")

        return p_vals

    def forward(self, x, parameters=None):
        """
        Apply forward pass, getting sampled activations and probabilities

        :param x: data tensor
        :param N_samples: number of samples to generate
        :param parameters: model parameters
        :param hyperparameters: model hyperparameters
        :return: Y: probability matrix, activation_mat: matrix of activation values (makes ELBO loss calculation easier)
        """
        self.set_parameters(parameters)

        # we will apply the local re-parameterisation trick - we need samples of w^T x which has mean mu_w^T x and
        # covariance x^T sigma_w x
        mean_i = torch.mv(x, self.w_mu)
        cov_mat = torch.diag(torch.exp(10 * self.w_log_var))
        std_i = torch.pow(torch.diag(torch.mm(x, torch.mm(cov_mat, x.t()))), 0.5)

        # z = w^T x
        z_i = torch.flatten(self.normal_dist.sample([self.N_samples, ]))

        # each row corresponds to a input datapoint, each column is a sample
        activation_mat = torch.einsum('i,j->ij', std_i, z_i) + mean_i.repeat(self.N_samples, 1).t()

        return activation_mat

    def compute_ELBO_loss_per_point(self, y, y_true, parameters=None):
        """
        Compute the ELBO loss per training datapoint, given the activation matrix produced by the forward pass.

        NOTE: this currently depends on the number of points

        :param activation_mat: Matrix of activation input. Each row is a data-point, each column is a sample
        :param Y_true: True labels
        :param parameters: Model parameters
        :param hyperparameters: Model hyperparameters
        :return: ELBO per point
        """
        self.set_parameters(parameters)

        def compute_KL_qp(q_mean, q_log_var_diag, p_mean, p_log_var_diag):
            q_var = torch.diag(torch.exp(q_log_var_diag))
            k = q_mean.shape[0]
            p_inv = torch.exp(p_log_var_diag)
            p_inv = 1/ p_inv
            p_inv = torch.diag(p_inv)
            m1_m2 = p_mean - q_mean
            # note that we a-priori know that q_var is a diagonal matrix
            KL = 0.5 * (torch.trace(torch.mm(p_inv, q_var)) + torch.dot(m1_m2, torch.mv(p_inv, m1_m2)) - k + torch.sum(
                p_log_var_diag) - torch.sum(q_log_var_diag))
            return KL

        activation_mat = y

        L = activation_mat.shape[0]
        N_samples = activation_mat.shape[1]

        mask = y_true.repeat(N_samples, 1).t()

        # compute the KL term
        # scale by the TOTAL number of data points!
        KL_term = -1 / L * compute_KL_qp(self.w_mu, 10 * self.w_log_var, self.prior_mu, 10 * self.prior_log_var)

        likelihood = self.act(activation_mat * mask)

        likelihood_term = 1 / N_samples * torch.einsum('ij->i', likelihood) * self.N_full / L
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

class ReparamMeanFieldMultiDimensionalLogisticRegression(Model):

    def __init__(self, parameters, hyperparameters):
        super(ReparamMeanFieldMultiDimensionalLogisticRegression, self).__init__(parameters, hyperparameters)

        # logger.debug("New logistic regression module")
        self.torch_module = ReparamLogisticRegressionTorchModule(nat_params_to_params_dict(parameters), self.hyperparameters)

        if self.hyperparameters['wrapped_optimizer_class'] is not None:
            base_optimizer = self.hyperparameters['base_optimizer_class'](self.torch_module.parameters(),
                                                                          **hyperparameters[
                                                                              'base_optimizer_parameters'])

            self.wrapped_optimizer = self.hyperparameters['wrapped_optimizer_class'](optimizer=base_optimizer,
                                                                                     model=self.torch_module,
                                                                                     loss_per_example=self.torch_module.compute_ELBO_loss_per_point,
                                                                                     **hyperparameters[
                                                                                         'wrapped_optimizer_parameters']
                                                                                     )

        self._training_curves = []
        self._derived_statistics_histories = []

    def set_parameters(self, nat_parameters):
        if nat_parameters is not None:
            parameters = nat_params_to_params_dict(nat_parameters)
            try:
                self.torch_module.set_parameters_from_numpy(parameters)
            except AttributeError:
                # the torch module hasn't been setup yet - just skip for the time being!
                pass

    def get_parameters(self):
        try:
            return params_to_nat_params_dict({
                'w_mu': self.torch_module.w_mu.detach().numpy(),
                'w_log_var': self.torch_module.w_log_var.detach().numpy()
            })
        except AttributeError:
            # if this get's called
            logger.error("Attempted to get parameters before this object was properly initialised")
            return self.get_default_parameters()

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}
        # also update model hypers
        try:
            self.torch_module.set_hyperparameters(self.hyperparameters)
        except AttributeError:
            # skip if the torch module doesn't exist yet
            pass

    @classmethod
    def get_default_parameters(cls):
        return {
            'w_pres': np.array([1]),
            'w_nat_mean': np.array([0])
        }

    @classmethod
    def get_default_hyperparameters(cls):
        return {
            "base_optimizer_class": None,
            "wrapped_optimizer_class": None,
            "base_optimizer_params": {},
            "wrapped_optimizer_params": {},
            "N_steps": 1,
            "N_samples": 50,
            "n_in": 2,
            "prediction": "prohibit"
        }

    def sample(self, x, parameters=None, hyperparameters=None):
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

        x_tensor = torch.tensor(x, dtype=torch.float32)

        return self.torch_module.predict(x_tensor).numpy()

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

        x_full = data["x"]
        y_full = data["y"]
        if "N_full" in self.hyperparameters:
            N_full = self.hyperparameters['N_full']
        else:
            N_full = x_full.shape[0]

        cav_nat_params = B.subtract_params(self.get_parameters(), t_i)
        # numpy dict for the effective prior
        # logger.debug(f"current parameters {self.get_parameters()}")
        # logger.debug(f"current t_i {t_i}")
        # logger.debug(f"cavity natural parameters {cav_nat_params}")
        self.torch_module.set_prior_parameters_from_numpy(cav_nat_params)
        self.torch_module.set_N_full(N_full)

        base_optimizer = self.hyperparameters['base_optimizer_class'](self.torch_module.parameters(),
                                                                      **self.hyperparameters[
                                                                          'base_optimizer_parameters'])

        self.wrapped_optimizer = self.hyperparameters['wrapped_optimizer_class'](optimizer=base_optimizer,
                                                                                 model=self.torch_module,
                                                                                 loss_per_example=self.torch_module.compute_ELBO_loss_per_point,
                                                                                 **self.hyperparameters[
                                                                                     'wrapped_optimizer_parameters']
                                                                                 )

        print_interval = np.ceil(self.hyperparameters['N_steps'] / 20)

        training_curve = np.empty(self.hyperparameters['N_steps'])
        derived_statistics_history = []

        self._derived_statistics_history = []
        # lets just do this for the time being
        for i in range(self.hyperparameters['N_steps']):
            # sample minibatch at each step ...
            if self.hyperparameters["batch_size"] > data["x"].shape[0]:
                logger.debug('Had to reduce minibatch size to match data size.')
            batch_size = np.min([self.hyperparameters["batch_size"], data["x"].shape[0]])
            mini_batch_indices = np.random.choice(data["x"].shape[0], batch_size, replace=False)
            # convert data into a tensor
            x = torch.tensor(x_full[mini_batch_indices, :], dtype=torch.float32)
            y_true = torch.tensor(y_full[mini_batch_indices], dtype=torch.float32)

            current_loss = self.wrapped_optimizer.fit_batch(x, y_true)
            derived_statistics = self.wrapped_optimizer.get_logged_statistics()
            derived_statistics_history.append(derived_statistics)
            training_curve[i] = current_loss
            if i % print_interval == 0:
                logger.debug("Loss: {:.3f} after {} steps".format(current_loss, i))

        # logger.debug(f"updated parameters {self.get_parameters()}")
        # logger.debug("Learnt Moments")
        # logger.debug(f"mean: {self.torch_module.w_mu}")
        # logger.debug(f"var: {self.torch_module.w_log_var}")
        # logger.debug("**\n")
        # if several fit batches are called, this puts all of their training curves into a list
        self._training_curves.append(training_curve)
        self._derived_statistics_histories.append(derived_statistics_history)

        return self.get_parameters()

    def get_incremental_log_record(self):
        ret = {
            "derived_statistics": self._derived_statistics_histories,
            "training_curves": self._training_curves,
        }
        self._training_curves = []
        self._derived_statistics_histories = []
        return ret

    def get_incremental_sacred_record(self):
        # we don't want anything from the model to be displayed directly to sacred
        return {}
