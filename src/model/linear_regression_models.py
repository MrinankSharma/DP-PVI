import numpy as np
import numpy.random as random
import torch
import torch.nn as nn
import logging

import src.utils.numpy_utils as B
from src.model.model import Model
from src.model.logistic_regression_models import params_to_nat_params, nat_params_to_params, nat_params_to_params_dict, params_to_nat_params_dict

logger = logging.getLogger(__name__)

class LinearRegression1DAnalyticNumpy(Model):

    def get_default_parameters(self):
        return {
            'w_nat_mean': 0,    # natural mean = mean / var
            'w_pres': 1         # precision = 1 / var
        }

    def get_default_hyperparameters(self):
        return {
            'model_noise': 1    # model_noise is standard deviation
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
                'w_nat_mean': (x.T @ y) / self.noise ** 2,
                'w_pres': (x.T @ x) / self.noise ** 2
            }
        )

    def predict(self, x, parameters=None, hyperparameters=None):
        super().predict(x, parameters, hyperparameters)

    def sample(self, x, parameters=None, hyperparameters=None):
        nat_mean = self.parameters['w_nat_mean']
        pres = self.parameters['w_pres']

        mean = nat_mean / pres
        var = 1 / pres

        return x * mean + random.normal(0, self.noise, x.shape)

    def get_incremental_sacred_record(self):
        return {}

    def get_incremental_log_record(self):
        return {}


class LinearRegressionMultiDimAnalyticNumpy(Model):

    def get_default_parameters(self):
        return {
            'w_nat_mean': np.zeros(2),    # natural mean = var inverse * mu
            'w_pres': np.ones(2)          # precision = var inverse
        }

    def get_default_hyperparameters(self):
        return {
            'n_in': 2,
            'model_noise': 1    # model_noise is standard deviation
        }

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)
        self.n_in = self.hyperparameters['n_in']
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
                'w_nat_mean': (x.T @ y) / self.noise ** 2,
                'w_pres': np.diag(x.T @ x) / self.noise ** 2
            }
        )

    def predict(self, x, parameters=None, hyperparameters=None):
        super().predict(x, parameters, hyperparameters)

    def sample(self, x, parameters=None, hyperparameters=None):
        nat_mean = self.parameters['w_nat_mean']
        pres = self.parameters['w_pres']

        mean = nat_mean / pres
        var = 1 / pres

        return x @ mean + random.normal(0, self.noise, x.shape[0])

    def get_incremental_sacred_record(self):
        return {}

    def get_incremental_log_record(self):
        return {}


class LinearRegressionTorchModule(nn.Module):

    def __init__(self, parameters, hyperparameters):
        super(LinearRegressionTorchModule, self).__init__()

        self.n_in = hyperparameters['n_in']
        self.noise = hyperparameters['model_noise']

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

    def predict(self, x, parameters=None):
        """
        Bayesian prediction on probability at certain point, where the predictive distribution is Gaussian
        so this is the 'fully Bayesian' probability

        :param x: data tensor. Each row is a data-point
        :param parameters: linear regression parameters
        :param hyperparameters: linear regression hyperparams
        :return: mean and variance of the predictive distribution of each y
        """
        self.set_parameters(parameters)
        
        y_mean = x @ self.w_mu
        y_var = (x ** 2) @ torch.exp(self.w_log_var) + self.noise ** 2

        return y_mean, y_var

    def forward(self, x, parameters=None):
        """
        Apply forward pass, return mean and variance of the predictive distribution and x,
        since we need x to maximize local free energy

        :param x: data tensor. Each row is a data-point
        :param parameters: model parameters
        :param hyperparameters: model hyperparameters
        :return: mean and variance of the predictive distribution of each y, as well as x
        """
        self.set_parameters(parameters)

        y_mean, y_var = self.predict(x)
        return y_mean, y_var, x

    def compute_loss_per_point(self, y_pred, y, parameters=None):
        """
        Compute the negative local free energy per training datapoint, given the training data tensor x and y

        :param y_pred: y_mean, y_var, x. Each row of x is a data-point
        :param y: training data tensor
        :param parameters: Model parameters
        :param hyperparameters: Model hyperparameters
        :return: negative local free energy per point
        """
        self.set_parameters(parameters)

        y_mean, y_var, x = y_pred

        # compute the differential entropy term
        diff_entropy = torch.log(torch.prod(torch.exp(self.w_log_var))) / 2

        # compute the likelihood term
        likelihood_term = ((x ** 2) @ torch.exp(self.w_log_var) + (x @ self.w_mu) ** 2 - 2 * (y.view(y.shape[0], 1) * x) @ self.w_mu) / (-2 * self.noise ** 2)
        
        # compute the prior term
        prior_var_inv = 1 / torch.exp(self.prior_log_var)
        prior_term = (prior_var_inv @ torch.exp(self.w_log_var) + (self.w_mu ** 2) @ prior_var_inv - 2 * (self.prior_mu * prior_var_inv) @ self.w_mu) / -2

        # compute loss per point
        loss_per_point = likelihood_term + (diff_entropy + prior_term) / x.shape[0]

        return -loss_per_point

    def sample(self, x, parameters=None):
        """
        Create some linear regression data. *** NOTE: This ignores the precisions of each of the values of w, and
        simply assuming the true (unknown) weight is w; this is different to finding the predictive distribution!! ***

        :param x: input values to predict at
        :param parameters: model parameters (not will not update)
        :param hyperparameters: model hyperparameters (will also not update)
        :return: y: tensor of predictions
        """
        self.set_parameters(parameters)
        
        return x @ self.w_mu.data + random.normal(0, self.noise, x.shape[0])


class LinearRegressionMultiDimSGD(Model):

    def __init__(self, parameters, hyperparameters):
        # initialise torch module first
        self.torch_module = LinearRegressionTorchModule(nat_params_to_params_dict(parameters), hyperparameters)
        super(LinearRegressionMultiDimSGD, self).__init__(parameters, hyperparameters)

        if self.hyperparameters['wrapped_optimizer_class'] is not None:
            base_optimizer = self.hyperparameters['base_optimizer_class'](self.torch_module.parameters(),
                                                                          **hyperparameters['base_optimizer_parameters'])

            self.wrapped_optimizer = self.hyperparameters['wrapped_optimizer_class'](optimizer=base_optimizer,
                                                                                     model=self.torch_module,
                                                                                     loss_per_example=self.torch_module.compute_loss_per_point,
                                                                                     **hyperparameters['wrapped_optimizer_parameters'])

        self._training_curves = []
        self._derived_statistics_histories = []

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
            'w_pres': np.ones(2),
            'w_nat_mean': np.zeros(2)
        }

    def get_default_hyperparameters(self):
        return {
            "base_optimizer_class": None,
            "wrapped_optimizer_class": None,
            "base_optimizer_params": {},
            "wrapped_optimizer_params": {},
            "N_steps": 1,
            "n_in": 2,
            "model_noise": 1    # model_noise is standard deviation
        }

    def sample(self, x, parameters):
        """
        Create some linear regression data. *** NOTE: This ignores the precisions of each of the values of w, and
        simply assuming the true (unknown) weight is w; this is different to finding the predictive distribution!! ***

        :param x: input values to predict at
        :param parameters: model parameters (not will not update)
        :param hyperparameters: model hyperparameters (will also not update)
        :return: y: tensor of predictions
        """
        return self.torch_module.sample(x, parameters).numpy()

    def predict(self, x, parameters=None, hyperparameters=None):
        """
        Bayesian prediction on probability at certain point, where the predictive distribution is Gaussian
        so this is the 'fully Bayesian' probability

        :param x: data tensor. Each row is a data-point
        :param parameters: linear regression parameters
        :param hyperparameters: linear regression hyperparams
        :return: mean and variance of the predictive distribution of each y
        """
        super().predict(x, parameters, hyperparameters)

        return self.torch_module.predict(x, parameters).numpy()

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
        x = torch.tensor(data["x"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float32)

        cav_nat_params = B.subtract_params(self.get_parameters(), t_i)
        # numpy dict for the effective prior
        self.torch_module.set_prior_parameters_from_numpy(cav_nat_params)

        print_interval = np.ceil(self.hyperparameters['N_steps'] / 20)

        training_curve = np.empty(self.hyperparameters['N_steps'])
        derived_statistics_history = []

        self._derived_statistics_history = []
        # lets just do this for the time being
        for i in range(self.hyperparameters['N_steps']):
            current_loss = self.wrapped_optimizer.fit_batch(x, y)
            derived_statistics = self.wrapped_optimizer.get_logged_statistics()
            derived_statistics_history.append(derived_statistics)
            training_curve[i] = current_loss
            if i % print_interval == 0:
                logger.info("Loss: {:.3f} after {} steps".format(current_loss, i))

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
