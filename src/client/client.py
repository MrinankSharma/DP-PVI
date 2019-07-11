from abc import ABC, abstractmethod

import ray
import torch
import numpy as np

import src.utils.numpy_backend as B

def zero_init_func(tensor):
    return torch.Tensor(tensor).fill_(0)


# @ray.remote
class Client(ABC):

    def __init__(self, model_class, data, model_parameters=None, model_hyperparameters=None, hyperparameters=None, metadata=None):
        if hyperparameters is None:
            hyperparameters = {}

        if metadata is None:
            metadata = {}

        self.hyperparameters = self.get_default_hyperparameters()
        self.metadata = self.get_default_metadata()

        self.set_hyperparameters(hyperparameters)
        self.set_metadata(metadata)

        self.data = data
        self.model = model_class(model_parameters, model_hyperparameters)
        self.tracking_data = {}

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    def set_metadata(self, metadata):
        self.metadata = {**self.metadata, **metadata}

    @abstractmethod
    def get_default_hyperparameters(self):
        return {}

    @abstractmethod
    def get_default_metadata(self):
        return {}

    @abstractmethod
    def compute_update(self, model_parameters=None, model_hyperparameters=None):
        if model_parameters is not None:
            self.model.set_parameters(model_parameters)
        if model_hyperparameters is not None:
            self.model.set_hyperparameters(model_hyperparameters)

    def get_tracking_data(self):
        return self.tracking_data


@ray.remote
class DPClient(Client):
    def __init__(self, model_class, data, model_parameters=None, model_hyperparameters=None, hyperparameters=None, metadata=None):
        super().__init__(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

        self.t_i = {}
        for key in model_parameters.keys():
            self.t_i[key] = self.t_i_init_func(model_parameters[key])

        self.lambda_i = self.model.get_parameters()


    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)

        self.privacy_function = hyperparameters['privacy_function']
        self.t_i_init_func = hyperparameters['t_i_init_function']

    def set_metadata(self, metadata):
        super().set_metadata(metadata)

    def get_default_hyperparameters(self):
        default_hyperparameters = {
                                      **super().get_default_hyperparameters(),
                                      **{
                                          'privacy_function': lambda x: x,
                                          't_i_init_function': lambda x: np.zeros(x.shape)
                                      }
                                  }
        return default_hyperparameters

    def get_default_metadata(self):
        return super().get_default_metadata()

    def compute_update(self, model_parameters=None, model_hyperparameters=None):
        super().compute_update(model_parameters, model_hyperparameters)

        t_i_old = self.t_i
        lambda_old = self.model.parameters

        # find the new optimal parameters for this clients data
        lambda_new = self.model.fit(self.data,
                                      t_i_old,
                                      model_parameters,
                                      model_hyperparameters)

        # compute the change in parameters needed
        delta_lambda_i = B.subtract_params(lambda_new,
                                                    lambda_old)

        # apply the privacy function, specified by the server
        # delta_lambda_i_tilde, privacy_stats = self.privacy_function(delta_lambda_i)
        delta_lambda_i_tilde = delta_lambda_i

        # compute the new
        lambda_new = B.add_parameters(lambda_old, delta_lambda_i_tilde)

        t_i_new = B.add_parameters(
            B.subtract_params(lambda_new,
                                       lambda_old),
            t_i_old
        )

        self.t_i = t_i_new

        # print(t_i_old, t_i_new, lambda_old, lambda_new, delta_lambda_i_tilde)

        return delta_lambda_i_tilde

