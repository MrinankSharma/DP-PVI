from abc import ABC, abstractmethod

import ray
import torch

def zero_init_func(tensor):
    return torch.Tensor(tensor).fill_(0)


@ray.remote
class Client(ABC):
    def __init__(self, model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata):
        self.set_hyperparameters(hyperparameters)
        self.set_metadata(metadata)

        self.data = data
        self.model = model_class(model_parameters, model_hyperparameters)
        self.tracking_data = {}

    @abstractmethod
    def set_hyperparameters(self, **kwargs):
        pass

    @abstractmethod
    def set_metadata(self, **kwargs):
        pass

    @abstractmethod
    def compute_update(self, model_parameters=None, model_hyperparameters=None):
        if model_parameters is not None:
            self.model.set_parameters(model_parameters)
        if model_hyperparameters is not None:
            self.model.set_hyperparameters(model_hyperparameters)

    def get_tracking_data(self):
        return self.tracking_data


@ray.remote
class DatasetLevelDPClient(Client):
    def __init__(self, lambda_prior, model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata):
        super().__init__(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

        self.t_i = {}
        for key in model_parameters.keys():
            self.t_i[key] = self.t_i_init_func(model_parameters[key])

        self.lambda_i = lambda_prior


    def set_hyperparameters(self, privacy_function, t_i_init_func = zero_init_func, **kwargs):
        super().set_hyperparameters(kwargs)

        self.privacy_function = privacy_function
        self.t_i_init_func = t_i_init_func

    def set_metadata(self, **kwargs):
        pass

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
        delta_lambda_i = self.model.subtract_params(lambda_new,
                                                    lambda_old)

        # apply the privacy function, specified by the server
        delta_lambda_i_tilde, privacy_stats = self.privacy_function(delta_lambda_i)

        # compute the new
        lambda_new = self.model.add_parameters(lambda_old, delta_lambda_i_tilde)

        t_i_new = self.model.add_parameters(
            self.model.subtract_params(lambda_new,
                                       lambda_old),
            t_i_old
        )

        self.t_i = t_i_new

        return delta_lambda_i_tilde

