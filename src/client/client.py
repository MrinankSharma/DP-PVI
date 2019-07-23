from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import ray
import torch

import src.utils.numpy_nest_utils as np_nest
import src.utils.numpy_utils as np_utils


def zero_init_func(tensor):
    return torch.Tensor(tensor).fill_(0)


# @ray.remote
class Client(ABC):

    def __init__(self, model_class, data, model_parameters=None, model_hyperparameters=None, hyperparameters=None,
                 metadata=None):
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
        self.log = defaultdict(list)
        self.times_updated = 0

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

    def get_log(self):
        return self.log

    @abstractmethod
    def log_update(self):
        """
        Log various things about the client in self.log. Flexible form.
        """
        pass

    @abstractmethod
    def log_sacred(self):
        """
        Log various things we may want to see in the sacred logs. Reduced form
        :return: A flat dictionary containing scalars of interest for the current state, the current iteration.
        """
        pass


@ray.remote
class DPClient(Client):
    def __init__(self, model_class, data, model_parameters=None, model_hyperparameters=None, hyperparameters=None,
                 metadata=None):
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
        return {
            **super().get_default_metadata(),
            **{
                'global_iteration': 0,
                'log_params': False,
                'log_t_i': False,
            }
        }

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
        delta_lambda_i = np_utils.subtract_params(lambda_new,
                                                  lambda_old)

        # apply the privacy function, specified by the server
        # delta_lambda_i_tilde, privacy_stats = self.privacy_function(delta_lambda_i)
        delta_lambda_i_tilde = delta_lambda_i

        # compute the new
        lambda_new = np_utils.add_parameters(lambda_old, delta_lambda_i_tilde)

        t_i_new = np_utils.add_parameters(
            np_utils.subtract_params(lambda_new,
                                     lambda_old),
            t_i_old
        )

        self.t_i = t_i_new

        self.times_updated += 1
        self.log_update()
        self.model.log_update()

        return delta_lambda_i_tilde

    def log_update(self):
        super().log_update()

        if 'global_iteration' in list(self.metadata.keys()):
            self.log['global_iteration'].append(self.metadata['global_iteration'])

        self.log['times_updated'].append(self.times_updated)

        if self.metadata['log_params']:
            self.log['params'].append(np_nest.structured_ndarrays_to_lists(self.model.get_parameters()))
        if self.metadata['log_t_i']:
            self.log['t_i'].append(np_nest.structured_ndarrays_to_lists(self.t_i))

    def log_sacred(self):
        log = {}

        if self.metadata['log_params']:
            log['params'] = np_nest.structured_ndarrays_to_lists(self.model.get_parameters())
        if self.metadata['log_t_i']:
            log['t_i'] = np_nest.structured_ndarrays_to_lists(self.t_i)

        log['model'] = self.model.log_sacred()

        return np_nest.flatten(log, sep='.'), self.times_updated
