from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

import src.utils.numpy_nest_utils as np_nest
import src.utils.numpy_utils as np_utils
from src.privacy_accounting.analysis import QueryWithLedger, OnlineAccountant


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

    @classmethod
    def get_default_hyperparameters(cls):
        return {}

    @classmethod
    def get_default_metadata(cls):
        return {}

    def get_update(self, model_parameters=None, model_hyperparameters=None):
        """ Method to wrap the update and then logging process.
        :param model_parameters: New model parameters from the server
        :param model_hyperparameters: New model hyperparameters from the server.
        :return:
        """

        # if model_parameters is not None:
        #     self.model.set_parameters(model_parameters)
        # if model_hyperparameters is not None:
        #     self.model.set_hyperparameters(model_hyperparameters)

        update = self.compute_update(model_parameters=model_parameters, model_hyperparameters=model_hyperparameters)

        self.log_update()

        return update

    @abstractmethod
    def compute_update(self, model_parameters=None, model_hyperparameters=None):
        """ Abstract method for computing the update itself.
        :return: The update step to return to the server
        """
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
        :return: Nested dictionaries of scalars to log, the current iteration
        """
        pass


# @ray.remote
class StandardClient(Client):
    def __init__(self, model_class, data, model_parameters=None, model_hyperparameters=None,
                 hyperparameters=None,
                 metadata=None):

        super().__init__(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

        self.t_i = {}
        for key in model_parameters.keys():
            self.t_i[key] = self.t_i_init_func(model_parameters[key])

        self.lambda_i = self.model.get_parameters()

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)

        self.t_i_init_func = hyperparameters['t_i_init_function']

    def set_metadata(self, metadata):
        super().set_metadata(metadata)

    @classmethod
    def get_default_hyperparameters(cls):
        default_hyperparameters = {
            **super().get_default_hyperparameters(),
            **{
                't_i_init_function': lambda x: np.zeros(x.shape),
            }
        }
        return default_hyperparameters

    @classmethod
    def get_default_metadata(cls):
        return {
            **super().get_default_metadata(),
            **{
                'global_iteration': 0,
                'log_params': False,
                'log_t_i': False,
                "log_model_info": True,
            }
        }

    def compute_update(self, model_parameters=None, model_hyperparameters=None):
        super().compute_update(model_parameters=None, model_hyperparameters=None)

        t_i_old = self.t_i
        lambda_old = self.model.get_parameters()

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
        if self.metadata['log_model_info']:
            self.log['model'].append(np_nest.structured_ndarrays_to_lists(self.model.get_incremental_log_record()))

    def log_sacred(self):
        log = {}

        if self.metadata['log_params']:
            log['params'] = np_nest.structured_ndarrays_to_lists(self.model.get_parameters())
        if self.metadata['log_t_i']:
            log['t_i'] = np_nest.structured_ndarrays_to_lists(self.t_i)

        log['model'] = self.model.get_incremental_sacred_record()

        return log, self.times_updated


# @ray.remote
class DPClient(StandardClient):
    """ Wrapper class to add privacy tracking to a client, and DP based optimisation."""

    def __init__(self, model_class, dp_query_class, accounting_dict, data, model_parameters=None,
                 model_hyperparameters=None, hyperparameters=None,
                 metadata=None):

        query = dp_query_class(**hyperparameters['dp_query_parameters'])
        self.dp_query = QueryWithLedger(query, data['x'].shape[0], float(
            model_hyperparameters['batch_size'] / float(data['x'].shape[0])))
        model_hyperparameters['wrapped_optimizer_parameters']['dp_sum_query'] = self.dp_query

        self.accountants = {}
        for k, v in accounting_dict.items():
            self.accountants[k] = OnlineAccountant(**v)

        super().__init__(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

    def compute_update(self, model_parameters=None, model_hyperparameters=None):
        delta_lambda_i_tilde = super().compute_update(model_parameters=None, model_hyperparameters=None)

        formatted_ledger = self.dp_query.ledger.get_formatted_ledger()
        for _, accountant in self.accountants.items():
            accountant.update_privacy(formatted_ledger)

        return delta_lambda_i_tilde

    @classmethod
    def get_default_hyperparameters(cls):
        return {
            **super().get_default_hyperparameters(),
            'dp_query_parameters': {}
        }

    def log_update(self):
        super().log_update()

        for k, v in self.accountants.items():
            self.log[k].append(v.privacy_bound)

        self.log['ledger'].append(self.dp_query.ledger.get_formatted_ledger())

    def log_sacred(self):
        log, times_updated = super().log_sacred()

        for k, v in self.accountants.items():
            log[k + '.epsilon'] = v.privacy_bound[0]
            log[k + '.delta'] = v.privacy_bound[1]

        return log, times_updated
