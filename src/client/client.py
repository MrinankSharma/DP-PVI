import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

import src.utils.numpy_nest_utils as np_nest
import src.utils.numpy_utils as np_utils
from src.privacy.analysis import QueryWithLedger, OnlineAccountant

from src.model.logistic_regression_models import nat_params_to_params_dict

logger = logging.getLogger(__name__)


def zero_init_func(tensor):
    return torch.Tensor(tensor).fill_(0)


def ensure_positive_t_i_factory(key):
    def inner_function(params):
        ret = dict(params)
        val = ret[key]
        if np.sum(val<0)>0:
            logger.error("Having to do precision clipping - this is not good!")
        val[val < 0] = 0.0
        ret[key] = val

        return ret

    return inner_function

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

    def get_update(self, model_parameters=None, model_hyperparameters=None, update_ti=True):
        """ Method to wrap the update and then logging process.
        :param model_parameters: New model parameters from the server
        :param model_hyperparameters: New model hyperparameters from the server.
        :return:
        """

        update = self.compute_update(model_parameters=model_parameters, model_hyperparameters=model_hyperparameters,
                                     update_ti=update_ti)

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

    def get_compiled_log(self):
        """
        At the end of an experiment, return all the data about this client we would like to save in a nested
        dictionary format.
        :return:
        """
        return self.get_log()

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons one may not be is simulated unavailability
        or a client had expended all of its privacy.
        :return:
        """
        return True


class StandardClient(Client):
    def __init__(self, model_class, data, model_parameters=None, model_hyperparameters=None,
                 hyperparameters=None,
                 metadata=None):

        super().__init__(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

        self.t_i = {}

        self.t_i = self.t_i_init_func()
        # for key in model_parameters.keys():
        #     self.t_i[key] = self.t_i_init_func(model_parameters[key])

        self.model.set_parameters(model_parameters)

    @classmethod
    def create_factory(cls, model_class, data, model_parameters=None, model_hyperparameters=None, hyperparameters=None,
                       metadata=None):

        return lambda: cls(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)

        self.t_i_init_func = self.hyperparameters['t_i_init_function']
        self.t_i_postprocess_funtion = self.hyperparameters['t_i_postprocess_function']
        self.damping_factor = self.hyperparameters['damping_factor']

    def set_metadata(self, metadata):
        super().set_metadata(metadata)

    @classmethod
    def get_default_hyperparameters(cls):
        default_hyperparameters = {
            **super().get_default_hyperparameters(),
            **{
                't_i_init_function': lambda x: np.zeros(x.shape),
                't_i_postprocess_function': lambda x: x,
                "damping_factor": 1.,
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
                'log_t_i': True,
                "log_model_info": False,
            }
        }

    def compute_update(self, model_parameters=None, model_hyperparameters=None, update_ti=True):
        super().compute_update(model_parameters, model_hyperparameters)

        t_i_old = self.t_i
        lambda_old = self.model.get_parameters()

        # find the new optimal parameters for this clients data
        lambda_new = self.model.fit(self.data,
                                    t_i_old,
                                    model_parameters,
                                    model_hyperparameters)

        delta_lambda_i = np_utils.subtract_params(lambda_new,
                                                  lambda_old)
        # logger.info(f"t_i_old {t_i_old}")
        # logger.info(f"lambda_old {lambda_old}")
        # logger.info(f"lambda_new {lambda_new}")

        delta_lambda_i = np_nest.apply_to_structure(lambda x: np.multiply(x, self.damping_factor), delta_lambda_i)

        # compute the new
        lambda_new = np_utils.add_parameters(lambda_old, delta_lambda_i)

        t_i_new = np_utils.add_parameters(
            np_utils.subtract_params(lambda_new,
                                     lambda_old),
            t_i_old
        )

        t_i_new = self.t_i_postprocess_funtion(t_i_new, t_i_old)
        # logger.info(f"t_i_new {t_i_new}")
        delta_lambda_i_tilde = np_utils.subtract_params(t_i_new, t_i_old)

        if update_ti:
            self.t_i = t_i_new
            self.times_updated += 1

        return delta_lambda_i_tilde

    def update_ti(self, delta_ti):
        t_i_new = np_utils.add_parameters(delta_ti, self.t_i)
        t_i_new = self.t_i_postprocess_funtion(t_i_new, self.t_i)
        self.t_i = t_i_new
        # logger.debug(f"New t_i {self.t_i}")
        self.times_updated += 1

    def log_update(self):
        super().log_update()

        if 'global_iteration' in list(self.metadata.keys()):
            self.log['global_iteration'].append(self.metadata['global_iteration'])

        self.log['times_updated'].append(self.times_updated)

        if self.metadata['log_params']:
            self.log['params'].append(np_nest.structured_ndarrays_to_lists(self.model.get_parameters()))
        if self.metadata['log_t_i']:
            self.log['t_i'].append(np_nest.structured_ndarrays_to_lists(np_nest.map_structure(np.mean, self.t_i)))
        if self.metadata['log_model_info']:
            self.log['model'].append(np_nest.structured_ndarrays_to_lists(self.model.get_incremental_log_record()))

    def log_sacred(self):
        log = {}
        log['model'] = self.model.get_incremental_sacred_record()

        return log, self.times_updated

    @property
    def parameters(self):
        return self.model.get_parameters()


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

    @classmethod
    def create_factory(cls, model_class, dp_query_class, data, accounting_dict, model_parameters, model_hyperparameters,
                       hyperparameters, metadata=None):

        return lambda: cls(model_class, dp_query_class, accounting_dict, data, model_parameters, model_hyperparameters,
                           hyperparameters, metadata)

    def compute_update(self, model_parameters=None, model_hyperparameters=None, update_ti=True):
        logger.debug("Computing Update")
        if not self.can_update():
            logger.warning('Incorrectly tired to update a client that cant be updated!')
            return np_nest.map_structure(np.zeros_like, self.model.get_parameters())
        delta_lambda_i_tilde = super().compute_update(model_parameters, model_hyperparameters, update_ti)

        logger.debug("Computing Privacy Cost")
        formatted_ledger = self.dp_query.ledger.get_formatted_ledger()
        for _, accountant in self.accountants.items():
            accountant.update_privacy(formatted_ledger)

        return delta_lambda_i_tilde

    @classmethod
    def get_default_hyperparameters(cls):
        return {
            **super().get_default_hyperparameters(),
            'dp_query_parameters': {},
            'max_epsilon': None
        }

    def log_update(self):
        super().log_update()

        for k, v in self.accountants.items():
            self.log[k].append(v.privacy_bound)

    def log_sacred(self):
        log, times_updated = super().log_sacred()

        for k, v in self.accountants.items():
            log[k + '.epsilon'] = v.privacy_bound[0]
            log[k + '.delta'] = v.privacy_bound[1]

        return log, times_updated

    def get_compiled_log(self):
        return self.log

    def can_update(self):
        if self.hyperparameters['max_epsilon'] is not None:
            for k, v in self.accountants.items():
                if v.privacy_bound[0] > self.hyperparameters['max_epsilon']:
                    logger.debug(f'client capped out on epsilon')
                    return False

        return True


class GradientVIClient(Client):
    """
    An alternative client to allow us to perform gradient based communication rather than local likelihood.
    Useful if you wish to do distributed batch VI opposed to PVI.
    """
    def __init__(self, model_class, data, model_parameters=None, model_hyperparameters=None,
                 hyperparameters=None,
                 metadata=None):

        super().__init__(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

        self.t_i = self.model.get_parameters()
        self.lambda_i = self.model.get_parameters()

    @classmethod
    def create_factory(cls, model_class, data, model_parameters=None, model_hyperparameters=None, hyperparameters=None,
                       metadata=None):

        return lambda: cls(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)

    def set_metadata(self, metadata):
        super().set_metadata(metadata)

    @classmethod
    def get_default_hyperparameters(cls):
        default_hyperparameters = {
            **super().get_default_hyperparameters(),
            **{
                'prior': None
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
                "log_model_info": False,
            }
        }

    def compute_update(self, model_parameters=None, model_hyperparameters=None, update_ti=True):
        super().compute_update(model_parameters, model_hyperparameters)

        parameters_old = self.model.get_parameters()
        t_i = np_nest.map_structure(np.subtract, parameters_old, self.hyperparameters['prior'])

        self.t_i = t_i

        parameters_new = self.model.fit(self.data, t_i)

        delta_lambda_i = np_utils.subtract_params(parameters_new,
                                                  parameters_old)

        logger.debug(f"Old Params: {parameters_old}\n"
                    f"New Params: {parameters_new}\n")

        self.times_updated += 1

        return delta_lambda_i

    def update_ti(self, delta_ti):
        pass

    def log_update(self):
        super().log_update()

        if 'global_iteration' in list(self.metadata.keys()):
            self.log['global_iteration'].append(self.metadata['global_iteration'])

        self.log['times_updated'].append(self.times_updated)

        if self.metadata['log_params']:
            self.log['params'].append(np_nest.structured_ndarrays_to_lists(self.model.get_parameters()))
        if self.metadata['log_model_info']:
            self.log['model'].append(np_nest.structured_ndarrays_to_lists(self.model.get_incremental_log_record()))

    def log_sacred(self):
        log = {}
        log['model'] = self.model.get_incremental_sacred_record()

        return log, self.times_updated

    @property
    def parameters(self):
        return self.model.get_parameters()


class DPGradientVIClient(GradientVIClient):
    """ Wrapper class to add privacy to gradient clients """
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

    @classmethod
    def create_factory(cls, model_class, dp_query_class, data, accounting_dict, model_parameters, model_hyperparameters,
                       hyperparameters, metadata=None):

        return lambda: cls(model_class, dp_query_class, accounting_dict, data, model_parameters, model_hyperparameters,
                           hyperparameters, metadata)

    def compute_update(self, model_parameters=None, model_hyperparameters=None, update_ti=True):
        logger.debug("Computing Update")
        if not self.can_update():
            logger.warning('Incorrectly tired to update a client tha cant be updated!')
            return np_nest.map_structure(np.zeros_like, self.model.get_parameters())
        delta_lambda_i_tilde = super().compute_update(model_parameters, model_hyperparameters, update_ti)

        logger.debug("Computing Privacy Cost")
        formatted_ledger = self.dp_query.ledger.get_formatted_ledger()
        for _, accountant in self.accountants.items():
            accountant.update_privacy(formatted_ledger)

        return delta_lambda_i_tilde

    @classmethod
    def get_default_hyperparameters(cls):
        return {
            **super().get_default_hyperparameters(),
            'dp_query_parameters': {},
            'max_epsilon': None
        }

    def log_update(self):
        super().log_update()

        for k, v in self.accountants.items():
            self.log[k].append(v.privacy_bound)

    def log_sacred(self):
        log, times_updated = super().log_sacred()

        for k, v in self.accountants.items():
            log[k + '.epsilon'] = v.privacy_bound[0]
            log[k + '.delta'] = v.privacy_bound[1]

        return log, times_updated

    def get_compiled_log(self):
        return self.log

    def can_update(self):
        if self.hyperparameters['max_epsilon'] is not None:
            for k, v in self.accountants.items():
                if v.privacy_bound[0] > self.hyperparameters['max_epsilon']:
                    logger.debug(f'client capped out on epsilon')
                    return False

        return True