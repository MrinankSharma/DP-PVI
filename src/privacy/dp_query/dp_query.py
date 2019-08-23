import abc
import collections

import torch

import numpy as np

import src.utils.torch_nest_utils as nest
import src.utils.numpy_nest_utils as np_nest


class DPQuery(abc.ABC):
    """ General class for a query applied to data in a DP fashion.
    Terminology derived from definitions in https://arxiv.org/abs/1812.06210
    """

    def set_ledger(self, ledger):
        """ Sets the ledger for this query to record privacy events in.
        :param ledger: A 'PrivacyLedger'.
        """

        del ledger
        raise TypeError(
            'DPQuery typ %s does not support set_ledger.' % type(self).__name__
        )

    def initial_global_state(self):
        """ Initial state for the global settings of the DPQuery. """

        return ()

    def derive_sample_params(self, global_state):
        """ Given some global state, derive the parameters to use for the next sample.

        :param global_state: A global state of the DPQuery.
        :return Parameters to use in the next sample.
        """

        del global_state
        return ()

    @abc.abstractmethod
    def initial_sample_state(self, param_groups):
        """ Returns an initial state to use at the start of the next sample.

        :param param_groups: The param_groups to use as a template for the blank state.
        :return: An initial sample state.
        """

    def preprocess_record(self, params, record):
        """ Preprocess a single record.

        Applies to a single clients record, this could be clipping a vector to a fixed L2
        norm.
        :param params: The parameters for this sample, in DP-SGD, the L2 bound.
        :param record: The record to process, in DP-SGD, a single mircobatch gradient.
        :return: A structure of tensors to apply.
        """

        del params
        return record

    @abc.abstractmethod
    def accumulate_preprocessed_record(self, sample_state, record):
        """ Accumulate a single record into the current sample state.

        Does simple aggregation of the recodrs, usually just summing.

        :param sample_state: The current sample state e.g. in DP-SGD the accumulated
        clipped gradients so far.
        :param record: The new record to accumulate.
        :return: The updated sample state.
        """

    def accumulate_record(self, params, sample_state, record):
        """ Accumulate a new record into the sample state.

        Simple delegation function to other defined functions.

        :param params: The sample parameters, e.g. clipping bound in DP-SGD.
        :param sample_state: The current sample state.
        :param record: The new record to accumulate.
        :return: The updated sample state with extra information about the record
        """
        preprocessed_record = self.preprocess_record(params, record)
        return self.accumulate_preprocessed_record(sample_state, preprocessed_record)

    @abc.abstractmethod
    def merge_sample_states(self, sample_state_1, sample_state_2):
        """ Merges two sample_states together.

        :param sample_state_1: The first sample state.
        :param sample_state_2: The second sample state.
        :return:  The merged sample states.
        """

    @abc.abstractmethod
    def get_noised_result(self, sample_state, global_parameters):
        """ Applies privacy noise after all the samples have been accumulated
        :param sample_state: The accumulated sample stata
        :param global_parameters: The current global parameters
        :return: A tuple (results, new_global_state) that contains the result of the
        query and the new query global state
        """


class SumAggregationDPQuery(DPQuery):
    """ Base class for when a DPQuery aggregates via summation """

    def initial_sample_state(self, param_groups):
        """ Return state of zeros the same shape as the parameter groups."""
        return nest.map_structure(torch.zeros_like, param_groups)

    def accumulate_preprocessed_record(self, sample_state, record):
        return nest.map_structure(torch.add, sample_state, record)

    def merge_sample_states(self, sample_state_1, sample_state_2):
        return nest.map_structure(torch.add, sample_state_1, sample_state_2)


class NumpyNoDPSumQuery(DPQuery):
    _GlobalState = collections.namedtuple(
        '_GlobalState', ['l2_norm_clip', 'noise_stddev']
    )

    def initial_global_state(self):
        return self._GlobalState(np.inf, 0)

    def initial_sample_state(self, param_groups):
        """ Return state of zeros the same shape as the parameter groups."""
        return np_nest.map_structure(np.zeros_like, param_groups)

    def accumulate_preprocessed_record(self, sample_state, record):
        return np_nest.map_structure(np.add, sample_state, record)

    def merge_sample_states(self, sample_state_1, sample_state_2):
        return np_nest.map_structure(np.add, sample_state_1, sample_state_2)

    def get_noised_result(self, sample_state, global_parameters):
        return sample_state, self._GlobalState(np.inf, 0)

    def get_record_derived_data(self):
        return {}