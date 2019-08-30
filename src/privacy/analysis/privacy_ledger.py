import collections

import torch
from math import ceil

import src.utils.numpy_utils as np_utils
import src.utils.torch_nest_utils as nest
from src.privacy.dp_query import dp_query
from src.utils.torch_tensor_buffer import TensorBuffer

SampleEntry = collections.namedtuple(  # pylint: disable=invalid-name
    'SampleEntry', ['population_size', 'selection_probability', 'queries'])

GaussianSumQueryEntry = collections.namedtuple(  # pylint: disable=invalid-name
    'GaussianSumQueryEntry', ['l2_norm_bound', 'noise_stddev'])


def format_ledger(sample_array, query_array):
    """ Converts array representation into a list of SampleEntries.

    :param sample_array: samples are whole mechanisms applied to the data, i.e. minbatch -> grads -> clip -> sum -> noise
    :param query_array: queries are the individual microbatch -> grads -> clip operations.
    Composing these together is just a way of keeping a detailed record of what was done. Could be done in a flatter way
    I think if you were worried about simplicity.
    :return: A numpy + named tuple version of the TensorBuffer stored operations that have been applied to the data
    """
    samples = []
    query_pos = 0
    sample_pos = 0
    for sample in sample_array:
        population_size, selection_probability, num_queries = sample
        queries = []
        for _ in range(int(num_queries)):
            query = query_array[query_pos]
            assert int(query[0]) == sample_pos
            queries.append(GaussianSumQueryEntry(
                *[np_utils.to_pure_python(item) for item in query[1:]]
            ))
            query_pos += 1
        samples.append(SampleEntry(np_utils.to_pure_python(population_size),
                                   np_utils.to_pure_python(selection_probability),
                                   queries))
        sample_pos += 1
    return samples


class PrivacyLedger(object):
    """ Class for keeping a record of all the privacy events that occur
    through a DPQuery executed on a given dataset.
    """

    def __init__(self, population_size, selection_probability):
        """ Initialise the privacy ledger

        :param population_size: A (variable) integer specifying the amount of data in
        the datashard being kept private by the query. I.e. The amount of data
        used to train in each epoch.
        :param selection_probability: A (variable) float specifying the probability
        that each record in the datashard is included in any given sample, i.e. the
        number of minibatches.
        """
        self._population_size = population_size
        self._selection_probability = selection_probability

        # Initial capacity such that we can hold one full epoch of updates
        init_capacity = ceil(1 / self._selection_probability)

        self._query_buffer = TensorBuffer(init_capacity, [3])
        self._sample_buffer = TensorBuffer(init_capacity, [3])

        self._sample_count = 0
        self._query_count = 0

    def record_sum_query(self, l2_clipping_bound, noise_stddev):
        """ Record a query that was issued in the ledger.

        :param l2_clipping_bound: Max L2 norm of the tensor group in the query.
        :param noise_stddev: The standard deviation of the noise applied to the sum.
        """
        self._query_count = self._query_count + 1
        self._query_buffer.append(torch.Tensor([self._sample_count, l2_clipping_bound, noise_stddev]))

    def finalise_sample(self):
        """ Finalises sample and records sample ledger entry"""
        sample_var = torch.Tensor([self._population_size, self._selection_probability, self._query_count])
        self._sample_buffer.append(sample_var)
        self._sample_count = self._sample_count + 1
        self._query_count = 0

    def get_formatted_ledger(self):
        """ Returns a formatted version of the ledger for use in privacy accounting """
        return format_ledger(self._sample_buffer.values.numpy(),
                             self._query_buffer.values.numpy())


class QueryWithLedger(dp_query.DPQuery):
    """ A class for DPQueries that stores the queries in a privacy ledger.

    Simple wrapper for a DQQuery-PrivacyLedger pair to ensure correct running.
    """

    def __init__(self,
                 query,
                 population_size=None,
                 selection_probability=None,
                 ledger=None):

        self._query = query
        if population_size is not None and selection_probability is not None:
            self.set_ledger(PrivacyLedger(population_size, selection_probability))
        elif ledger is not None:
            self.set_ledger(ledger)
        else:
            raise ValueError('One of (population_size, selection_probability) or '
                             'ledger must be specified.')

    @property
    def ledger(self):
        return self._ledger

    def set_ledger(self, ledger):
        self._ledger = ledger
        self._query.set_ledger(ledger)

    @property
    def query(self):
        return self._query

    def initial_global_state(self):
        """See base class."""
        return self._query.initial_global_state()

    def derive_sample_params(self, global_state):
        """See base class."""
        return self._query.derive_sample_params(global_state)

    def initial_sample_state(self, template):
        """See base class."""
        return self._query.initial_sample_state(template)

    def preprocess_record(self, params, record):
        """See base class."""
        return self._query.preprocess_record(params, record)

    def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
        """See base class."""
        return self._query.accumulate_preprocessed_record(
            sample_state, preprocessed_record)

    def merge_sample_states(self, sample_state_1, sample_state_2):
        """See base class."""
        return self._query.merge_sample_states(sample_state_1, sample_state_2)

    def get_noised_result(self, sample_state, global_state):
        """Ensures sample is recorded to the ledger and returns noised result."""
        result, new_global_state = self._query.get_noised_result(sample_state, global_state)
        self._ledger.finalise_sample()
        op = lambda tensor: tensor.clone().detach()
        return nest.map_structure(op, result), new_global_state

    def get_record_derived_data(self):
        return self._query.get_record_derived_data()


class QueryWithPerClientLedger(dp_query.DPQuery):
    """ A class for DPQueries that stores the queries in a privacy ledger.

    Simple wrapper for a DQQuery-PrivacyLedger pair to ensure correct running.
    """

    def __init__(self,
                 query,
                 num_clients,
                 selection_probability):

        # note that this query ledger is not actually relevant here!!
        self.M = num_clients
        self._query = query
        self.set_ledgers([PrivacyLedger(num_clients, selection_probability) for _ in range(num_clients)])

    @property
    def query(self):
        return self._query

    @property
    def ledgers(self):
        return self._ledgers

    def set_ledgers(self, ledgers):
        self._ledgers = ledgers

    def initial_global_state(self):
        """See base class."""
        return self._query.initial_global_state()

    def derive_sample_params(self, global_state):
        """See base class."""
        return self._query.derive_sample_params(global_state)

    def initial_sample_state(self, template):
        """See base class."""
        return self._query.initial_sample_state(template)

    def preprocess_record(self, params, record):
        """See base class."""
        return self._query.preprocess_record(params, record)

    def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
        """See base class."""
        return self._query.accumulate_preprocessed_record(
            sample_state, preprocessed_record)

    def merge_sample_states(self, sample_state_1, sample_state_2):
        """See base class."""
        return self._query.merge_sample_states(sample_state_1, sample_state_2)

    def get_noised_result(self, sample_state, global_state, selected_indices):
        """Ensures sample is recorded to the ledger and returns noised result."""
        result, new_global_state = self._query.get_noised_result(sample_state, global_state)
        # record sum queries for each client who was selected
        for i in range(self.M):
            if i in selected_indices:
                self.ledgers[i].record_sum_query(global_state.l2_norm_clip, global_state.noise_stddev)
                self.ledgers[i].finalise_sample()

        if isinstance(result, torch.Tensor):
            op = lambda tensor: tensor.clone().detach()
            return nest.map_structure(op, result), new_global_state
        else:
            return result, new_global_state

    def get_record_derived_data(self):
        return self._query.get_record_derived_data()

    def get_formatted_ledgers(self):
        """ Returns a formatted version of the ledger for use in privacy accounting """
        return [x.get_formatted_ledger() for x in self.ledgers]
