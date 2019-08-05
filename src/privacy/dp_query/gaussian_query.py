import collections

import torch

import src.utils.torch_nest_utils as nest
from src.privacy.dp_query import dp_query


class GaussianDPQuery(dp_query.SumAggregationDPQuery):
    """ Implements the DPQuery interface for Gaussian noise sum queries

    Accumulates records via sum and applies Gaussian noise
    """

    _GlobalState = collections.namedtuple(
        '_GlobalState', ['l2_norm_clip', 'noise_stddev']
    )

    def __init__(self, l2_norm_clip, noise_stddev):
        """
        :param l2_norm_clip: Clipping bound to apply to the records magnitude.
        :param noise_stddev: The noise magnitude to apply to the query.
        """
        self._l2_norm_clip = l2_norm_clip
        self._record_l2_norm = None
        self._noise_stddev = noise_stddev
        self._ledger = None

    def set_ledger(self, ledger):
        self._ledger = ledger

    def make_global_state(self, l2_norm_clip, noise_stddev):
        return self._GlobalState(
            torch.tensor(l2_norm_clip, dtype=torch.float32),
            torch.tensor(noise_stddev, dtype=torch.float32)
        )

    def initial_global_state(self):
        return self.make_global_state(self._l2_norm_clip, self._noise_stddev)

    def derive_sample_params(self, global_state):
        return global_state.l2_norm_clip

    def initial_sample_state(self, param_groups):
        return nest.map_structure(torch.zeros_like, param_groups)

    def get_record_derived_data(self):
        return {
            "l2_norm:": self._record_l2_norm.numpy().item()
        }

    def preprocess_record(self, params, record):
        """
        Return the scaled record and also the l2 norm (to deduce whether clipping occured or not)

        :param params:
        :param record:
        :return:
        """
        l2_norm_clip = params
        l2_norm = torch.sqrt(nest.reduce_structure(lambda p: torch.norm(torch.flatten(p), p=2) ** 2,
                                                   torch.add,
                                                   record))
        self._record_l2_norm = l2_norm
        if l2_norm < l2_norm_clip:
            return record
        else:
            return nest.map_structure(lambda p: torch.div(p, torch.abs(l2_norm / l2_norm_clip)), record)

    def get_noised_result(self, sample_state, global_state):
        def add_noise(p):
            return p + (torch.randn_like(p) * global_state.noise_stddev)

        if self._ledger:
            self._ledger.record_sum_query(global_state.l2_norm_clip, global_state.noise_stddev)

        return nest.map_structure(add_noise, sample_state), global_state