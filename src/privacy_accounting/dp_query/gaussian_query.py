import collections

import torch

import src.utils.torch_nest_utils as nest
from src.privacy_accounting.dp_query import dp_query

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
        self._noise_stddev = noise_stddev
        self.ledger = None

    def set_ledger(self, ledger):
        self.ledger = ledger

    def make_global_state(self, l2_norm_clip, noise_stddev):
        return self._GlobalState(
            torch.Tensor([l2_norm_clip]),
            torch.Tensor([noise_stddev])
        )

    def initial_global_state(self):
        return self.make_global_state(self._l2_norm_clip, self._noise_stddev)

    def derive_sample_params(self, global_state):
        return global_state.l2_norm_clip

    def initial_sample_state(self, param_groups):
        return nest.map_structure(torch.zeros_like, param_groups)

    def preprocess_record(self, params, record):
        l2_norm_clip = params
        l2_norm  = torch.sqrt(nest.reduce_structure(lambda p: torch.norm(torch.flatten(p), p=2) ** 2,
                                         torch.add,
                                         record))

        if l2_norm < l2_norm_clip:
            return record

        else:
            return nest.map_structure(lambda p: torch.div(p, torch.abs(l2_norm/l2_norm_clip)), record)

    def get_noised_result(self, sample_state, global_parameters):
        def add_noise(p):
            return p + (torch.randn_like(p) * global_parameters.noise_stddev)

        return nest.map_structure(add_noise, sample_state)