from collections import defaultdict
from math import ceil

import numpy as np
import torch

import src.utils.torch_nest_utils as nest
from src.privacy.analysis import QueryWithLedger
from src.privacy.dp_query import GaussianDPQuery
from .wrapper_optimizer import WrapperOptimizer


class DPOptimizer(WrapperOptimizer):

    def __init__(self,
                 optimizer,
                 model,
                 loss_per_example,
                 dp_sum_query,
                 num_microbatches=None):

        self.optimizer = optimizer
        self.model = model
        self.loss_per_example = loss_per_example
        self.dp_sum_query = dp_sum_query
        self.num_microbatches = num_microbatches

        self._global_parameters = self.dp_sum_query.initial_global_state()
        self._derived_records_data = defaultdict(list)

    def fit_batch(self, x: torch.Tensor, y: torch.Tensor):
        loss = self.loss_per_example(self.model(x), y)

        param_groups = self.optimizer.param_groups

        # Get the correct shape gradient tensors to then set to the intial
        # state of the sample. Often all zero for zero gradients.
        sample_state = self.dp_sum_query.initial_sample_state(
            nest.parameters_to_tensor_groups(param_groups, 'data')
        )
        sample_params = self.dp_sum_query.derive_sample_params(self._global_parameters)

        microbatch_size = 1 if self.num_microbatches is None else ceil(loss.shape[0] / self.num_microbatches)

        microbatches_losses = loss.split(microbatch_size, dim=0)

        def process_microbatch(losses, sample_state):
            self.optimizer.zero_grad()
            microbatch_loss = losses.mean(dim=0)
            microbatch_loss.backward(retain_graph=True)
            record = self.get_grads(param_groups)
            sample_state = self.dp_sum_query.accumulate_record(sample_params, sample_state, record)
            derived_record_data = self.dp_sum_query.get_record_derived_data()
            return sample_state, derived_record_data

        self._derived_records_data = defaultdict(list)

        for losses in microbatches_losses:
            sample_state, derived_record_data = process_microbatch(losses,
                                                                   sample_state)  # accumulate up the clipped microbatch gradients

            for k, v in derived_record_data.items():
                self._derived_records_data[k].append(v)

        self._derived_records_data = dict(self._derived_records_data)

        for k, v in self._derived_records_data.items():
            # summarise statistics instead
            self._derived_records_data[k] = np.percentile(np.array(v), [10.0, 30.0, 50.0, 70.0, 90.0])

        final_grads, _ = self.dp_sum_query.get_noised_result(sample_state, self._global_parameters)

        self.apply_grads(param_groups, grads=final_grads)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # return the loss at the start (for efficiency purposes)
        return torch.sum(loss).detach().numpy()

    def apply_grads(self, param_groups, grads):
        for param_group, grad_group in zip(param_groups, grads):
            for p, grad in zip(param_group['params'], grad_group):
                p.grad.data = grad

    def get_grads(self, param_groups):
        grads = []

        for group in param_groups:
            group_grads = []
            for p in group['params']:
                group_grads.append(p.grad.data.clone().detach())
            grads.append(group_grads)

        return grads

    def get_logged_statistics(self):
        return self._derived_records_data


class DPGaussianOptimizer(DPOptimizer):
    """ Specific Gaussian mechanism optimizer for L2 clipping and noise privacy """

    def __init__(self,
                 l2_norm_clip,
                 noise_multiplier,
                 ledger=None,
                 *args,
                 **kwargs):
        dp_sum_query = GaussianDPQuery(l2_norm_clip, l2_norm_clip * noise_multiplier)

        if ledger:
            dp_sum_query = QueryWithLedger(dp_sum_query, ledger=ledger)

        super().__init__(
            dp_sum_query=dp_sum_query,
            *args,
            **kwargs
        )

    @property
    def ledger(self):
        return self.dp_sum_query.ledger
