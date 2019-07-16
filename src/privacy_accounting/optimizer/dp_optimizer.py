from math import ceil

import torch

import torch.optim as optim
from torch.optim import Adam

import src.utils.torch_nest_utils as nest

class DPOptimiser(object):

    def __init__(self,
                 optimiser,
                 model,
                 vector_loss,
                 dp_sum_query,
                 microbatch_size=1):

        self.optimiser = optimiser
        self.model = model
        self.vector_loss = vector_loss
        self.dp_sum_query = dp_sum_query
        self.micobatch_size = microbatch_size

        self._global_parameters = self.dp_sum_query.initial_global_state()

    def fit_batch(self, x: torch.Tensor, y: torch.Tensor):

        loss = self.vector_loss(self.model(x), y)

        param_groups = self.optimiser.param_groups

        sample_state = self.dp_sum_query.initial_sample_state(
            nest.parameters_to_tensor_groups(param_groups, 'data')
        )
        sample_params = self.dp_sum_query.derive_sample_params(self._global_parameters)

        microbatches_losses = loss.split(self.micobatch_size, dim=0)

        def process_microbatch(losses, sample_state):
            self.optimiser.zero_grad()
            microbatch_loss = losses.mean(dim=0)
            microbatch_loss.backward(retain_graph=True)
            record = self.get_grads(param_groups)
            sample_state = self.dp_sum_query.accumulate_record(sample_params, sample_state, record)
            return sample_state

        for losses in microbatches_losses:
            sample_state = process_microbatch(losses, sample_state)

        final_grads = self.dp_sum_query.get_noised_result(sample_state, self._global_parameters)

        self.apply_grads(param_groups, grads=final_grads)

        self.optimiser.step()
        self.optimiser.zero_grad()

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