import torch

from .wrapper_optimizer import WrapperOptimizer


class StandardOptimizer(WrapperOptimizer):

    def __init__(self, optimizer, model, loss_per_example):
        self.optimizer = optimizer
        self.model = model
        # note that this is a function handle
        self.loss_per_example = loss_per_example
        self._total_loss = 0

    def fit_batch(self, X: torch.Tensor, y: torch.Tensor):
        loss = self.loss_per_example(self.model(X), y)
        self._total_loss = torch.sum(loss)

        self.optimizer.zero_grad()
        self._total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self._total_loss.detach().numpy()

    def get_logged_statistics(self):
        return {}

    def get_step_summary(self):
        return {}
