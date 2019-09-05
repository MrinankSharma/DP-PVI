from abc import ABC, abstractmethod


class WrapperOptimizer(ABC):
    @abstractmethod
    def fit_batch(self, x, y):
        pass

    def get_logged_statistics(self):
        return {}

    def get_step_summary(self):
        return {}