from abc import ABC, abstractmethod


class WrapperOptimizer(ABC):
    @abstractmethod
    def fit_batch(self, x, y):
        pass
