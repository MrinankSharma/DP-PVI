import abc


class WrapperOptimiser(abc.ABC):
    @abc.abstractmethod
    def fit_batch(self, x, y):
        pass
