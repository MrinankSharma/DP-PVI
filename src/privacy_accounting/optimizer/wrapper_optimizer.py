import abc


class WrapperOptimizer(abc.ABCMeta):
    @abc.abstractmethod
    def fit_batch(self, x, y):
        pass
