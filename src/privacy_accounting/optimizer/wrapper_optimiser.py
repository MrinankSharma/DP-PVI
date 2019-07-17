import abc


class WrapperOptimiser(abc.ABCMeta):
    @abc.abstractmethod
    def fit_batch(self, x, y):
        pass
