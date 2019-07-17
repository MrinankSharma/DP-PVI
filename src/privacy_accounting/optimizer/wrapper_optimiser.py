import abc

class WrapperOptimiser(object):
    @abc.abstractmethod
    def fit_batch(self,x, y):
        pass