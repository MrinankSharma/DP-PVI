from abc import ABC, abstractmethod
from collections import defaultdict


class Model(ABC):
    def __init__(self, parameters=None, hyperparameters=None):

        if parameters is None:
            parameters = {}

        if hyperparameters is None:
            hyperparameters = {}

        self.parameters = self.get_default_parameters()
        self.hyperparameters = self.get_default_hyperparameters()

        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

        self.log = defaultdict(list)

    def set_parameters(self, parameters):
        self.parameters = {**self.parameters, **parameters}

    def get_parameters(self):
        '''
        In case you need to repackage parameters somehow from a form other than the dictionary.
        :return: a dictionary of the parameters
        '''
        return self.parameters

    @classmethod
    def get_default_parameters(cls):
        '''
        :return: A default set of parameters for the model. These might be all zero. Mostly used to get the shape that
        the parameters should be to make parameter server code more general.
        '''
        pass

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    @classmethod
    def get_default_hyperparameters(cls):
        '''
        :return: A default set of hyperparameters( for the model. These might be all zero. Mostly used to get the shape that
        the hyperparameters should be to make parameter server code more general and easier to remeber what could go in here.
        '''
        pass

    @abstractmethod
    def fit(self, data, t_i, parameters=None, hyperparameters=None):
        '''
        :param data: The local data to refine the model with
        :param t_i: The local contribution of the client
        :param parameters: optinal updated model parameters
        :param hyperparameters: optional updated hyperparameters
        :return: lambda_i_new, t_i_new, the new model parameters and new local contribution
        '''

        if parameters is not None:
            self.set_parameters(parameters)

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)

    @abstractmethod
    def predict(self, x, parameters=None, hyperparameters=None):
        '''
        :param x: The data to make predictions about
        :param parameters: optinal updated model parameters
        :param hyperparameters: optional updated hyperparameters
        :return: the model's predictions of the data
        '''

        if parameters is not None:
            self.set_parameters(parameters)

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)

    def sample(self, x, parameters=None, hyperparameters=None):
        '''
        Sample from the output of the model. Useful for generating data
        :param x: The data to make predictions about
        :param parameters: optinal updated model parameters
        :param hyperparameters: optional updated hyperparameters
        :return: the model's predictions of the data
        '''
        pass

    @abstractmethod
    def get_incremental_log_record(self):
        """
        Log various things about the model in self.log. Flexible form.
        """
        pass

    @abstractmethod
    def get_incremental_sacred_record(self):
        """
        Log various things we may want to see in the sacred logs. Reduced form
        :return: A *flat* dictionary containing scalars of interest for the current state.
        """
        pass
