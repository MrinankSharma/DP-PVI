from abc import ABC, abstractmethod
import ray

@ray.remote
class Model(ABC):
    def __init__(self, parameters, hyperparameters):
        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

    def set_parameters(self, parameters):
        self.parameters = parameters

    @abstractmethod
    def get_default_parameters(self):
        '''
        :return: A default set of parameters for the model. These might be all zero. Mostly used to get the shape that
        the parameters should be to make parameter server code more general.
        '''

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def fit(self, data, t_i, parameters=None, hyperparameters=None):
        '''
        :param data: The local data to refine the model with
        :param t_i: The local contribution of the client
        :param parameters: optinal updated model parameters
        :param hyperparameters: optional updated hyperparameters
        :return: lambda_i_new, t_i_new, the new model parameters and new local contribution
        '''

        if parameters is not None:
            self.set_parameters(**parameters)

        if hyperparameters is not None:
            self.set_hyperparameters(**hyperparameters)

    def predict(self, x, parameters=None, hyperparameters=None):
        '''
        :param x: The data to make predictions about
        :param parameters: optinal updated model parameters
        :param hyperparameters: optional updated hyperparameters
        :return: the model's predictions of the data
        '''

        if parameters is not None:
            self.set_parameters(**parameters)

        if hyperparameters is not None:
            self.set_hyperparameters(**hyperparameters)

    def add_parameters(self, *params):
        '''
        Assuming a list of dicts of parameter tensors, add each corresponding element from each together
        :param params: list of dicts with the same keys of tensors
        :return: a dict containing the addition of these parameters
        '''
        return_dict = {}

        for key in params[0].keys:
            value = params[0][key]
            for i in range(len(params) - 1):
                value = value + params[i+1][key]
            return_dict[key] = value

        return return_dict

    def subtract_params(self, *params):
        '''
        Assuming a list of dicts of parameter tensors, subtract each corresponding element from the first element
        :param params: list of dicts with the same keys of tensors
        :return: a dict containing the addition of these parameters
        '''

        return_dict = {}

        for key in params[0].keys:
            value = params[0][key]
            for i in range(len(params) - 1):
                value = value - params[i + 1][key]
            return_dict[key] = value

        return return_dict