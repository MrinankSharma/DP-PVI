class Model(object):
    def __init__(self, parameters, hyperparameters):
        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

    def set_parameters(self, parameters):
        pass

    def set_hyperparameters(self, hyperparameters):
        pass

    def fit(self, data, t_i, parameters=None, hyperparameters=None):
        '''
        :param data: The local data to refine the model with
        :param t_i: The local contribution of the client
        :param parameters: optinal updated model parameters
        :param hyperparameters: optional updated hyperparameters
        :return: lambda_i, t_i, the new model parameters and new local contribution
        '''

        if parameters is not None:
            self.set_parameters(parameters)

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)

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