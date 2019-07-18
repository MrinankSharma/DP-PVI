from abc import ABC, abstractmethod

import ray

import src.utils.numpy_backend as B


class ParameterServer(ABC):
    def __init__(self, model_class, prior, clients=[], hyperparameters=None, metadata=None):

        if hyperparameters is None:
            hyperparameters = {}

        if metadata is None:
            metadata = {}

        self.hyperparameters = self.get_default_hyperparameters()
        self.metadata = self.get_default_metadata()

        self.set_hyperparameters(hyperparameters)
        self.set_metadata(metadata)

        self.set_clients(clients)
        self.model = model_class()
        self.prior = prior
        self.parameters = prior

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    def set_metadata(self, metadata):
        self.metadata = {**self.metadata, **metadata}

    @abstractmethod
    def get_default_hyperparameters(self):
        return {}

    @abstractmethod
    def get_default_metadata(self):
        return {}

    @abstractmethod
    def tick(self):
        '''
        Defines what the Parameter Server should do on each update round. Could be all client synchronous updates,
        async updates, might check for new clients etc
        '''
        pass

    @abstractmethod
    def should_stop(self):
        '''
        Defines when the Parameter server should stop running. Might be a privacy limit, or anything else.'''
        pass

    def set_clients(self, clients):
        self.clients = clients

    def add_client(self, client):
        self.clients.append(client)


class SyncronousPVIParameterServer(ParameterServer):

    def __init__(self, model_class, prior, max_iterations=100, clients=[], hyperparameters=None, metadata=None):
        super().__init__(model_class, prior, clients=clients, hyperparameters=hyperparameters, metadata=metadata)
        self.iterations = 0
        self.max_iterations = max_iterations

    def tick(self):
        if self.should_stop():
            return False

        lambda_old = self.parameters

        # delta_is = [client.compute_update.remote(lambda_old) for client in self.clients]
        delta_is = ray.get([client.compute_update.remote(model_parameters=lambda_old) for client in self.clients])

        # print(delta_is)

        lambda_new = B.add_parameters(lambda_old, *delta_is)

        self.parameters = lambda_new

        print(lambda_old, delta_is)

        self.iterations += 1

    def should_stop(self):
        if self.iterations > self.max_iterations:
            return True
        else:
            return False

    def get_default_hyperparameters(self):
        return super().get_default_hyperparameters()

    def get_default_metadata(self):
        return super().get_default_metadata()


def clip_and_noise(parameters, bound, noise_sigma):
    for name in parameters.keys():
        parameters[name] = B.clip(parameters[name], bound) + B.gaussian_noise(parameters.shape, noise_sigma)
    return parameters


class SyncronousDPPVIParameterServer(ParameterServer):

    def __init__(self, model_class, prior, max_iterations=100, clients=[]):
        super().__init__(model_class, prior, clients=clients)
        self.iterations = 0
        self.max_iterations = max_iterations

        for client in self.clients: client.set_hyperparameters.remote(
            {'privacy_function': lambda x: clip_and_noise(x, self.clipping_bound, self.privacy_noise_std)}
        )

    def tick(self):
        if self.should_stop():
            return False

        lambda_old = self.parameters

        # delta_is = [client.compute_update.remote(lambda_old) for client in self.clients]
        delta_is = ray.get([client.compute_update.remote(model_parameters=lambda_old) for client in self.clients])

        # print(delta_is)

        lambda_new = B.add_parameters(lambda_old, *delta_is)

        self.parameters = lambda_new

        print(lambda_old, delta_is)

        self.iterations += 1

    def should_stop(self):
        if self.iterations > self.max_iterations:
            return True
        else:
            return False

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)

        self.clipping_bound = self.hyperparameters['clipping_bound']
        self.privacy_noise_std = self.hyperparameters['privacy_noise_std']

    def get_default_hyperparameters(self):
        return {
            **super().get_default_hyperparameters(),
            **{
                'clipping_bound': B.inf,
                'privacy_noise_std': 0
            }
        }

    def get_default_metadata(self):
        return super().get_default_metadata()
