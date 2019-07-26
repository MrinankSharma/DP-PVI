import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import ray

import src.utils.numpy_utils as B
from src.privacy_accounting.analysis import QueryWithLedger, OnlineAccountant
from src.utils.yaml_string_dumper import YAMLStringDumper

logger = logging.getLogger(__name__)
pretty_dump = YAMLStringDumper()


class ParameterServer(ABC):
    def __init__(self, model_class, prior, clients=None, hyperparameters=None, metadata=None, model_parameters=None,
                 model_hyperparameters=None, ):

        if hyperparameters is None:
            hyperparameters = {}

        if metadata is None:
            metadata = {}

        self.hyperparameters = self.get_default_hyperparameters()
        self.metadata = self.get_default_metadata()

        self.set_hyperparameters(hyperparameters)
        self.set_metadata(metadata)

        self.set_clients(clients)
        self.model = model_class(parameters=model_parameters, hyperparameters=model_hyperparameters)
        self.prior = prior
        self.parameters = prior

        self.log = defaultdict(list)

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
        if clients is None:
            self.clients = []
        else:
            self.clients = clients

    def add_client(self, client):
        self.clients.append(client)

    def get_log(self):
        return self.log

    @abstractmethod
    def log_update(self):
        """
        Log various things about the server in self.log. Flexible form.
        """
        pass

    @abstractmethod
    def log_sacred(self):
        """
        Log various things we may want to see in the sacred logs. Reduced form
        :return: Nested dictionaries of scalars to log, the current iteration
        """
        pass

    def get_compiled_log(self):
        """
        Get full log, including logs from each client
        :return: full log
        """
        final_log = {}
        final_log['server'] = self.get_log()
        client_logs = [client.get_log() for client in self.clients]
        for i, log in enumerate(client_logs):
            final_log['client_' + str(i)] = log

        return final_log


class SyncronousPVIParameterServer(ParameterServer):

    def __init__(self, model_class, prior, max_iterations=100, clients=None, hyperparameters=None, metadata=None,
                 model_parameters=None, model_hyperparameters=None, ):
        super().__init__(model_class, prior, clients=clients, hyperparameters=hyperparameters, metadata=metadata,
                         model_parameters=model_parameters, model_hyperparameters=model_hyperparameters)
        self.iterations = 0
        self.max_iterations = max_iterations

    def tick(self):
        if self.should_stop():
            return False

        lambda_old = self.parameters

        # delta_is = [client.compute_update.remote(lambda_old) for client in self.clients]
        logger.info("Getting Client Updates")
        delta_is = [client.get_update(model_parameters=lambda_old) for client in self.clients]

        logger.info("Received client updates")
        # print(delta_is)

        lambda_new = B.add_parameters(lambda_old, *delta_is)

        self.parameters = lambda_new
        # update the model parameters
        self.model.set_parameters(self.parameters)
        logger.info(f"Iteration {self.iterations} complete.\nNew Parameters:\n {pretty_dump.dump(lambda_new)}\n")

        self.iterations += 1

    def should_stop(self):
        if self.iterations > self.max_iterations - 1:
            return True
        else:
            return False

    def get_default_hyperparameters(self):
        return super().get_default_hyperparameters()

    def get_default_metadata(self):
        return super().get_default_metadata()

    def log_update(self):
        pass

    def log_sacred(self):
        return {}, self.iterations


class DPSyncronousPVIParameterServer(ParameterServer):

    def __init__(self, model_class, dp_query_class, accounting_dict, prior, max_iterations=100, clients=None,
                 hyperparameters=None, metadata=None, model_parameters=None, model_hyperparameters=None, ):
        super().__init__(model_class, prior, clients=clients, hyperparameters=hyperparameters, metadata=metadata,
                         model_parameters=model_parameters, model_hyperparameters=model_hyperparameters, )
        self.iterations = 0
        self.max_iterations = max_iterations

        num_clients = 0 if clients is None else len(clients)
        dp_query = dp_query_class(**hyperparameters['dp_query_parameters'])
        self.dp_query = QueryWithLedger(dp_query, num_clients, 1.0 / num_clients)

        self.query_global_state = self.dp_query.initial_global_state()

        self.accountants = {}
        for k, v in accounting_dict.items():
            self.accountants[k] = OnlineAccountant(**v)

    def tick(self):
        if self.should_stop():
            return False

        lambda_old = self.parameters

        # delta_is = [client.compute_update.remote(lambda_old) for client in self.clients]
        delta_is = ray.get([client.get_update(model_parameters=lambda_old) for client in self.clients])

        sample_state = self.dp_query.initial_sample_state(delta_is[0])
        sample_params = self.dp_query.derive_sample_params(self.query_global_state)

        for delta_i in delta_is:
            sample_state = self.dp_query.accumulate_record(sample_params, sample_state, delta_i)

        delta_i_tilde, _ = self.dp_query.get_noised_result(sample_state)

        lambda_new = B.add_parameters(lambda_old, delta_i_tilde)

        self.parameters = lambda_new

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

    def log_update(self):
        super().log_update()

        for k, v in self.accountants.items():
            self.log[k].append(v.privacy_bound)

        self.log['ledger'].append(self.dp_query.ledger.get_formatted_ledger())

    def log_sacred(self):
        log, times_updated = super().log_sacred()

        for k, v in self.accountants.items():
            log[k + '.epsilon'] = v.privacy_bound[0]
            log[k + '.delta'] = v.privacy_bound[1]

        return log, times_updated
