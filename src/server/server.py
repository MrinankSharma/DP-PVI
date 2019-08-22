import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import ray

import src.utils.numpy_nest_utils as np_nest
import src.utils.numpy_utils as B
from src.privacy.analysis import QueryWithLedger, OnlineAccountant, QueryWithPerClientLedger
from src.utils.numpy_nest_utils import structured_ndarrays_to_lists
from src.utils.yaml_string_dumper import YAMLStringDumper

pretty_dump = YAMLStringDumper()

logger = logging.getLogger(__name__)


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
        self.iterations = 0

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

    def get_clients(self):
        return self.clients

    def get_client_sacred_logs(self):
        client_sacred_logs = [client.log_sacred() for client in self.get_clients()]
        return client_sacred_logs

    def add_client(self, client):
        self.clients.append(client)

    def get_log(self):
        return self.log

    def log_update(self):
        """
        Log various things about the server in self.log. Flexible form.
        """
        self.log["params"].append(structured_ndarrays_to_lists(self.parameters))

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
        client_logs = [client.get_compiled_log() for client in self.clients]
        for i, log in enumerate(client_logs):
            final_log['client_' + str(i)] = log

        return final_log

    def get_model_predictions(self, data):
        return self.model.predict(data["x"])

    def get_num_iterations(self):
        return self.iterations

    def get_parameters(self):
        return self.parameters


class SyncronousPVIParameterServer(ParameterServer):

    def __init__(self, model_class, prior, max_iterations=100, clients_factories=None, hyperparameters=None,
                 metadata=None,
                 model_parameters=None, model_hyperparameters=None):
        clients = [factory() for factory in clients_factories]
        super().__init__(model_class, prior, clients=clients, hyperparameters=hyperparameters, metadata=metadata,
                         model_parameters=model_parameters, model_hyperparameters=model_hyperparameters)
        self.max_iterations = max_iterations

    def tick(self):
        if self.should_stop():
            return False

        lambda_old = self.parameters

        # delta_is = [client.compute_update.remote(lambda_old) for client in self.clients]
        logger.debug("Getting Client Updates")
        delta_is = [client.get_update(model_parameters=lambda_old) for client in self.clients]

        logger.debug("Received client updates")
        # print(delta_is)
        lambda_new = B.add_parameters(lambda_old, *delta_is)

        self.parameters = lambda_new
        # update the model parameters
        self.model.set_parameters(self.parameters)
        logger.debug(f"Iteration {self.iterations} complete.\nNew Parameters:\n {pretty_dump.dump(lambda_new)}\n")
        [client.set_metadata({"global_iteration": self.iterations}) for client in self.clients]

        self.log_update()

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


class DPSequentialIndividualPVIParameterServer(ParameterServer):

    def __init__(self, model_class, dp_query_class, accounting_dict, prior, max_iterations=100, client_factories=None,
                 hyperparameters=None, metadata=None, model_parameters=None, model_hyperparameters=None):

        clients = [factory() for factory in client_factories]
        super().__init__(model_class, prior, clients=clients, hyperparameters=hyperparameters, metadata=metadata,
                         model_parameters=model_parameters, model_hyperparameters=model_hyperparameters, )
        self.iterations = 0
        self.max_iterations = max_iterations

        num_clients = len(clients)
        dp_query = dp_query_class(**hyperparameters['dp_query_parameters'])
        # assume that we aren't benefiting from subsampling here
        self.dp_query_with_ledgers = QueryWithPerClientLedger(dp_query, num_clients, 1.0)

        self.query_global_state = self.dp_query_with_ledgers.initial_global_state()

        self.accountants = []
        # create accountants
        for i in range(num_clients):
            client_i_accountants = {}
            for k, v in accounting_dict.items():
                client_i_accountants[k] = OnlineAccountant(**v)
            self.accountants.append(client_i_accountants)

    def tick(self):
        if self.should_stop():
            return False

        lambda_old = self.parameters
        L = self.hyperparameters["L"]
        M = len(self.clients)

        # generate index
        c = np.random.choice(M, L, replace=False)
        logger.debug(f"Selected clients {c}")
        # delta_is = [client.compute_update.remote(lambda_old) for client in self.clients]

        delta_is = []
        client_params = []
        for indx, client in enumerate(self.clients):
            client_params.append(client.parameters)
            if indx in c:
                # selected to be updated
                delta_is.append(
                    client.get_update(model_parameters=lambda_old, model_hyperparameters=None, update_ti=False))

        # print(delta_is)
        sample_state = self.dp_query_with_ledgers.initial_sample_state(delta_is[0])
        sample_params = self.dp_query_with_ledgers.derive_sample_params(self.query_global_state)
        self.query_global_state = self.dp_query_with_ledgers.initial_global_state()

        derived_data = defaultdict(list)
        for indx, delta_i in enumerate(delta_is):
            sample_state = self.dp_query_with_ledgers.accumulate_record(sample_params, sample_state, delta_i)
            for k, v in self.dp_query_with_ledgers.get_record_derived_data().items():
                derived_data[k].append(v)

        delta_i_tilde, _ = self.dp_query_with_ledgers.get_noised_result(sample_state, self.query_global_state, c)
        lambda_new, delta_i_tilde = self.hyperparameters["lambda_postprocess_func"](lambda_old, delta_i_tilde,
                                                                                    client_params, c)

        self.parameters = lambda_new
        t_i_update = np_nest.map_structure(lambda x: np.divide(x, L), delta_i_tilde)
        formatted_ledgers = self.dp_query_with_ledgers.get_formatted_ledgers()

        logger.debug(f"l2 clipping norms: {derived_data}")
        for k, v in derived_data.items():
            # summarise statistics instead
            derived_data[k] = np.percentile(np.array(v), [10.0, 30.0, 50.0, 70.0, 90.0])

        for indx, client in enumerate(self.clients):
            client.set_metadata({"global_iteration": self.iterations})
            if indx in c:
                client.update_ti(t_i_update)
                for k, v in self.accountants[indx].items():
                    v.update_privacy(formatted_ledgers[indx])

        self.model.set_parameters(self.parameters)
        logger.debug(f"Iteration {self.iterations} complete.\nNew Parameters:\n {pretty_dump.dump(lambda_new)}\n")

        self.log_update()
        self.log["derived_data"].append(structured_ndarrays_to_lists(derived_data))

        self.iterations += 1

    def should_stop(self):
        if self.iterations > self.max_iterations:
            return True
        else:
            return False

    def get_default_hyperparameters(self):
        return {**super().get_default_hyperparameters(), "L": 1, "lambda_postprocess_func": lambda x: x}

    def get_default_metadata(self):
        return super().get_default_metadata()

    def log_update(self):
        super().log_update()

        for i in range(len(self.clients)):
            for k, v in self.accountants[i].items():
                self.log[f"client_{i}_{k}.epsilon"].append(v.privacy_bound[0])

    def log_sacred(self):
        log = defaultdict(list)
        for i in range(len(self.clients)):
            for k, v in self.accountants[i].items():
                log[f"client_{i}_{k}.epsilon"].append(v.privacy_bound[0])

        return log, self.iterations

    def get_compiled_log(self):
        """
        Get full log, including logs from each client
        :return: full log
        """
        final_log = {}
        server_log = self.get_log()
        client_logs = [client.get_compiled_log() for client in self.clients]
        formatted_ledgers = self.dp_query_with_ledgers.get_formatted_ledgers()

        for i in range(len(self.clients)):
            for k, v in self.accountants[i].items():
                server_log[f"client_{i}_{k}.delta"].append(v.privacy_bound[1])
            client_logs[i][f"ledger"] = formatted_ledgers[i]
            final_log['client_' + str(i)] = client_logs[i]
        final_log['server'] = server_log

        return final_log
