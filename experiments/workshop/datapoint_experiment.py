import os
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

# set numpy environment variables
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6

import datetime
import logging
import time

import numpy as np
import ray
# ray must be imported before pyarrow
import pyarrow

import torch
from sacred import Experiment

import src.privacy.analysis.moment_accountant as moment_accountant
import src.utils.numpy_nest_utils as numpy_nest
# noinspection PyUnresolvedReferences
from experiments.workshop.MongoDBOption import TestOption, ExperimentOption, DatabaseOption
from experiments.workshop.ingredients.data_distribution import dataset_dist_ingred, generate_dataset_distribution_func
from experiments.workshop.ingredients.dataset_ingredient import dataset_ingredient, load_data
from experiments.workshop.measure_performance import compute_prediction_accuracy, compute_log_likelihood
from experiments.utils import save_log, save_pickle
from src.client.client import DPClient
from src.model.logistic_regression_models import MeanFieldMultiDimensionalLogisticRegression, \
    postprocess_MF_logistic_ti, nat_params_to_params_dict,  postprocess_MF_logistic_ti_simple
from src.privacy.dp_query import GaussianDPQuery
from src.privacy.optimizer import DPOptimizer
from src.server import AsynchronousParameterServer
from src.utils.yaml_string_dumper import YAMLStringDumper

ex = Experiment('datapoint_dp_pvi', [dataset_ingredient, dataset_dist_ingred])
logger = logging.getLogger(__name__)
pretty_dump = YAMLStringDumper()


@ex.config
def default_config(dataset, dataset_dist):
    experiment_tag = 'datapoint_dp_pvi'

    PVI_settings = {
        'damping_factor': 0.1,
        'damping_decay': 0,
        'async': True
    }

    privacy_settings = {
        'sigma_relative': 1e-5,
        'C': 1000,
        'target_delta': 'adaptive',
        'q': 0,
        'max_epsilon': None
    }

    dataset = {
        'name': 'adult',
        'scaled': True,
        'ordinal_cat_encoding': False,
        'train_proportion': 0.8,
        'data_base_dir': 'data',
    }

    dataset_dist = {
        'M': 10,
        'client_size_factor': 0,
        'class_balance_factor': 0,
        'dataset_seed': None,
    }

    optimisation_settings = {
        'lr': 0.2,
        'N_steps': 200,
        'lr_decay': 0,
        'L_min': 10,
    }

    ray_cfg = {
        'redis_address': None,
        'num_cpus': 1,
        'num_gpus': 0,
    }

    prediction = {
        'interval': 1,
        'type': 'prohibit',
    }

    N_iterations = 50
    prior_pres = 1.0
    N_samples = 50

    t_i_pres_init = 4000

    log_level = 'info'
    save_t_is = False
    logging_base_directory = 'logs'
    slack_json_file = 'slack.json'


@ex.automain
def run_experiment(ray_cfg, prior_pres, privacy_settings, optimisation_settings, PVI_settings, N_samples, N_iterations,
                   prediction,
                   experiment_tag, logging_base_directory,  save_t_is,
                   _run, _config, seed, t_i_pres_init):
    torch.set_num_threads(int(ray_cfg["num_cpus"]))
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        training_set, test_set, d_in = load_data()
        clients_data, nis, prop_positive, M = generate_dataset_distribution_func()(training_set["x"], training_set["y"])

        _run.info = {
            **_run.info,
            "prop_positive": prop_positive,
            "n_is": nis,
        }

        logger.info(f"Proportions Positive Are: {prop_positive}")
        logger.info(f"Num datapoints are: {nis}")

        if ray_cfg["redis_address"] is None:
            logger.info("Creating new ray server")
            ray.init(num_cpus=ray_cfg["num_cpus"], num_gpus=ray_cfg["num_gpus"], logging_level=logging.INFO,
                     local_mode=True)
        else:
            logger.info("Connecting to existing server")
            ray.init(redis_address=ray_cfg["redis_address"], logging_level=logging.INFO)

        prior_params = {
            "w_nat_mean": np.zeros(d_in, dtype=np.float32),
            "w_pres": prior_pres * np.ones(d_in, dtype=np.float32)
        }

        init_params = {
             "w_nat_mean": np.zeros(d_in, dtype=np.float32),
             "w_pres": (prior_pres + M*t_i_pres_init) * np.ones(d_in, dtype=np.float32)
        }

        logger.debug(f"Prior Parameters:\n\n{pretty_dump.dump(prior_params)}\n")

        if privacy_settings['target_delta'] == 'adaptive':
            deltas = []
            for data in clients_data:
                delta = 10 ** (float(np.floor(np.log10(1 / data['x'].shape[0]))))
                deltas.append(delta)
        else:
            deltas = [privacy_settings['target_delta']] * len(clients_data)

        model = MeanFieldMultiDimensionalLogisticRegression

        # client factories for each client - this avoids pickling of the client object for ray internals
        client_factories = [DPClient.create_factory(
            model_class=model,
            dp_query_class=GaussianDPQuery,
            data=clients_data[i],
            accounting_dict={
                'MomentAccountant': {
                    'accountancy_update_method': moment_accountant.compute_online_privacy_from_ledger,
                    'accountancy_parameters': {
                        'target_delta': deltas[i]
                    }
                }
            },
            model_parameters=init_params,
            model_hyperparameters={
                "base_optimizer_class": torch.optim.Adagrad,
                "wrapped_optimizer_class": DPOptimizer,
                "base_optimizer_parameters": {'lr': optimisation_settings["lr"]},
                "wrapped_optimizer_parameters": {},
                "N_steps": optimisation_settings["N_steps"],
                "N_samples": N_samples,
                "n_in": d_in,
                "batch_size": int(np.max(
                    [optimisation_settings['L_min'], np.ceil(privacy_settings['q'] * clients_data[i]['x'].shape[0])])),
                "reset_optimiser": True,
            },
            hyperparameters={
                'dp_query_parameters': {
                    'l2_norm_clip': privacy_settings["C"],
                    'noise_stddev': privacy_settings["C"] * privacy_settings["sigma_relative"]
                },
                't_i_init_function': lambda: {"w_nat_mean": np.zeros(d_in),
                                      "w_pres": t_i_pres_init*np.ones(d_in)},
                't_i_postprocess_function': postprocess_MF_logistic_ti_simple,
                'max_epsilon': privacy_settings['max_epsilon'],
            },
            metadata={
                'client_index': i,
                'test_self': {
                    'accuracy': compute_prediction_accuracy,
                    'log_lik': compute_log_likelihood
                }
            }
        ) for i in range(M)]

        # custom decorator based on passed in resources!
        remote_decorator = ray.remote(num_cpus=int(ray_cfg["num_cpus"]), num_gpus=int(ray_cfg["num_gpus"]))
        server = remote_decorator(AsynchronousParameterServer).remote(
            model_class=model,
            model_parameters=init_params,
            model_hyperparameters={
                "prediction": prediction["type"],
            },
            prior=prior_params,
            client_factories=client_factories,
            max_iterations=N_iterations,
            hyperparameters={
                "damping_factor": PVI_settings['damping_factor'],
                "damping_decay": PVI_settings['damping_decay']
            }
        )

        total_communications = 0

        while not ray.get(server.should_stop.remote()):
            # dispatch work to ray and grab the log
            st_tick = time.time()
            communications_this_round = ray.get(server.tick.remote())
            total_communications += communications_this_round
            num_iterations = ray.get(server.get_num_iterations.remote())

            st_log = time.time()
            sacred_log = {}
            sacred_log["server"], _ = ray.get(server.log_sacred.remote())
            params = ray.get(server.get_parameters.remote())
            moment_params = nat_params_to_params_dict(params)
            sacred_log[f"mean_mu"] = np.mean(np.abs(moment_params["w_mu"]))
            sacred_log[f"mean_log_var"] = -np.mean(np.abs(moment_params["w_log_var"]))

            client_sacred_logs = ray.get(server.get_client_sacred_logs.remote())
            for i, log in enumerate(client_sacred_logs):
                sacred_log["client_" + str(i)] = log[0]
            sacred_log = numpy_nest.flatten(sacred_log, sep=".")

            st_pred = time.time()
            # predict every interval, and also for the last "interval" runs.
            if ((num_iterations - 1) % prediction["interval"] == 0) or (
                    N_iterations - num_iterations < prediction["interval"]):
                # y_pred_train = ray.get(server.get_model_predictions.remote(training_set))
                y_pred_test = ray.get(server.get_model_predictions.remote(test_set))
                # sacred_log["train_all"] = compute_log_likelihood(y_pred_train, training_set["y"])
                # sacred_log["train_accuracy"] = compute_prediction_accuracy(y_pred_train, training_set["y"])
                sacred_log["test_all"] = compute_log_likelihood(y_pred_test, test_set["y"])
                test_acc = compute_prediction_accuracy(y_pred_test, test_set["y"])
                sacred_log["test_accuracy"] = test_acc
            end_pred = time.time()

            for k, v in sacred_log.items():
                _run.log_scalar(k + '_time', v, num_iterations)
                _run.log_scalar(k + '_communications', v, total_communications)
            end = time.time()

            logger.info(f"Server Ticket Complete\n"
                        f"Server Timings:\n"
                        f"  Server Tick: {st_log - st_tick:.2f}s\n"
                        f"  Predictions: {end_pred - st_pred:.2f}s\n"
                        f"  Logging:     {end - end_pred + st_pred - st_log:.2f}s\n\n"
                        f"Iteration Number:{num_iterations}\n")
            logger.debug(f"Parameters:\n"
                         f" {pretty_dump.dump(params)}\n")

        final_log = ray.get(server.get_compiled_log.remote())
        final_log["N_i"] = nis
        final_log["Proportion_positive"] = prop_positive
        t = datetime.datetime.now()

        ex.add_artifact(
            save_log(final_log, "full_log", ex.get_experiment_info()["name"], experiment_tag, logging_base_directory,
                     _run.info["test"], t), 'full_log.json')

        if save_t_is:
            t_is = [client.t_i for client in ray.get(server.get_clients.remote())]
            ex.add_artifact(save_pickle(
                t_is, 't_is', ex.get_experiment_info()["name"], experiment_tag, logging_base_directory,
                _run.info["test"], t
            ), 't_is.pkl')

        return test_acc

    except pyarrow.lib.ArrowIOError:
        raise Exception("Experiment Terminated - was this you?")
