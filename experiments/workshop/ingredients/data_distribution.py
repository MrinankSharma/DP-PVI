import os
# set numpy environment variables
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import logging
import sys

import numpy as np
import torch
from sacred import Ingredient

dataset_dist_ingred = Ingredient('dataset_dist')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("Dataset Distribution")


@dataset_dist_ingred.config
def cfg():
    M = 10
    client_size_factor = 0
    class_balance_factor = 0
    dataset_seed = None

@dataset_dist_ingred.capture
def generate_dataset_distribution_func(_run, M, client_size_factor, class_balance_factor, dataset_seed):
    def dataset_distribution_function(_run, x, y, M, client_size_factor, class_balance_factor, dataset_seed):
        # this function ought to return a list of (x, y) tuples.
        # you need to set the seed in the main experiment file to ensure that this function becomes deterministic

        random_state = np.random.get_state()

        if dataset_seed is not None:
            np.random.seed(dataset_seed)

        if M % 2 != 0: raise ValueError('Num clients should be even for nice maths')

        N = x.shape[0]
        small_client_size = int(np.floor((1 - client_size_factor) * N/M))
        big_client_size = int(np.floor((1 + client_size_factor) * N/M))

        class_balance = np.mean(y < 0)

        small_client_class_balance = class_balance + (1 - class_balance) * class_balance_factor
        small_client_negative_class_size = int(np.floor(small_client_size * small_client_class_balance))
        small_client_positive_class_size = int(small_client_size - small_client_negative_class_size)

        if small_client_negative_class_size * M/2 > class_balance * N:
            raise ValueError(f'Not enough positive class instances to fill the small clients. Client size factor:{client_size_factor}, class balance factor:{class_balance_factor}')

        pos_inds = np.where(y > 0)
        neg_inds = np.where(y < 0)

        logger.debug(f'x shape {x.shape}')
        logger.debug(f'positive indices {pos_inds}')
        logger.debug(f'negative indices {neg_inds}')

        y_pos = y[pos_inds]
        y_neg = y[neg_inds]

        x_pos = x[pos_inds]
        x_neg = x[neg_inds]

        client_data = []

        # populate small classes
        for i in range(int(M/2)):
            client_x_pos = x_pos[:small_client_positive_class_size]
            x_pos = x_pos[small_client_positive_class_size:]
            client_y_pos = y_pos[:small_client_positive_class_size]
            y_pos = y_pos[small_client_positive_class_size:]

            client_x_neg = x_neg[:small_client_negative_class_size]
            x_neg = x_neg[small_client_negative_class_size:]
            client_y_neg = y_neg[:small_client_negative_class_size]
            y_neg = y_neg[small_client_negative_class_size:]

            client_x = np.concatenate([client_x_pos, client_x_neg])
            client_y = np.concatenate([client_y_pos, client_y_neg])

            shuffle_inds = np.random.permutation(client_x.shape[0])

            client_x = client_x[shuffle_inds, :]
            client_y = client_y[shuffle_inds]

            client_data.append({'x': client_x, 'y': client_y})

        # recombine remaining data and shuffle

        x = np.concatenate([x_pos, x_neg])
        y = np.concatenate([y_pos, y_neg])
        shuffle_inds = np.random.permutation(x.shape[0])

        x = x[shuffle_inds]
        y = y[shuffle_inds]

        # distribute among large clients
        for i in range(int(M/2)):
            client_x = x[:big_client_size]
            client_y = y[:big_client_size]

            x = x[big_client_size:]
            y = y[big_client_size:]

            client_data.append({'x': client_x, 'y': client_y})

        N_is = [data['x'].shape[0] for data in client_data]
        props_positive = [np.mean(data['y'] > 0) for data in client_data]

        np.random.set_state(random_state)

        logger.info(f'N_is {N_is}')
        logger.info(f'Props positive: {props_positive}')

        return client_data, N_is, props_positive, M

    return lambda x, y: dataset_distribution_function(_run, x, y, M, client_size_factor, class_balance_factor, dataset_seed)
