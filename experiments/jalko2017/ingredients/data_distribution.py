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
    M = 20
    rho = 380
    sample_rho_noise_scale = 0
    inhomo_scale = 0
    dataset_seed = None


@dataset_dist_ingred.capture
def generate_dataset_distribution_func(_run, M, rho, sample_rho_noise_scale, inhomo_scale, dataset_seed):
    def dataset_distribution_function(_run, x, y, M, rho, sample_rho_noise_scale, inhomo_scale, dataset_seed):
        # this function ought to return a list of (x, y) tuples.
        # you need to set the seed in the main experiment file to ensure that this function becomes deterministic

        np_random_state = np.random.get_state()

        if dataset_seed is not None:
            print(dataset_seed, type(dataset_seed))
            np.random.seed(dataset_seed)
            # torch.manual_seed(dataset_seed)

        clients_data = []
        prop_positive = []
        N = x.shape[0]
        if M * rho > N:
            raise ValueError(f'Cannot distribute {N} datapoints between {M} clients with {rho} on each. Aborting')

        indx = np.arange(0, N)
        logger.info(f"M={M}, rho={rho}")

        N_is = np.floor(np.clip(np.random.laplace(loc=rho, scale=sample_rho_noise_scale, size=M), rho - 3*sample_rho_noise_scale, rho+3*sample_rho_noise_scale))
        while np.sum(N_is) > N or np.min(N_is) < 1:
            logger.warning(f"Having to regenerate client dataset sizes - perhaps silly settings used?\n"
                           f"Note that rho is automatically scaled back if needed")
            N_is = np.floor(np.random.normal(loc=rho, scale=sample_rho_noise_scale, size=M))

        p_vals = np.ones(N)

        positive_indx = np.nonzero(y > 0)
        for i in range(M):
            p_vals[p_vals > 0] = 1
            target_multiplier_mag = np.exp(np.clip(np.random.laplace(loc=0, scale=inhomo_scale), -3, 3))
            p_vals[positive_indx] = p_vals[positive_indx] * target_multiplier_mag

            # renormalise!
            p_vals = p_vals / np.sum(p_vals)
            indx_i = np.random.choice(indx, int(N_is[i]), replace=False, p=p_vals)
            p_vals[indx_i] = 0.0
            x_i = x[indx_i, :]
            y_i = y[indx_i]
            clients_data.append(({
                "x": x_i,
                "y": y_i
            }))
            prop_positive.append(np.mean(y_i > 0))

        _run.info = {
            **_run.info,
            "prop_positive": prop_positive,
            "n_is": N_is.tolist(),
        }

        np.random.set_state(np_random_state)

        return clients_data, N_is.tolist(), prop_positive, M

    return lambda x, y: dataset_distribution_function(_run, x, y, M, rho, sample_rho_noise_scale, inhomo_scale, dataset_seed)
