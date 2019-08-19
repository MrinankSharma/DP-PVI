import logging
import os
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
    rho = 1900
    sample_rho_noise_scale = 0
    inhomo_scale = 0


@dataset_dist_ingred.capture
def generate_dataset_distribution_func(M, rho, sample_rho_noise_scale, inhomo_scale):
    def dataset_distribution_function(x, y, M, rho, sample_rho_noise_scale, inhomo_scale):
        # this function ought to return a list of (x, y) tuples.
        # you need to set the seed in the main experiment file to ensure that this function becomes deterministic
        clients_data = []
        prop_positive = []
        N = x.shape[0]
        if M * rho > N:
            rho = np.floor(N / M)

        indx = np.arange(0, N)
        logger.info(f"M={M}, rho={rho}")

        N_is = np.floor(np.random.laplace(loc=rho, scale=sample_rho_noise_scale, size=M))
        while np.sum(N_is) > N or np.min(N_is) < 1:
            logger.warning(f"Having to regenerate client dataset sizes - perhaps silly settings used?\n"
                           f"Note that rho is automatically scaled back if needed")
            N_is = np.floor(np.random.normal(loc=rho, scale=sample_rho_noise_scale, size=M))

        p_vals = np.ones(N)

        positive_indx = np.nonzero(y > 0)
        for i in range(M):
            p_vals[p_vals > 0] = 1
            target_multiplier_mag = np.exp(np.random.uniform(-inhomo_scale, inhomo_scale))
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

        return clients_data, N_is.tolist(), prop_positive, M

    return lambda x, y: dataset_distribution_function(x, y, M, rho, sample_rho_noise_scale, inhomo_scale)
