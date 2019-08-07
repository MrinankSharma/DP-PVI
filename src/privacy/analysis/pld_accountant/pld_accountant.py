""" Paper: https://arxiv.org/abs/1906.03049
    Referenced code from: https://github.com/DPBayes/PLD-Accountant """

import os
import pickle

import numpy as np
from ray.services import logger
from repoze.lru import lru_cache

from src.privacy.analysis.utils import grab_pickled_accountant_results


@lru_cache(maxsize=100)
def get_FF1_add_remove(sigma, q, nx, L):
    """ For a single Gaussian Mechanism, compute the FFT approximation points for
    the privacy loss distribution, under the addition/removal adjacency definition.
    :param sigma: The effective noise applied
    :param q: The sample probability
    :param nx: The number of approximation points
    :param L: The clipping length of the approximation
    :return: The FFT approximation points
    """

    try:
        filename = f"add_remove_{sigma}_{q}_{nx}_{L}.p"
        saved_result_flag, result, fp = grab_pickled_accountant_results(filename)
    except (AttributeError, EOFError, ImportError, IndexError, pickle.UnpicklingError):
        logger.error("Error reading accountant Pickle!")

    if saved_result_flag:
        return result

    # Evaluate the PLD distribution,
    # This is the case of substitution relation (subsection 5.1)
    if q == 1.0:
        q = 1 - 1E-5

    half = int(nx / 2)

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    ii = int(np.floor(float(nx * (L + np.log(1 - q)) / (2 * L))))

    # Evaluate the PLD distribution,
    # The case of remove/add relation (Subsection 5.1)
    Linvx = (sigma ** 2) * np.log((np.exp(x[ii + 1:]) - (1 - q)) / q) + 0.5
    ALinvx = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * ((1 - q) * np.exp(-Linvx * Linvx / (2 * sigma ** 2)) +
                                                      q * np.exp(-(Linvx - 1) * (Linvx - 1) / (2 * sigma ** 2)))
    dLinvx = (sigma ** 2 * np.exp(x[ii + 1:])) / (np.exp(x[ii + 1:]) - (1 - q))

    fx = np.zeros(nx)
    fx[ii + 1:] = np.real(ALinvx * dLinvx)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    FF1 = np.fft.fft(fx * dx)

    try:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, 'wb+') as dump:
            pickle.dump(FF1, dump)
    except (FileNotFoundError, pickle.PickleError, pickle.PicklingError):
        logger.error("Error with saving accountant pickle")

    return FF1


@lru_cache(maxsize=100)
def get_FF1_substitution(sigma, q, nx, L):
    """ For a single Gaussian Mechanism, compute the FFT approximation points for
    the privacy loss distribution, under the substitution adjacency definition.
    :param sigma: The effective noise applied
    :param q: The sample probability
    :param nx: The number of approximation points
    :param L: The clipping length of the approximation
    :return: The FFT approximation points
    """

    try:
        filename = f"sub_{sigma}_{q}_{nx}_{L}.p"
        saved_result_flag, result, fp = grab_pickled_accountant_results(filename)
    except (AttributeError, EOFError, ImportError, IndexError, pickle.UnpicklingError):
        logger.error("Error reading accountant Pickle!")

    if saved_result_flag:
        return result

    # Evaluate the PLD distribution,
    # This is the case of substitution relation (subsection 5.2)

    half = int(nx / 2)

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    c = q * np.exp(-1 / (2 * sigma ** 2))
    ey = np.exp(x)
    term1 = (-(1 - q) * (1 - ey) + np.sqrt((1 - q) ** 2 * (1 - ey) ** 2 + 4 * c ** 2 * ey)) / (2 * c)
    term1 = np.maximum(term1, 1e-16)
    Linvx = (sigma ** 2) * np.log(term1)

    sq = np.sqrt((1 - q) ** 2 * (1 - ey) ** 2 + 4 * c ** 2 * ey)
    nom1 = 4 * c ** 2 * ey - 2 * (1 - q) ** 2 * ey * (1 - ey)
    term1 = nom1 / (2 * sq)
    nom2 = term1 + (1 - q) * ey
    nom2 = nom2 * (sq + (1 - q) * (1 - ey))
    dLinvx = sigma ** 2 * nom2 / (4 * c ** 2 * ey)

    ALinvx = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * ((1 - q) * np.exp(-Linvx * Linvx / (2 * sigma ** 2)) +
                                                      q * np.exp(-(Linvx - 1) * (Linvx - 1) / (2 * sigma ** 2)))

    fx = np.real(ALinvx * dLinvx)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    FF1 = np.fft.fft(fx * dx)
    try:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, 'wb+') as dump:
            pickle.dump(FF1, dump)
    except (FileNotFoundError, pickle.PickleError, pickle.PicklingError):
        logger.error("Error with saving accountant pickle")

    return FF1


def get_delta_add_remove(effective_z_t, q_t, target_eps=1.0, nx=1E6, L=20.0, F_prod=None):
    """ Computes the approximation of the exact privacy as per https://arxiv.org/abs/1906.03049
    for a fixed epsilon, for a list of given mechanisms applied. Considers neighbouring sets
    as those with the addition/removal property.

    :param effective_z_t: 1D numpy array of the effective noises applied in the mechanisms.
    :param q_t: 1D numpy array of the selection probabilities of the data used in the mechanisms.
    :param target_eps: The target epsilon to aim for.
    :param nx: The number of discretisation points to use.
    :param L: The range truncation parameter
    :param F_prod: Specify a previous F_prod for computing online accountancy. If none, assume not
    online.
    :return: (epsilon, delta) privacy bound.
    """

    nx = int(nx)
    half = int(nx / 2)

    tol_newton = 1e-10  # set this to, e.g., 0.01*target_delta

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    fx_table = []
    if F_prod is None:
        F_prod = np.ones(x.size)

    ncomp = effective_z_t.size

    if (q_t.size != ncomp):
        logger.error('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):
        # Change from original: cache results for speedup on similar requests

        sigma = float(effective_z_t[ij])
        q = float(q_t[ij])

        FF1 = get_FF1_add_remove(sigma, q, nx, L)

        # Compute the DFT
        F_prod = F_prod * FF1

    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx * (L + target_eps) / (2 * L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod / dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1 - np.exp(target_eps - x)
    integrand = exp_e * cfx
    sum_int = np.sum(integrand[jj + 1:])
    delta = sum_int * dx

    # print('Unbounded DP-delta after ' + str(int(ncomp)) + ' compositions defined by sigma and q arrays:' + str(
    #     np.real(delta)) + ' (epsilon=' + str(target_eps) + ')')

    # Change from original: return signature consistent across methods to give epsilon and delta.
    return (target_eps, np.real(delta)), F_prod


def get_delta_substitution(effective_z_t, q_t, target_eps=1.0, nx=1E6, L=20.0, F_prod=None):
    """ Computes the approximation of the exact privacy as per https://arxiv.org/abs/1906.03049
    for a fixed epsilon, for a list of given mechanisms applied. Considers neighbouring sets
    as those with the substitution property.

    :param effective_z_t: 1D numpy array of the effective noises applied in the mechanisms.
    :param q_t: 1D numpy array of the selection probabilities of the data used in the mechanisms.
    :param target_eps: The target epsilon to aim for.
    :param nx: The number of discretisation points to use.
    :param L: The range truncation parameter
    :param F_prod: Specify a previous F_prod for computing online accountancy. If none, assume not
    online.
    :return: (epsilon, delta) privacy bound.
    """

    nx = int(nx)
    half = int(nx / 2)

    tol_newton = 1e-10  # set this to, e.g., 0.01*target_delta

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    fx_table = []
    if F_prod is None:
        F_prod = np.ones(x.size)

    ncomp = effective_z_t.size

    if (q_t.size != ncomp):
        logger.error('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):
        # Change from original: cache results for speedup on similar requests

        sigma = effective_z_t[ij]
        q = q_t[ij]

        FF1 = get_FF1_substitution(sigma, q, nx, L)
        F_prod = F_prod * FF1

    # first jj for which 1-exp(target_eps-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx * (L + np.real(target_eps)) / (2 * L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod / dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(target_eps) and \delta'(target_eps)
    exp_e = 1 - np.exp(target_eps - x)
    integrand = exp_e * cfx
    sum_int = np.sum(integrand[jj + 1:])
    delta = sum_int * dx

    # print('Bounded DP-delta after ' + str(int(ncomp)) + ' compositions defined by sigma and q arrays:' + str(np.real(delta)) + ' (epsilon=' + str(target_eps) + ')')
    # Change from original: return signature consistent across methods to give epsilon and delta.
    return (target_eps, np.real(delta)), F_prod


def get_eps_add_remove(effective_z_t, q_t, target_delta=1e-6, nx=1E6, L=20.0, F_prod=None):
    """ Computes the approximation of the exact privacy as per https://arxiv.org/abs/1906.03049
    for a fixed epsilon, for a list of given mechanisms applied. Considers neighbouring sets
    as those with the addition/removal property.

    :param effective_z_t: 1D numpy array of the effective noises applied in the mechanisms.
    :param q_t: 1D numpy array of the selection probabilities of the data used in the mechanisms.
    :param target_delta: The target delta to aim for.
    :param nx: The number of discretisation points to use.
    :param L: The range truncation parameter
    :param F_prod: Specify a previous F_prod for computing online accountancy. If none, assume not
    online.
    :return: (epsilon, delta) privacy bound.
    """

    nx = int(nx)
    half = int(nx / 2)

    tol_newton = 1e-10  # set this to, e.g., 0.01*target_delta

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    fx_table = []

    if F_prod is None:
        F_prod = np.ones(x.size)

    ncomp = effective_z_t.size

    if (q_t.size != ncomp):
        logger.error('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):
        # Change from original: cache results for speedup on similar requests
        sigma = float(effective_z_t[ij])
        q = float(q_t[ij])

        # this isn't doing the right thing!!
        FF1 = get_FF1_add_remove(sigma, q, nx, L)

        # Compute the DFT
        F_prod = F_prod * FF1

    # Initial value \epsilon_0
    eps_0 = 0

    exp_e = 1 - np.exp(eps_0 - x)
    # first jj for which 1-exp(eps_0-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx * (L + eps_0) / (2 * L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod / dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(eps_0) and \delta'(eps_0)
    exp_e = 1 - np.exp(eps_0 - x)
    dexp_e = -np.exp(eps_0 - x)
    integrand = exp_e * cfx
    integrand2 = dexp_e * cfx
    sum_int = np.sum(integrand[jj + 1:])
    sum_int2 = np.sum(integrand2[jj + 1:])
    delta_temp = sum_int * dx
    derivative = sum_int2 * dx

    # Here tol is the stopping criterion for Newton's iteration
    # e.g., 0.1*delta value or 0.01*delta value (relative error small enough)
    while np.abs(delta_temp - target_delta) > tol_newton:

        # print('Residual of the Newton iteration: ' + str(np.abs(delta_temp - target_delta)))

        # Update epsilon
        eps_0 = eps_0 - (delta_temp - target_delta) / derivative

        if (eps_0 < -L or eps_0 > L):
            break

        # Integrands and integral domain
        exp_e = 1 - np.exp(eps_0 - x)
        dexp_e = -np.exp(eps_0 - x)
        # first kk for which 1-exp(eps_0-x)>0,
        # i.e. start of the integral domain
        kk = int(np.floor(float(nx * (L + np.real(eps_0)) / (2 * L))))

        integrand = exp_e * cfx
        sum_int = np.sum(integrand[kk + 1:])
        delta_temp = sum_int * dx

        # Evaluate \delta(eps_0) and \delta'(eps_0)
        integrand = exp_e * cfx
        integrand2 = dexp_e * cfx
        sum_int = np.sum(integrand[kk + 1:])
        sum_int2 = np.sum(integrand2[kk + 1:])
        delta_temp = sum_int * dx
        derivative = sum_int2 * dx

    if (np.real(eps_0) < -L or np.real(eps_0) > L):
        logger.error('Epsilon out of [-L,L] window, please check the parameters.')
        return (float('inf'), float('inf')), F_prod
    else:
        # print('Unbounded DP-epsilon after ' + str(int(ncomp)) + ' compositions defined by sigma and q arrays: ' + str(np.real(eps_0)) + ' (delta=' + str(target_delta) + ')')
        return (np.real(eps_0), target_delta), F_prod


def get_eps_substitution(effective_z_t, q_t, target_delta=1e-6, nx=1E6, L=20.0, F_prod=None):
    """ Computes the approximation of the exact privacy as per https://arxiv.org/abs/1906.03049
    for a fixed epsilon, for a list of given mechanisms applied. Considers neighbouring sets
    as those with the substitution property.

    :param effective_z_t: 1D numpy array of the effective noises applied in the mechanisms.
    :param q_t: 1D numpy array of the selection probabilities of the data used in the mechanisms.
    :param target_delta: The target delta to aim for.
    :param nx: The number of discretisation points to use.
    :param L: The range truncation parameter
    :param F_prod: Specify a previous F_prod for computing online accountancy. If none, assume not
    online.
    :return: (epsilon, delta) privacy bound.
    """

    nx = int(nx)
    half = int(nx / 2)

    tol_newton = 1e-10  # set this to, e.g., 0.01*target_delta

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    # Initial value \epsilon_0
    eps_0 = 0

    fx_table = []
    if F_prod is None:
        F_prod = np.ones(x.size)

    ncomp = effective_z_t.size

    if (q_t.size != ncomp):
        logger.error('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):
        sigma = effective_z_t[ij]
        q = q_t[ij]

        FF1 = get_FF1_substitution(sigma, q, nx, L)
        F_prod = F_prod * FF1

    exp_e = 1 - np.exp(eps_0 - x)

    # first jj for which 1-exp(eps_0-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx * (L + np.real(eps_0)) / (2 * L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((F_prod / dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(eps_0) and \delta'(eps_0)
    exp_e = 1 - np.exp(eps_0 - x)
    dexp_e = -np.exp(eps_0 - x)
    integrand = exp_e * cfx
    integrand2 = dexp_e * cfx
    sum_int = np.sum(integrand[jj + 1:])
    sum_int2 = np.sum(integrand2[jj + 1:])
    delta_temp = sum_int * dx
    derivative = sum_int2 * dx

    # Here tol is the stopping criterion for Newton's iteration
    # e.g., 0.1*delta value or 0.01*delta value (relative error small enough)
    while np.abs(delta_temp - target_delta) > tol_newton:

        # print('Residual of the Newton iteration: ' + str(np.abs(delta_temp - target_delta)))

        # Update epsilon
        eps_0 = eps_0 - (delta_temp - target_delta) / derivative

        if (eps_0 < -L or eps_0 > L):
            break

        # Integrands and integral domain
        exp_e = 1 - np.exp(eps_0 - x)
        dexp_e = -np.exp(eps_0 - x)

        # first kk for which 1-exp(eps_0-x)>0,
        # i.e. start of the integral domain
        kk = int(np.floor(float(nx * (L + np.real(eps_0)) / (2 * L))))

        integrand = exp_e * cfx
        sum_int = np.sum(integrand[kk + 1:])
        delta_temp = sum_int * dx

        # Evaluate \delta(eps_0) and \delta'(eps_0)
        integrand = exp_e * cfx
        integrand2 = dexp_e * cfx
        sum_int = np.sum(integrand[kk + 1:])
        sum_int2 = np.sum(integrand2[kk + 1:])
        delta_temp = sum_int * dx
        derivative = sum_int2 * dx

    if (np.real(eps_0) < -L or np.real(eps_0) > L):
        logger.error('Epsilon out of [-L,L] window, please check the parameters.')
        return (float('inf'), float('inf')), F_prod
    else:
        # print('Bounded DP-epsilon after ' + str(int(ncomp)) + ' compositions defined by sigma and q arrays: ' + str(
        #     np.real(eps_0)) + ' (delta=' + str(target_delta) + ')')
        return (np.real(eps_0), target_delta), F_prod
    #
    # print('Bounded DP-epsilon after ' + str(int(ncomp)) + ' compositions:' + str(np.real(eps_0)) + ' (delta=' + str(target_delta) + ')')
    # return np.real(eps_0)


def get_eps_add_remove_fixed_params(target_delta=1e-6, sigma=2.0, q=0.01, ncomp=1E4, nx=1E6, L=20.0):
    nx = int(nx)

    tol_newton = 1e-10  # set this to, e.g., 0.01*target_delta

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    # first ii for which x(ii+1)>log(1-q),
    # i.e. start of the integral domain
    ii = int(np.floor(float(nx * (L + np.log(1 - q)) / (2 * L))))

    # Initial value \epsilon_0
    eps_0 = 0

    # Evaluate the PLD distribution,
    # The case of remove/add relation (Subsection 5.1)
    ey = np.exp(x[ii + 1:])
    Linvx = (sigma ** 2) * np.log((np.exp(x[ii + 1:]) - (1 - q)) / q) + 0.5

    ALinvx = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * ((1 - q) * np.exp(-Linvx * Linvx / (2 * sigma ** 2)) +
                                                      q * np.exp(-(Linvx - 1) * (Linvx - 1) / (2 * sigma ** 2)));
    dLinvx = (sigma ** 2) * ey / (ey - (1 - q));

    fx = np.zeros(nx)
    fx[ii + 1:] = np.real(ALinvx * dLinvx)
    half = int(nx / 2)

    # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
    temp = np.copy(fx[half:])
    fx[half:] = np.copy(fx[:half])
    fx[:half] = temp

    # Compute the DFT
    FF1 = np.fft.fft(fx * dx)

    exp_e = 1 - np.exp(eps_0 - x)
    # Find first jj for which 1-exp(eps_0-x)>0,
    # i.e. start of the integral domain
    jj = int(np.floor(float(nx * (L + eps_0) / (2 * L))))

    # Compute the inverse DFT
    cfx = np.fft.ifft((FF1 ** ncomp / dx))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    temp = np.copy(cfx[half:])
    cfx[half:] = cfx[:half]
    cfx[:half] = temp

    # Evaluate \delta(eps_0) and \delta'(eps_0)
    exp_e = 1 - np.exp(eps_0 - x)
    dexp_e = -np.exp(eps_0 - x)
    integrand = exp_e * cfx
    integrand2 = dexp_e * cfx
    sum_int = np.sum(integrand[jj + 1:])
    sum_int2 = np.sum(integrand2[jj + 1:])
    delta_temp = sum_int * dx
    derivative = sum_int2 * dx

    # Here tol is the stopping criterion for Newton's iteration
    # e.g., 0.1*delta value or 0.01*delta value (relative error small enough)

    while np.abs(delta_temp - target_delta) > tol_newton:
        # print('Residual of the Newton iteration: ' + str(np.abs(delta_temp - target_delta)))

        # Update epsilon
        eps_0 = eps_0 - (delta_temp - target_delta) / derivative

        if (eps_0 < -L or eps_0 > L):
            break

        # Integrands and integral domain
        exp_e = 1 - np.exp(eps_0 - x)
        dexp_e = -np.exp(eps_0 - x)
        # Find first kk for which 1-exp(eps_0-x)>0,
        # i.e. start of the integral domain
        kk = int(np.floor(float(nx * (L + np.real(eps_0)) / (2 * L))))

        integrand = exp_e * cfx
        sum_int = np.sum(integrand[kk + 1:])
        delta_temp = sum_int * dx

        # Evaluate \delta(eps_0) and \delta'(eps_0)
        integrand = exp_e * cfx
        integrand2 = dexp_e * cfx
        sum_int = np.sum(integrand[kk + 1:])
        sum_int2 = np.sum(integrand2[kk + 1:])
        delta_temp = sum_int * dx
        derivative = sum_int2 * dx

    if (np.real(eps_0) < -L or np.real(eps_0) > L):
        print('Error: epsilon out of [-L,L] window, please check the parameters.')
        return float('inf'), target_delta
    else:
        print(
            'Add-Remove DP-epsilon after ' + str(int(ncomp)) + ' compositions:' + str(np.real(eps_0)) + ' (delta=' + str(
                target_delta) + ')')
        return np.real(eps_0)


def compute_privacy_loss_from_ledger(ledger, target_eps=None, target_delta=None, adjacency_definition='add_remove',
                                     nx=1E6, L=50.0):
    """ Compute the privacy loss of the queries entered into a ledger
    using the Gaussian Mechanism utilising an approximation to the true privacy bound
    (https://arxiv.org/abs/1906.03049) with one of epsilon or delta fixed.

    :param ledger: The ledger of queries to compute for
    :param target_eps: A target epsilon to aim for.
    :param target_delta: A target delta to aim for.
    :param adjacency_definition: The definition of adjacent datasets to use. Can be
    'add_remove' or 'substitution'
    :param nx: The number of discretisation points to use.
    :param L: The range truncation parameter
    :return: epsilon, delta
    """

    effective_z_t = []
    q_t = []

    for sample in ledger:
        # note this specific effective z calculation allows for different scale factors to be applied!
        effective_z = sum([
            (q.noise_stddev / q.l2_norm_bound) ** -2 for q in sample.queries
        ]) ** -0.5

        effective_z_t.append(effective_z)
        q_t.append(sample.selection_probability)

    effective_z_t = np.array(effective_z_t)
    q_t = np.array(q_t)

    if target_delta is None and target_eps is None:
        raise ValueError(
            "One of the target values must not be None")

    if target_eps is not None and target_delta is not None:
        raise ValueError(
            "Exactly one out of eps and delta must be None. (None is).")

    if adjacency_definition is 'add_remove':
        if target_eps is not None:
            privacy_bound, _ = get_delta_add_remove(effective_z_t, q_t, target_eps=target_eps, nx=nx, L=L)
            return privacy_bound
        else:
            privacy_bound, _ = get_eps_add_remove(effective_z_t, q_t, target_delta=target_delta, nx=nx, L=L)
            return privacy_bound

    elif adjacency_definition is 'substitution':
        if target_eps is not None:
            privacy_bound, _ = get_delta_substitution(effective_z_t, q_t, target_eps=target_eps, nx=nx, L=L)
            return privacy_bound
        else:
            privacy_bound, _ = get_eps_substitution(effective_z_t, q_t, target_delta=target_delta, nx=nx, L=L)
            return privacy_bound

    raise ValueError('adjacency_definition must be one of "substitution" or "add_remove".')


def compute_online_privacy_from_ledger(ledger, F_prod,
                                       target_delta=None, target_eps=None,
                                       adjacency_definition='add_remove', nx=1E6, L=50.0):
    """ Compute new PLD privacy in an online fashion, to speed up computation.

    :param ledger: The ledger of queries to compute for. An incremental ledger,
    NOT the whole ledger.
    :param target_eps: A target epsilon to aim for.
    :param target_delta: A target delta to aim for.
    :param adjacency_definition: The definition of adjacent datasets to use. Can be
    'add_remove' or 'substitution'
    :param nx: The number of discretisation points to use.
    :param L: The range truncation parameter
    :return:
    """

    if ledger == [] and F_prod is None:
        return tuple([None, None]), None

    effective_z_t = []
    q_t = []

    for sample in ledger:
        # note this specific effective z calculation allows for different scale factors to be applied!
        effective_z = sum([
            (q.noise_stddev / q.l2_norm_bound) ** -2 for q in sample.queries
        ]) ** -0.5

        effective_z_t.append(effective_z)
        q_t.append(sample.selection_probability)

    effective_z_t = np.array(effective_z_t)
    q_t = np.array(q_t)

    if target_delta is None and target_eps is None:
        raise ValueError(
            "One of the target values must not be None")

    if target_eps is not None and target_delta is not None:
        raise ValueError(
            "Exactly one out of eps and delta must be None. (None is).")

    if adjacency_definition is 'add_remove':
        if target_eps is not None:
            privacy_bound, F_prod = get_delta_add_remove(effective_z_t, q_t, target_eps=target_eps, nx=nx, L=L,
                                                         F_prod=F_prod)
            return privacy_bound, F_prod
        else:
            privacy_bound, F_prod = get_eps_add_remove(effective_z_t, q_t, target_delta=target_delta, nx=nx, L=L,
                                                       F_prod=F_prod)
            return privacy_bound, F_prod

    if adjacency_definition is 'substitution':
        if target_eps is not None:
            privacy_bound, F_prod = get_delta_substitution(effective_z_t, q_t, target_eps=target_eps, nx=nx, L=L,
                                                           F_prod=F_prod)
            return privacy_bound, F_prod
        else:
            privacy_bound, F_prod = get_eps_substitution(effective_z_t, q_t, target_delta=target_delta, nx=nx, L=L,
                                                         F_prod=F_prod)
            return privacy_bound, F_prod

    raise ValueError('adjacency_definition must be one of "substitution" or "add_remove".')
