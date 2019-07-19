""" Paper: https://arxiv.org/abs/1906.03049
    Referenced code from: https://github.com/DPBayes/PLD-Accountant """

import numpy as np

from repoze.lru import lru_cache


def get_delta_add_remove(effective_z_t, q_t, target_eps=1.0, nx=1E6, L=20.0):
    """ Computes the approximation of the exact privacy as per https://arxiv.org/abs/1906.03049
    for a fixed epsilon, for a list of given mechanisms applied. Considers neighbouring sets
    as those with the addition/removal property.

    :param effective_z_t: 1D numpy array of the effective noises applied in the mechanisms.
    :param q_t: 1D numpy array of the selection probabilities of the data used in the mechanisms.
    :param target_eps: The target epsilon to aim for.
    :param nx: The number of discretisation points to use.
    :param L: The range truncation parameter
    :return: (epsilon, delta) privacy bound.
    """

    nx = int(nx)
    half = int(nx / 2)

    tol_newton = 1e-10  # set this to, e.g., 0.01*target_delta

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    fx_table = []
    F_prod = np.ones(x.size)

    ncomp = effective_z_t.size

    if (q_t.size != ncomp):
        print('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):
        # Change from original: cache results for speedup on similar requests
        @lru_cache(maxsize=100)
        def get_fx(sigma, q):
            # first ii for which x(ii)>log(1-q),
            # i.e. start of the integral domain
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

            return fx

        sigma = effective_z_t[ij]
        q = q_t[ij]

        fx = get_fx(sigma, q)

        # Compute the DFT
        FF1 = np.fft.fft(fx * dx)
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
    return target_eps, np.real(delta)


def get_delta_substitution(effective_z_t, q_t, target_eps=1.0, nx=1E6, L=20.0):
    """ Computes the approximation of the exact privacy as per https://arxiv.org/abs/1906.03049
    for a fixed epsilon, for a list of given mechanisms applied. Considers neighbouring sets
    as those with the substitution property.

    :param effective_z_t: 1D numpy array of the effective noises applied in the mechanisms.
    :param q_t: 1D numpy array of the selection probabilities of the data used in the mechanisms.
    :param target_eps: The target epsilon to aim for.
    :param nx: The number of discretisation points to use.
    :param L: The range truncation parameter
    :return: (epsilon, delta) privacy bound.
    """

    nx = int(nx)
    half = int(nx / 2)

    tol_newton = 1e-10  # set this to, e.g., 0.01*target_delta

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    fx_table = []
    F_prod = np.ones(x.size)

    ncomp = effective_z_t.size

    if (q_t.size != ncomp):
        print('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):
        # Change from original: cache results for speedup on similar requests
        @lru_cache(maxsize=100)
        def get_fx(sigma, q):
            # Evaluate the PLD distribution,
            # This is the case of substitution relation (subsection 5.2)
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

            return fx

        sigma = effective_z_t[ij]
        q = q_t[ij]

        fx = get_fx(sigma, q)

        FF1 = np.fft.fft(fx * dx)  # Compute the DFFT
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
    return target_eps, np.real(delta)


def get_eps_add_remove(effective_z_t, q_t, target_delta=1e-6, nx=1E6, L=20.0):
    """ Computes the approximation of the exact privacy as per https://arxiv.org/abs/1906.03049
        for a fixed epsilon, for a list of given mechanisms applied. Considers neighbouring sets
        as those with the addition/removal property.

        :param effective_z_t: 1D numpy array of the effective noises applied in the mechanisms.
        :param q_t: 1D numpy array of the selection probabilities of the data used in the mechanisms.
        :param target_delta: The target delta to aim for.
        :param nx: The number of discretisation points to use.
        :param L: The range truncation parameter
        :return: (epsilon, delta) privacy bound.
    """

    nx = int(nx)
    half = int(nx / 2)

    tol_newton = 1e-10  # set this to, e.g., 0.01*target_delta

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    fx_table = []
    F_prod = np.ones(x.size)

    ncomp = effective_z_t.size

    if (q_t.size != ncomp):
        print('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):
        # Change from original: cache results for speedup on similar requests
        @lru_cache(maxsize=100)
        def get_fx(sigma, q):
            # first ii for which x(ii)>log(1-q),
            # i.e. start of the integral domain
            ii = int(np.floor(float(nx * (L + np.log(1 - q)) / (2 * L))))

            # Evaluate the PLD distribution,
            # The case of remove/add relation (Subsection 5.1)
            Linvx = (sigma ** 2) * np.log((np.exp(x[ii + 1:]) - (1 - q)) / q) + 0.5
            ALinvx = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * ((1 - q) * np.exp(-Linvx * Linvx / (2 * sigma ** 2)) +
                                                              q * np.exp(
                        -(Linvx - 1) * (Linvx - 1) / (2 * sigma ** 2)))
            ey = np.exp(x[ii + 1:])
            dLinvx = (sigma ** 2) / (1 - (1 - q) / ey)

            fx = np.zeros(nx)
            fx[ii + 1:] = np.real(ALinvx * dLinvx)

            # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
            temp = np.copy(fx[half:])
            fx[half:] = np.copy(fx[:half])
            fx[:half] = temp

            return fx

        sigma = effective_z_t[ij]
        q = q_t[ij]

        fx = get_fx(sigma, q)

        # Compute the DFT
        FF1 = np.fft.fft(fx * dx)
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
        print('Error: epsilon out of [-L,L] window, please check the parameters.')
        return float('inf')
    else:
        # print('Unbounded DP-epsilon after ' + str(int(ncomp)) + ' compositions defined by sigma and q arrays: ' + str(np.real(eps_0)) + ' (delta=' + str(target_delta) + ')')
        return np.real(eps_0), target_delta


def get_eps_substitution(effective_z_t, q_t, target_delta=1e-6, nx=1E6, L=20.0):
    nx = int(nx)
    half = int(nx / 2)

    tol_newton = 1e-10  # set this to, e.g., 0.01*target_delta

    dx = 2.0 * L / nx  # discretisation interval \Delta x
    x = np.linspace(-L, L - dx, nx, dtype=np.complex128)  # grid for the numerical integration

    # Initial value \epsilon_0
    eps_0 = 0

    fx_table = []
    F_prod = np.ones(x.size)

    ncomp = effective_z_t.size

    if (q_t.size != ncomp):
        print('The arrays for q and sigma are of different size!')
        return float('inf')

    for ij in range(ncomp):
        # Change from original: cache results for speedup on similar requests
        @lru_cache(maxsize=100)
        def get_fx(sigma, q):
            # Evaluate the PLD distribution,
            # This is the case of substitution relation (subsection 5.2)
            ey = np.exp(x)
            c = q * np.exp(-1 / (2 * sigma ** 2))
            term1 = (-(1 - q) * (1 - ey) + np.sqrt((1 - q) ** 2 * (1 - ey) ** 2 + 4 * c ** 2 * ey)) / (2 * c)
            term1 = np.maximum(term1, 1e-16)
            Linvx = (sigma ** 2) * np.log(term1)

            sq = np.sqrt((1 - q) ** 2 * (1 - ey) ** 2 + 4 * c ** 2 * ey)
            nom1 = 4 * c ** 2 * ey - 2 * (1 - q) ** 2 * ey * (1 - ey)
            term1 = nom1 / (2 * sq)
            nom2 = term1 + (1 - q) * ey
            nom2 = nom2 * (sq + (1 - q) * (1 - ey))
            dLinvx = sigma ** 2 * nom2 / (4 * c ** 2 * ey)

            ALinvx = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * ((1 - q) * np.exp(-Linvx * Linvx / (2 * sigma ** 2))
                                                              + q * np.exp(
                        -(Linvx - 1) * (Linvx - 1) / (2 * sigma ** 2)))

            fx = np.real(ALinvx * dLinvx)

            # Flip fx, i.e. fx <- D(fx), the matrix D = [0 I;I 0]
            temp = np.copy(fx[half:])
            fx[half:] = np.copy(fx[:half])
            fx[:half] = temp

        sigma = effective_z_t[ij]
        q = q_t[ij]

        fx = get_fx(sigma, q)

        FF1 = np.fft.fft(fx * dx)  # Compute the DFFT
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
        print('Error: epsilon out of [-L,L] window, please check the parameters.')
        return float('inf')
    else:
        print('Bounded DP-epsilon after ' + str(int(ncomp)) + ' compositions defined by sigma and q arrays: ' + str(
            np.real(eps_0)) + ' (delta=' + str(target_delta) + ')')
        return np.real(eps_0)
    #
    # print('Bounded DP-epsilon after ' + str(int(ncomp)) + ' compositions:' + str(np.real(eps_0)) + ' (delta=' + str(target_delta) + ')')
    # return np.real(eps_0)


def compute_prvacy_loss_from_ledger(ledger, target_eps=None, target_delta=None, adjacency_definition='add_remove',
                                    nx=1E6, L=20.0):
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
            return get_delta_add_remove(effective_z_t, q_t, target_eps=target_eps, nx=nx, L=L)
        else:
            return get_eps_add_remove(effective_z_t, q_t, target_delta=target_delta, nx=nx, L=L)

    if adjacency_definition is 'add_remove':
        if target_eps is not None:
            return get_delta_substitution(effective_z_t, q_t, target_eps=target_eps, nx=nx, L=L)
        else:
            return get_eps_substitution(effective_z_t, q_t, target_delta=target_delta, nx=nx, L=L)

    raise ValueError('adjacency_definition must be one of "substitution" or "add_remove".')
