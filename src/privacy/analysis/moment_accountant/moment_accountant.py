import numpy as np

from .log_moment_utils import generate_log_moments


def _compute_eps(max_lambda, total_log_moments, target_delta):
    """ Compute the epsilon upper bound for a fixed delta

    :param max_lambda: The maximum moment considered
    :param total_log_moments: The total of each moment accumulated
    :param target_delta: The target delta
    :return: epsilon bound, delta bound, order of moment responsible for the bound
    """
    eps_bounds = []
    for lambda_i in range(1, max_lambda + 1):
        lambda_i_eps_bound = (1.0 / lambda_i) * (total_log_moments[lambda_i - 1] - np.log(target_delta))
        eps_bounds.append(lambda_i_eps_bound)

    max_index = np.argmin(eps_bounds)
    return eps_bounds[max_index], target_delta, max_index + 1


def _compute_delta(max_lambda, total_log_moments, target_eps):
    """ Compute the delta upper bound for a fixed epsilon

    :param max_lambda: The maximum moment considered
    :param total_log_moments: The total of each moment accumulated
    :param target_eps: The target epsilon
    :return: epsilon bound, delta bound, order of moment responsible for the bound
    """
    delta_bounds = []
    for lambda_i in range(1, max_lambda + 1):
        lambda_i_eps_bound = np.exp(total_log_moments[lambda_i - 1] - lambda_i * target_eps)
        delta_bounds.append(lambda_i_eps_bound)

    max_index = np.argmin(delta_bounds)

    return target_eps, delta_bounds[max_index], max_index + 1


def get_privacy_spent(max_lambda, total_log_moments, target_eps, target_delta):
    """ Compute the (epsilon, delta) privacy bound for a given set of log
     moments with one of epsilon or delta fixed.

    :param max_lambda: The maximum moment to consider
    :param total_log_moments: The total moments, up to the maximum moment
    :param target_eps: A target epsilon to aim for.
    :param target_delta: A target delta to aim for.
    :return: epsilon, delta
    """
    if target_delta is None and target_eps is None:
        raise ValueError(
            "One of the target values must not be None")

    if target_eps is not None and target_delta is not None:
        raise ValueError(
            "Exactly one out of eps and delta must be None. (None is).")

    if target_eps is not None:
        epsilon, delta, lambda_i = _compute_delta(max_lambda, total_log_moments, target_eps)
        return epsilon, delta
    else:
        epsilon, delta, lambda_i = _compute_eps(max_lambda, total_log_moments, target_delta)
        return epsilon, delta


def compute_log_moments_from_ledger(ledger, max_lambda=32):
    """ Compute the moments of the queries entered into a ledger
    using the Gaussian Mechanism utilising the Moments Accountant.

    :param ledger: The ledger of queries to compute for
    :param max_lambda: The maximum moment to compute
    :return: The upper bounds for all moments up to max_lambda
    """
    total_log_moments = np.zeros(max_lambda, dtype=float)
    for sample in ledger:
        # note this specific effective z calculation allows for different scale factors to be applied!
        effective_z = round(sum([
            (q.noise_stddev / q.l2_norm_bound) ** -2 for q in sample.queries
        ]) ** -0.5, 5)
        total_log_moments += generate_log_moments(sample.selection_probability, effective_z, max_lambda)

    return total_log_moments


def compute_privacy_loss_from_ledger(ledger, target_delta=None, target_eps=None, max_lambda=32):
    """ Compute the privacy loss of the queries entered into a ledger
    using the Gaussian Mechanism utilising the Moments Accountant with
    one of epsilon or delta fixed.

    :param ledger: The ledger of queries to compute for
    :param max_lambda: The maximum moment to compute
    :param target_eps: A target epsilon to aim for.
    :param target_delta: A target delta to aim for.
    :return: epsilon, delta
    """

    total_log_moments = compute_log_moments_from_ledger(ledger, max_lambda)
    return get_privacy_spent(max_lambda, total_log_moments, target_eps, target_delta)


def compute_online_privacy_from_ledger(ledger, log_moments, target_delta=None, target_eps=None, max_lambda=32):
    """ Compute the privacy loss of the queries entered into a ledger
        using the Gaussian Mechanism utilising the Moments Accountant with
        one of epsilon or delta fixed and composes this with the privacy of
        previously seen entries, summarised by the log moments.

        :param ledger: The ledger of queries to compute for. An incremental
        ledger, NOT the whole ledger.
        :param log_moments: summary log moments of previous queries
        :param max_lambda: The maximum moment to compute
        :param target_eps: A target epsilon to aim for.
        :param target_delta: A target delta to aim for.
        :return: epsilon, delta
        """
    if not ledger and target_delta is None:
        return (target_eps, 0), None
    elif not ledger and target_eps is None:
        return (0, target_delta), None
    elif log_moments is None:
        log_moments = compute_log_moments_from_ledger(ledger, max_lambda=max_lambda)
    else:
        log_moments = log_moments + compute_log_moments_from_ledger(ledger, max_lambda=max_lambda)

    privacy_bound = get_privacy_spent(max_lambda, log_moments, target_eps=target_eps, target_delta=target_delta)

    return privacy_bound, log_moments
