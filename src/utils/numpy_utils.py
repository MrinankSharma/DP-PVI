import numpy as np

inf = np.inf


def add_parameters(*params):
    '''
    Assuming a list of dicts of parameter tensors, add each corresponding element from each together
    :param params: list of dicts with the same keys of tensors
    :return: a dict containing the addition of these parameters
    '''
    return_dict = {}

    for key in params[0].keys():
        value = params[0][key]
        for i in range(len(params) - 1):
            value = value + params[i + 1][key]
        return_dict[key] = np.array(value, dtype=np.float64)

    return return_dict


def subtract_params(*params):
    '''
    Assuming a list of dicts of parameter tensors, subtract each corresponding element from the first element
    :param params: list of dicts with the same keys of tensors
    :return: a dict containing the addition of these parameters
    '''

    return_dict = {}

    for key in params[0].keys():
        value = params[0][key]
        for i in range(len(params) - 1):
            value = value - params[i + 1][key]
        return_dict[key] = np.array(value, dtype=np.float64)

    return return_dict


def clip(parameter, bound):
    return np.clip(parameter, -bound, bound)


def gaussian_noise(shape, std, mean=0):
    return np.random.normal(mean, std, shape)


def to_pure_python(val):
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.tolist()[0]
        else:
            return val.tolist()
    elif np.issubdtype(val, np.float32) or np.issubdtype(val, np.int) or np.issubdtype(val, np.float64):
        return val.tolist()

    return val
