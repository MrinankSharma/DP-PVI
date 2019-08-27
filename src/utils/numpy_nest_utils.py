import collections
import numbers

import numpy as np


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def structured_lists_to_lists(l):
    ret = []
    for x in l:
        if isinstance(x, list):
            ret.append(structured_lists_to_lists(x))
        elif isinstance(x, dict):
            ret.append(structured_ndarrays_to_lists(x))
        elif isinstance(x, np.ndarray):
            if x.size == 1:
                ret.append(x.tolist()[0])
            else:
                ret.append(x.tolist())
        elif isinstance(x, numbers.Number):
            ret.append(x)
    return ret


def structured_ndarrays_to_lists(d):
    ret = {}
    for k, v in d.items():
        if isinstance(v, collections.MutableMapping):
            ret[k] = structured_ndarrays_to_lists(v)
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                l = v.tolist()
                ret[k] = l[0]
            else:
                ret[k] = v.tolist()
        elif isinstance(v, list):
            # bit of a hack - assume that the list is fine!
            ret[k] = structured_lists_to_lists(v)

    return ret


def map_structure(op, *param_dicts):
    ret = {}
    for k in param_dicts[0].keys():
        ret[k] = op(*[param_dicts[i][k] for i in range(len(param_dicts))])

    return ret


def apply_to_structure(op, param_dict):
    ret = {}
    for k in param_dict.keys():
        ret[k] = op(param_dict[k])

    return ret


def reduce_structure(reduce, accumulate, *param_dicts):
    result = None

    for d in param_dicts:
        for key, value in d.items():
            if result:
                result = accumulate(reduce(value), result)
            else:
                result = reduce(value)

    return result
