import collections

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
    return ret
