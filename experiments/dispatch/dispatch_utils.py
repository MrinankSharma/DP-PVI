import yaml
import itertools

def nested_dict_to_option_strings(d, base_key=None):
    ret = []
    base_key_to_use = "" if base_key is None else f"{base_key}."
    for k, v in d.items():
        # print(f"{k} has value {v}")
        if isinstance(v, list):
            list_values = []
            for l in v:
                list_values.append((base_key_to_use + k, l))
            ret.append(list_values)
        elif isinstance(v, dict):
            ret.extend(nested_dict_to_option_strings(v, base_key_to_use + k))
        else:
            ret.append([(base_key_to_use + k, v)])
    return ret

