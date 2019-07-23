import itertools


def map_structure(operation, *param_groups):
    """ Expects a pair of list of lists of identical structure representing
    two sets of parameter states. This will apply the desired operation to
    each element in the two groups
    :param param_groups_1: The first parameter group
    :param param_groups_2: The second parameter group
    :return: A list of lists which is the result of the operation applied to
    each pair of elements in the inputs
    """

    param_groups_return = []

    for groups in zip(*param_groups):
        group_grads = []
        for ps in zip(*groups):
            group_grads.append(operation(*ps))
        param_groups_return.append(group_grads)

    return param_groups_return


def reduce_structure(reduce, accumulate, param_groups):
    """ Applies an operation to the elements of a parameter group and accumulates
    the result via specificed functions
    :param reduce: The reducing operation
    :param accumulate: The accumulating operation
    :param param_groups: The
    :return: The accumulated result
    """

    result = None

    for group in param_groups:
        for p in group:
            if result:
                result = accumulate(reduce(p), result)
            else:
                result = reduce(p)

    return result


def flatten(param_groups):
    """ Flattens out a pair of list of lists representing a param_group into
    a single list
    :param param_groups: The group to flatten
    :return: The flattened list
    """
    return itertools.chain(*param_groups)


def parameters_to_tensor_groups(parameter_groups, attribute):
    """ Function to reduce a parameter group from an optimizer to tensor_group with
    the same structure and just the requested parameter
    :param parameter_groups: The parameter group to extract from
    :param attribute: The attribute from the parameter to extrect, e.g. data or grad
    :return: a tensor_group of the same shape ocnting the requested attribute
    """

    tensor_group_return = []

    for group in parameter_groups:
        group_attrs = []
        for ps in group["params"]:
            group_attrs.append(ps.__getattribute__(attribute))
            tensor_group_return.append(group_attrs)

    return tensor_group_return