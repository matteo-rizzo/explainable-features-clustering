def intersect_dicts(dict_a: dict, dict_b: dict, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using dict_a values
    return {k: v for k, v in dict_a.items() if k in dict_b
            and not any(x in k for x in exclude) and v.shape == dict_b[k].shape}
