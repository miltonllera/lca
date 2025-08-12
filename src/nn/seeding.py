import numpy as np
import jax.random as jr


def init_central_seed(shape, key=None):
    _, H, W = shape
    init = np.zeros(shape)
    init[3:, H//2, W//2] = 1.0
    return init


def init_random(shape, key):
    return jr.uniform(key, shape, minval=-1, maxval=1)

