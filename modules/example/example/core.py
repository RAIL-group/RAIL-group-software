import numpy as np


def get_random_matrix(mat_shape):
    """Return a matrix of random numbers of specified shape."""
    return np.random.rand(*mat_shape)
