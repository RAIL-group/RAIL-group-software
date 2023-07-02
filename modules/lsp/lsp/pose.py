"""Defines the Pose class used throughout lsp."""
import math
import numpy as np
from common import Pose  # noqa


# A utility function
def compute_path_length(path):
    """Compute the length of a path."""
    length = 0

    # Optionally convert a list of poses to a numpy array
    if type(path) is list:
        poses = path
        path = np.zeros([2, len(poses)])
        for ii, pose in enumerate(poses):
            path[0, ii] = pose.x
            path[1, ii] = pose.y

    # Compute the path length
    for ii in range(1, path.shape[1]):
        length += math.sqrt(
            math.pow(path[0, ii - 1] - path[0, ii], 2) +
            math.pow(path[1, ii - 1] - path[1, ii], 2))

    return length
