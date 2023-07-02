"""Functions devoted to converting between various data types."""
import common
import math
import numpy as np


def ranges_from_depth_image(depth, max_range=200.0):
    """Convert depth image into ranges."""
    dshape = depth.shape
    dslice = depth[dshape[0] // 2, :, :]
    ranges = (1.0 * dslice[:, 0] + dslice[:, 1] / 256.0 +
              dslice[:, 2] / 256.0 / 256.0) / 256.0 * max_range
    return ranges.astype(np.float32)


def depths_from_depth_image(depth_image):
    return (1.0 * depth_image[:, :, 0] + depth_image[:, :, 1] / 256.0 +
            depth_image[:, :, 2] / 256.0 / 256.0) / 256.0 * 200.0


def image_aligned_to_robot(image, r_pose):
    """Permutes an image from axis-aligned to robot frame."""
    cols = image.shape[1]
    roll_amount = int(round(-cols * r_pose.yaw / (2 * math.pi)))
    return np.roll(image, shift=roll_amount, axis=1)


def image_aligned_from_robot_to_global(image, r_pose):
    inv_pose = common.Pose(r_pose.x, r_pose.y, -r_pose.yaw)
    return image_aligned_to_robot(image, inv_pose)


def image_aligned_to_subgoal(image, r_pose, subgoal):
    """Permutes an image from axis-aligned to subgoal-pointing frame.
    The subgoal should appear at the center of the image."""
    cols = image.shape[1]
    sp = subgoal.get_centroid()
    yaw = np.arctan2(sp[1] - r_pose.y, sp[0] - r_pose.x) - r_pose.yaw
    roll_amount = int(round(-cols * yaw / (2 * math.pi)))
    return np.roll(image, shift=roll_amount, axis=1)
