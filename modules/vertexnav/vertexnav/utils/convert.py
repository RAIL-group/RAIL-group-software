import math
import numpy as np
import shapely

import vertexnav
"""Functions devoted to converting between (mostly) the various data types
introduced in the vertexnav package."""


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


def image_aligned_to_subgoal(image, r_pose, subgoal):
    """Permutes an image from axis-aligned to subgoal-pointing frame.
    The subgoal should appear at the center of the image."""
    cols = image.shape[1]
    sp = subgoal.get_centroid()
    yaw = np.arctan2(sp[1] - r_pose.y, sp[0] - r_pose.x) - r_pose.yaw
    roll_amount = int(round(-cols * yaw / (2 * math.pi)))
    return np.roll(image, shift=roll_amount, axis=1)


def get_corner_vector_from_obs(observation, size, pose, dtype='gap'):
    """Get the flattened vector of gaps. The size of the vector typically
    corresponds to the number of columns in the corresponding image."""
    is_gap_vector = np.zeros([size], dtype=int)
    gap_dist = np.zeros([size], dtype=int)
    directions, angles_rad = vertexnav.utils.calc.directions_vec(size)
    point = np.array([pose.x, pose.y])

    if not dtype == 'gap' and not dtype == 'convex':
        raise ValueError('Corner type "{}" unsupported.'.format(dtype))

    def dist(a, b):
        d = b - a
        return math.sqrt(d[0]**2 + d[1]**2)

    for corner in observation:
        ind = np.argmax(np.cos(angles_rad - corner.angle_rad))
        is_gap_vector[ind] = 1
        gap_dist[ind] = dist(corner.position, point)

    return is_gap_vector, gap_dist


def get_vertex_grid_data_from_obs(observation, size, pose, max_range,
                                  num_range, num_bearing):
    """Get the flattened vector of gaps. The size of the vector typically
    corresponds to the number of columns in the corresponding image.

    'observation' is assumed to be a vector of PerfectVertexDetection objects
    and ordered according to their 'angle_rad'.
    """

    # Initialize some vectors
    is_vertex_vector = np.zeros([num_range, num_bearing], dtype=int)
    is_left_gap_vector = np.zeros([num_range, num_bearing], dtype=int)
    is_right_gap_vector = np.zeros([num_range, num_bearing], dtype=int)
    is_corner_vector = np.zeros([num_range, num_bearing], dtype=int)
    is_point_vertex_vector = np.zeros([num_range, num_bearing], dtype=int)
    is_in_view = np.zeros([num_range, num_bearing], dtype=int)

    # Lookup vectors
    vec_range, vec_bearing = vertexnav.utils.calc.range_bearing_vecs(
        max_range, num_range, num_bearing)

    point = np.array([pose.x, pose.y])

    def dist(a, b):
        d = b - a
        return math.sqrt(d[0]**2 + d[1]**2)

    for vertex in observation:
        # For all gap types, add the detection to the 'is_vertex' vector and
        # compute distance.
        vertex_range = dist(vertex.position, point)
        ind_bearing = np.argmax(np.cos(vec_bearing - vertex.angle_rad))
        ind_range = np.argmin(abs(vec_range - vertex_range))

        is_vertex_vector[ind_range, ind_bearing] = 1

        label = vertex.detection_type.label
        is_right_gap_vector[ind_range, ind_bearing] = label[0]
        is_corner_vector[ind_range, ind_bearing] = label[1]
        is_left_gap_vector[ind_range, ind_bearing] = label[2]
        is_point_vertex_vector[ind_range, ind_bearing] = label[3]

    # Also generate the observed region:
    dr = vec_range[1] - vec_range[0]
    vis_poly = shapely.geometry.Polygon(
        vertexnav.noisy.compute_conservative_space_from_obs(pose,
                                                            observation,
                                                            radius=max_range))
    vis_poly = vis_poly.buffer(1.5 * dr)
    for ir, vr in np.ndenumerate(vec_range):
        for ib, vb in np.ndenumerate(vec_bearing):
            x = vr * math.cos(vb) + pose.x
            y = vr * math.sin(vb) + pose.y
            if vis_poly.contains(shapely.geometry.Point(x, y)):
                is_in_view[ir, ib] = 1

    # assert((is_vertex_vector == (
    #     is_left_gap_vector + is_right_gap_vector +
    #     is_corner_vector + is_point_vertex_vector)).all())

    return {
        'is_vertex': is_vertex_vector,
        'is_left_gap': is_left_gap_vector,
        'is_right_gap': is_right_gap_vector,
        'is_corner': is_corner_vector,
        'is_point_vertex': is_point_vertex_vector,
        'is_in_view': is_in_view,
    }
