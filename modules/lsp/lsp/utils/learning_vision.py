import math
import common
import environments
import numpy as np


def get_angle_rad(position, obs_pose):
    return (math.atan2(position[1] - obs_pose.y, position[0] - obs_pose.x) -
            obs_pose.yaw) % (2 * math.pi)


def dist(a, b):
    d = b - a
    return math.sqrt(d[0]**2 + d[1]**2)


def get_directions(num_im_cols):
    """Returns an array of 'direction vectors' for a panoramic image
    from Unity"""

    angles_rad = np.linspace(-math.pi, math.pi, num_im_cols + 1)[:-1]
    directions = np.vstack((np.cos(angles_rad), np.sin(angles_rad)))
    return (directions, angles_rad)


def get_range_bearing_vecs(max_range, num_range, num_bearing):
    vec_range = np.linspace(start=0.0, stop=max_range, num=num_range + 1)
    vec_range = vec_range[1:]
    _, vec_bearing = get_directions(num_bearing)
    return vec_range, vec_bearing


def get_rel_goal_loc_vecs(pose, goal_pose, num_bearing, subgoal=None):
    # Lookup vectors
    _, vec_bearing = get_directions(num_bearing)
    if subgoal is None:
        vec_bearing = vec_bearing + pose.yaw
    else:
        sp = subgoal.get_centroid()
        subgoal_yaw = np.arctan2(sp[1] - pose.y, sp[0] - pose.x)
        vec_bearing = vec_bearing + subgoal_yaw

    robot_point = np.array([pose.x, pose.y])
    goal_point = np.array([goal_pose.x, goal_pose.y])
    rel_goal_point = goal_point - robot_point

    goal_loc_x_vec = rel_goal_point[0] * np.cos(
        vec_bearing) + rel_goal_point[1] * np.sin(vec_bearing)
    goal_loc_y_vec = -rel_goal_point[0] * np.sin(
        vec_bearing) + rel_goal_point[1] * np.cos(vec_bearing)

    return (goal_loc_x_vec[:, np.newaxis].T, goal_loc_y_vec[:, np.newaxis].T)


def get_range_bearing_indices(obs_pose, lookup_point, vec_range, vec_bearing):
    """Helper function for computing lookup indices for setting/getting values
    in a range-bearing grid. Also returns a bool 'is_inside' that is True if
    the lookup point is inside the grid and False if the lookup point is
    outside the grid (and projected inside)."""
    p_angle_rad = get_angle_rad(lookup_point, obs_pose)
    p_range = dist(lookup_point, np.array([obs_pose.x, obs_pose.y]))
    ind_bearing = np.argmax(np.cos(vec_bearing - p_angle_rad))
    ind_range = np.argmin(abs(vec_range - p_range))
    is_inside = p_range <= (vec_range[-1] + vec_range[1] - vec_range[0])

    return is_inside, ind_range, ind_bearing


def get_oriented_input_data(pano_image, robot_pose, goal_pose, subgoal):
    """Helper function that returns a dictionary of the input data provided to the
neural network in the 'oriented' data configuration. The 'pano_image' is assumed
to be in the robot coordinate frame, and will be 'rotated' such that the subgoal
of interest is at the center of the image. Similarly, the goal information will
be stored as two vectors with each element corresponding to the sin and cos of
the relative position of the goal in the 'oriented' image frame."""

    # Re-orient the image based on the subgoal centroid
    oriented_pano_image = environments.utils.convert.image_aligned_to_subgoal(
        pano_image, robot_pose, subgoal)

    # Compute the goal information
    num_bearing = pano_image.shape[1] // 4
    goal_loc_x_vec, goal_loc_y_vec = get_rel_goal_loc_vecs(
        robot_pose, goal_pose, num_bearing, subgoal)

    sc = subgoal.get_centroid()
    subgoal_pose = common.Pose(sc[0], sc[1], 0)
    subgoal_loc_x_vec, subgoal_loc_y_vec = get_rel_goal_loc_vecs(
        robot_pose, subgoal_pose, num_bearing, subgoal)

    return {
        'image': oriented_pano_image,
        'goal_loc_x': goal_loc_x_vec,
        'goal_loc_y': goal_loc_y_vec,
        'subgoal_loc_x': subgoal_loc_x_vec,
        'subgoal_loc_y': subgoal_loc_y_vec,
    }


def get_oriented_data_from_obs(updated_frontiers, pose, goal, image):
    """Get a list of dict objects, each containing the data needed to train a
learning algorithm to estimate subgoal propperties. The data should be saved
such that the images associated with each frontier are oriented towards that
particular frontier."""

    # Initialize some vectors
    data = []
    for s in updated_frontiers:
        datum = get_oriented_input_data(image, pose, goal, s)
        datum['is_feasible'] = s.prob_feasible
        datum['delta_success_cost'] = s.delta_success_cost
        datum['exploration_cost'] = s.exploration_cost
        datum['positive_weighting'] = s.positive_weighting
        datum['negative_weighting'] = s.negative_weighting
        data.append(datum)

    return data
