import numpy as np
import math


def get_frontier_data_vector(ranges,
                             robot_pose,
                             goal_pose,
                             frontier):
    """Get the vector provided to the learner (laser scan) for a chosen frontier."""
    # Store the goal pose (in the frame of the robot)
    grx, gry = transform_to_robot_frame(robot_pose,
                                        goal_pose.x, goal_pose.y)

    # Store the frontier centroid point (in the frame of the robot)
    centroid = frontier.get_centroid()
    frx, fry = transform_to_robot_frame(robot_pose,
                                        centroid[0], centroid[1])

    return {
        'laser_scan': ranges,
        'goal_rel_pos': np.array([grx, gry]),
        'frontier_rel_pos': np.array([frx, fry]),
    }


def transform_to_robot_frame(robot_pose, x, y):
    """Transforms an x, y pair to the frame of the robot."""
    cx = x - robot_pose.x
    cy = y - robot_pose.y

    rotation_mat = np.array(
        [[math.cos(robot_pose.yaw), -math.sin(robot_pose.yaw)],
            [math.sin(robot_pose.yaw),
             math.cos(robot_pose.yaw)]])

    out = np.matmul(rotation_mat.T, np.array([cx, cy]))

    return out[0], out[1]
