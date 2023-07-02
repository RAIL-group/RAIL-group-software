"""
Includes definitions for various 'Robot' classes which are useful for
storing the agent's state and getting the agent's available actions during
navigation. The primary robot class is the "Turtlebot_Robot", which defines
a basic set of motion primitives relevant to differential drive robots.
"""

from vertexnav_accel import Pose
import copy
import math
import numpy as np
from . import utils


class Motion_Primitive():
    def __init__(self, poses, cost, map_data=None):
        self.poses = poses
        self.cost = cost
        self.map_data = map_data

    def transform(self, pose):
        """Transform the motion primitive by a given pose.
        This is equivalent to 'right multiplying' the pose/transform.
        The resulting primitive should be such that the output poses
        are the input poses applied to the pose.
        """
        new_poses = [p * pose for p in self.poses]
        return Motion_Primitive(poses=new_poses, cost=self.cost)


class Robot:
    """Base robot class.
    All robots must define the 'move' function, which defines how they can
    move through space. The Robot class is also useful for defining how an
    embodied agent travels through space and for storing a history of poses.
    """
    def __init__(self, pose):
        self.pose = copy.copy(pose)
        self.all_poses = [copy.copy(self.pose)]
        self.does_use_motion_primitives = 0
        self.net_motion = 0

    def move(self, plan, distance=1.0):

        # Start at 1 so that the "rounded current robot pose" is not used
        ii = 1
        remaining_dist = distance
        while remaining_dist > 0 and ii < plan.shape[1]:
            subgoal = plan[:, ii]
            dx = subgoal[0] - self.pose.x
            dy = subgoal[1] - self.pose.y
            step_dist = math.sqrt(dx * dx + dy * dy)
            if step_dist < remaining_dist:
                remaining_dist -= step_dist
                self.pose.x = subgoal[0]
                self.pose.y = subgoal[1]
                self.net_motion += step_dist
                ii += 1
            else:
                self.pose.x += remaining_dist * dx / step_dist
                self.pose.y += remaining_dist * dy / step_dist
                self.net_motion += remaining_dist
                self.all_poses += [copy.copy(self.pose)]
                return

    def max_travel_distance(self, num_recent_poses):
        """Returns the farthest distance between two poses out of the last N poses"""
        max_dist = 0.0
        if len(self.all_poses) < num_recent_poses:
            return 1e10
        for pose_a in self.all_poses[-num_recent_poses:]:
            for pose_b in self.all_poses[-num_recent_poses:]:
                max_dist = max(max_dist,
                               Pose.cartesian_distance(pose_a, pose_b))
        return max_dist

    @property
    def pose_m(self):
        if self.map_data is None:
            return self.pose

        return Pose(x=self.pose.x * self.map_data['resolution'] +
                    self.map_data['x_offset'],
                    y=self.pose.y * self.map_data['resolution'] +
                    self.map_data['y_offset'],
                    yaw=self.pose.yaw)


class Motion_Primitive_Robot(Robot):
    """A simple robot that moves using motion primitives.
    This is the base class for robots that use motion primitives to move.
    As such, the 'get_motion_primitives' function must be defined and the
    move class has been updated to include the selected primitive for motion.
    """
    def __init__(self,
                 pose,
                 num_primitives=32,
                 primitive_length=2.8,
                 map_data=None,
                 unity_bridge=None):
        self.pose = copy.copy(pose)
        self.all_poses = [copy.copy(self.pose)]
        self.net_motion = 0
        self.does_use_motion_primitives = 1
        self.num_primitives = num_primitives
        self.primitive_length = primitive_length
        self.map_data = map_data
        self.unity_bridge = unity_bridge
        self.has_moved = False
        self.still_count = 0

    def _get_local_points(self, range_thresh):
        def _transform_rays(rays, sensor_pose):
            """Transform (rotate and offset) a laser scan according to pose."""
            origin = np.array([[sensor_pose.x], [sensor_pose.y]])
            rotation_mat = np.array(
                [[math.cos(sensor_pose.yaw), -math.sin(sensor_pose.yaw)],
                 [math.sin(sensor_pose.yaw),
                  math.cos(sensor_pose.yaw)]])

            return np.matmul(rotation_mat, rays) + origin

        # Get the depth image & raw ranges
        if self.unity_bridge is None:
            return None

        self.unity_bridge.move_object_to_pose("robot", self.pose)
        pano_depth_image = self.unity_bridge.get_image(
            "robot/pano_depth_camera")
        pano_depth_image = utils.convert.depths_from_depth_image(
            pano_depth_image) / self.unity_bridge.sim_scale

        # Get the close points in the local frame
        ranges = pano_depth_image[pano_depth_image.shape[0] // 2]
        is_close = ranges <= range_thresh
        if is_close.any():
            directions, _ = utils.calc.directions_vec(ranges.size)
            close_points = (ranges * directions)[:, is_close]

            # Transform to global frame and return
            no_yaw_pose = self.pose
            no_yaw_pose.yaw = 0
            return _transform_rays(close_points, self.pose)
        else:
            return None

    def get_motion_primitives(self):
        """Returns the motion primitives available to the robot at the current
        time (can be a function of robot state)."""
        # Create motion primitives
        return [
            Motion_Primitive(poses=[
                Pose(x=self.primitive_length *
                     math.cos(2 * math.pi * i / self.num_primitives),
                     y=self.primitive_length *
                     math.sin(2 * math.pi * i / self.num_primitives),
                     yaw=0) * self.pose
            ], cost=self.primitive_length)
            for i in range(self.num_primitives)
        ]

    def move(self, goal, path, inflation_rad):
        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        is_still = False
        if dist(path[-1], (self.pose.x, self.pose.y)) < 0.001:
            print("is still")
            is_still = True
        else:
            self.has_moved = True

        if is_still and not self.has_moved:
            return

        primitives = self.get_motion_primitives()

        # Get distances
        max_range = 6.0 * inflation_rad + 2 * self.primitive_length
        obs_points = self._get_local_points(range_thresh=max_range)

        def get_obs_dist(primitive):
            if obs_points is None:
                return 100000
            else:
                pose = primitive.poses[-1]
                dists = np.linalg.norm(obs_points - [[pose.x], [pose.y]],
                                       axis=0)
                assert 2 * dists.size == obs_points.size
                return min(dists)

        # Cost = distance + WEIGHT * linear
        # costs = [
        #     dist((prim.poses[-1].x, prim.poses[-1].y), path[1]) + 100.0 * max(
        #         inflation_rad - proposed_world.get_dist(prim.poses[-1]), 0)
        #     for prim in primitives
        # ]
        costs = [
            dist((prim.poses[-1].x, prim.poses[-1].y), path[1])
            # - 0.8*dist((prim.poses[-1].x, prim.poses[-1].y), path[-1])
            + 10000.0 * max(1.0 - get_obs_dist(prim) / inflation_rad, 0) +
            prim.cost * max(8.0 -
                            (get_obs_dist(prim) / inflation_rad - 1)**2, 0) /
            6.0  # Soft cost
            + prim.cost for prim in primitives
        ]

        if is_still:
            costs = [
                self.primitive_length *
                max(12.0 - (get_obs_dist(prim) / inflation_rad - 1.0), 0) /
                3.0  # Soft cost
                + prim.cost for prim in primitives
            ]
            self.still_count += 1
        else:
            self.still_count = max(0, self.still_count - 2)

        # if len(path) > 1:
        #     second_costs = [
        #         dist((prim.poses[-1].x, prim.poses[-1].y), path[2])
        #         + 100.0 * max(inflation_rad - get_obs_dist(prim), 0)
        #         + 0.5 * max(2 * inflation_rad - get_obs_dist(prim), 0)  # Soft cost
        #         for prim in primitives]
        #     costs = [min(c, sc) for c, sc in zip(costs, second_costs)]

        ind = costs.index(min(costs))
        primitive = primitives[ind]
        self.net_motion += primitive.cost
        self.pose = Pose(primitive.poses[-1].x, primitive.poses[-1].y,
                         primitive.poses[-1].yaw)
        for pose in primitive.poses:
            p = Pose(pose.x, pose.y, pose.yaw)
            self.all_poses.append(p)


class Turtlebot_Robot(Motion_Primitive_Robot):
    """A simple 'turtlebot' robot class."""
    def __init__(self,
                 pose,
                 num_primitives=20,
                 max_yaw=math.pi / 2,
                 primitive_length=1.0,
                 map_data=None,
                 unity_bridge=None):
        self.pose = Pose(pose.x, pose.y, pose.yaw)
        self.all_poses = [self.pose]
        self.net_motion = 0
        self.does_use_motion_primitives = 1
        self.num_primitives = num_primitives
        self.max_yaw = max_yaw
        self.primitive_length = primitive_length
        self.map_data = map_data
        self.unity_bridge = unity_bridge
        self.still_count = 0
        self.has_moved = False

    def get_motion_primitives(self):
        """Returns the motion primitives available to the robot at the current
        time (can be a function of robot state)."""
        r0 = self.primitive_length
        N = self.num_primitives
        angles = [(i * 1.0 / N - 1) * self.max_yaw / 2
                  for i in range(2 * N + 1)]
        primitive_list = [
            Motion_Primitive(poses=[
                Pose(x=r0 * math.cos(angle) * (math.cos(angle)),
                     y=r0 * math.sin(angle) * (math.cos(angle)),
                     yaw=2 * angle) * self.pose
            ], cost=r0) for angle in angles
        ]
        primitive_list += [
            Motion_Primitive(poses=[Pose(x=0, y=0, yaw=math.pi) * self.pose],
                             cost=2 * r0)
        ]
        return primitive_list
