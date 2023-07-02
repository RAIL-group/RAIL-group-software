import math
import numpy as np
import random

from common import Pose
from gridmap.utils import inflate_grid
from gridmap.constants import COLLISION_VAL


class MapGenBase(object):
    def __init__(self, args):
        self.grid = None
        self.hr_grid = None
        self._inflated_mask = None
        self.resolution_m = 1.0
        self.pose_gen_counter = 0
        self.args = args
        self.map_counter = 0
        self.descriptor = ''

    def gen_map(self, random_seed=None):
        pass

    def get_start_goal_poses(self, min_separation=1, max_separation=1e10):
        """Loop through the points in the grid and get a pair of poses, subject to a certain condition.
        Returns:
        did_succeed (Bool)
        robot_pose (Pose)
        goal_pose (Pose)"""
        # If the inflated grid has not been generated, generate it
        if self._inflated_mask is None:
            planning_resolution = self.args.base_resolution * self.args.planning_downsample_factor
            inflation_radius = self.args.inflation_radius_m / planning_resolution
            self._inflated_mask = inflate_grid(
                self.grid,
                inflation_radius=inflation_radius,
                collision_val=COLLISION_VAL) < 1

        # Now sample a random point
        allowed_indices = np.where(self._inflated_mask)

        counter = 0
        while True and counter < 2000:
            idx_start = random.randint(0, allowed_indices[0].size - 1)
            idx_goal = random.randint(0, allowed_indices[0].size - 1)
            start = Pose(x=allowed_indices[0][idx_start],
                         y=allowed_indices[1][idx_start],
                         yaw=0)

            goal = Pose(x=allowed_indices[0][idx_goal],
                        y=allowed_indices[1][idx_goal],
                        yaw=0)

            # Confirm that the poses are in 'allowed' cells
            if not self._inflated_mask[
                    start.x, start.y] or not self._inflated_mask[goal.x,
                                                                 goal.y]:
                continue

            dist = math.sqrt(
                math.pow(start.x - goal.x, 2) + math.pow(start.y - goal.y, 2))
            if dist >= min_separation and dist <= max_separation:
                return (True, start, goal)

            counter += 1

        return (False, None, None)
