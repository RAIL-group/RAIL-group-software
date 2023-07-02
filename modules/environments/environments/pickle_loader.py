import math
import numpy as np
import pickle
import random
import scipy

from . import base_generator
from common import Pose
from gridmap.constants import COLLISION_VAL


def _downsample_grid(hr_grid, downsample_factor):
    grid = scipy.ndimage.maximum_filter(hr_grid, size=downsample_factor)
    grid = grid[::downsample_factor, ::downsample_factor]
    return grid


def _inflate_grid_label(grid, inflation_radius, label_val):
    """This removes any information about uncertainty; it thresholds the grid
    before occupancy_grid
    """
    flattened_grid = np.zeros(grid.shape)
    flattened_grid[grid == label_val] = 1

    kernel_size = int(1 + 2 * math.ceil(inflation_radius))
    cind = int(math.ceil(inflation_radius))
    y, x = np.ogrid[-cind:kernel_size - cind, -cind:kernel_size - cind]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[y * y + x * x <= inflation_radius * inflation_radius] = 1

    inflated_mask = scipy.ndimage.filters.convolve(flattened_grid,
                                                   kernel,
                                                   mode='constant',
                                                   cval=0)
    return inflated_mask


class MapGenPLoader(base_generator.MapGenBase):
    _map_data = None

    def gen_map(self, random_seed=None):
        self.args.planning_downsample_factor = 1
        cirriculum_fraction = self.args.cirriculum_fraction

        # Initialize the random generators
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Simple processing of the loaded grid
        if isinstance(self.args.map_file, str):
            map_file = self.args.map_file
        elif isinstance(self.args.map_file, list):
            if cirriculum_fraction is None:
                map_file = random.choice(self.args.map_file)
            else:
                assert cirriculum_fraction > 0.0
                assert cirriculum_fraction <= 1.0

                map_areas = []
                for mf in self.args.map_file:
                    with open(mf, 'rb') as pfile:
                        map_data = pickle.load(pfile)
                        map_areas.append((mf, map_data['occ_grid'].size))

                sorted_map_files = [
                    md[0] for md in sorted(map_areas, key=lambda x: x[1])
                ]
                num_maps = math.ceil(cirriculum_fraction *
                                     (len(sorted_map_files) - 1))
                map_file = random.choice(sorted_map_files[:num_maps])
        else:
            raise TypeError("arg 'map_file' must be string or list.")

        with open(map_file, 'rb') as pfile:
            self._map_data = pickle.load(pfile)

        self.hr_grid = self._map_data['occ_grid'].copy()
        self.grid = _downsample_grid(self.hr_grid,
                                     self.args.planning_downsample_factor)
        return self.hr_grid, self.grid, self._map_data

    def get_start_goal_poses(self, min_separation=1, max_separation=1e10):
        """Loop through the points in the grid and get a pair of poses, subject to a certain condition.
        Returns:
        did_succeed (Bool)
        robot_pose (NamedTuple('PoseT', 'x y yaw'))
        goal_pose (NamedTuple('PoseT', 'x y yaw'))
"""
        # If the inflated grid has not been generated, generate it
        if self._inflated_mask is None:
            planning_resolution = self.args.base_resolution * self.args.planning_downsample_factor
            inflation_radius = self.args.inflation_radius_m / planning_resolution
            self._inflated_mask = _inflate_grid_label(
                self.grid,
                inflation_radius=inflation_radius,
                label_val=COLLISION_VAL) < 1

        allowed_indices = np.where(self._inflated_mask)
        allowed_indices_start = allowed_indices
        allowed_indices_goal = allowed_indices

        counter = 0
        while True and counter < 2000:
            idx_start = random.randint(0, allowed_indices_start[0].size - 1)
            start = Pose(x=allowed_indices_start[0][idx_start],
                         y=allowed_indices_start[1][idx_start],
                         yaw=random.uniform(0, 2 * math.pi))

            idx_goal = random.randint(0, allowed_indices_goal[0].size - 1)
            goal = Pose(x=allowed_indices_goal[0][idx_goal],
                        y=allowed_indices_goal[1][idx_goal],
                        yaw=random.uniform(0, 2 * math.pi))

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
