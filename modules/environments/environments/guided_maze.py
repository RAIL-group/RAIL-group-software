import math
import numpy as np
import random
import skimage.graph

from common import Pose
from . import base_generator

DEFAULT_PARAMETERS = {
    'base_resolution': 0.3,
    'inflation_radius_m': 0.75,
    'laser_max_range_m': 18.0,
    'num_breadcrumb_elements': 2000,
}


def gen_map_maze_base(width_cells=7,
                      height_cells=10,
                      path_width=10,
                      wide_path_width=18,
                      all_wide=False,
                      random_seed=None):

    # Some parameters
    pw = path_width
    ww = path_width + 2
    PW = wide_path_width
    dPW = (PW - pw) // 2  # Wide path width difference

    assert (PW < ww + pw)

    # Initialize the random generator
    random.seed(random_seed)
    np.random.seed(random_seed)

    # store parameters
    w, h = width_cells, height_cells

    # Initialize the wall and cell objects
    v_walls = [(x, y, x + 1, y) for x in range(w - 1) for y in range(h)]
    h_walls = [(x, y, x, y + 1) for x in range(w) for y in range(h - 1)]
    walls = v_walls + h_walls
    cells = [set([(x, y)]) for x in range(w) for y in range(h)]

    # Init the grid
    grid = np.zeros([w * 2 - 1, h * 2 - 1])
    grid[::2, ::2] = 1

    def get_grid_cell(cx, cy):
        return np.array([cx * 2, cy * 2])

    def get_wall_cell(w):
        return (get_grid_cell(w[0], w[1]) + get_grid_cell(w[2], w[3])) // 2

    random.shuffle(walls)

    # Build the maze
    for wall in walls:
        set_a = None
        set_b = None

        for s in cells:
            if (wall[0], wall[1]) in s:
                set_a = s
            if (wall[2], wall[3]) in s:
                set_b = s

        if set_a is not set_b:
            cells.remove(set_a)
            cells.remove(set_b)
            cells.append(set_a.union(set_b))
            c = get_wall_cell(wall)
            grid[c[0], c[1]] = 1

    # Turn this into a numpy array
    path_grid = grid
    path_grid[path_grid == 0] = -1
    start = (2 * random.randint(0, w - 1), 2 * random.randint(0, h - 1))
    end = start
    while end == start:
        end = (2 * random.randint(0, w - 1), 2 * random.randint(0, h - 1))

    path, _ = skimage.graph.route_through_array(path_grid,
                                                start,
                                                end,
                                                fully_connected=False)
    path = np.array(path).T

    grid[path[0], path[1]] = 2

    xd = np.ones([2 * w]) * pw
    xd[::2] = ww
    xd = xd.astype(int)
    xp = (np.cumsum(xd) + 0 * ww).astype(int)

    yd = np.ones([2 * h]) * pw
    yd[::2] = ww
    yd = yd.astype(int)
    yp = (np.cumsum(yd) + 0 * ww).astype(int)

    hr_grid = np.zeros([xp[-1] + ww, yp[-1] + ww])
    semantic_grid = np.zeros([xp[-1] + ww, yp[-1] + ww])
    semantic_labels = {
        'wall': 1,
        'hallway': 2,
        'goal_path': 3,
    }

    for xx in range(2 * w - 1):
        for yy in range(2 * h - 1):
            if grid[xx, yy] == 1:
                # The path is clear
                if all_wide:
                    hr_grid[xp[xx]:xp[xx] + xd[xx + 1] + dPW,
                            yp[yy]:yp[yy] + yd[yy + 1] + dPW] = 1
                    semantic_grid[xp[xx] - dPW:xp[xx] + xd[xx + 1] + dPW,
                                  yp[yy] - dPW:yp[yy] + yd[yy + 1] +
                                  dPW] = semantic_labels['hallway']
                else:
                    hr_grid[xp[xx]:xp[xx] + xd[xx + 1],
                            yp[yy]:yp[yy] + yd[yy + 1]] = 1
                    semantic_grid[xp[xx] - dPW:xp[xx] + xd[xx + 1],
                                  yp[yy] - dPW:yp[yy] +
                                  yd[yy + 1]] = semantic_labels['hallway']

    for xx in range(2 * w - 1):
        for yy in range(2 * h - 1):
            if grid[xx, yy] == 2:
                # The path connects the start and goal
                hr_grid[xp[xx] - dPW:xp[xx] + xd[xx + 1] + dPW,
                        yp[yy] - dPW:yp[yy] + yd[yy + 1] + dPW] = 1
                semantic_grid[xp[xx] - dPW:xp[xx] + xd[xx + 1] + dPW,
                              yp[yy] - dPW:yp[yy] + yd[yy + 1] +
                              dPW] = semantic_labels['goal_path']

    out_grid = np.ones(semantic_grid.shape)
    out_grid[semantic_grid == semantic_labels['goal_path']] = 0
    out_grid[semantic_grid == semantic_labels['hallway']] = 0

    start = [xp[start[0]] + pw // 2, yp[start[1]] + pw // 2]
    end = [xp[end[0]] + pw // 2, yp[end[1]] + pw // 2]

    return {
        'occ_grid': out_grid,
        'semantic_grid': semantic_grid,
        'semantic_labels': semantic_labels,
        'start': start,
        'end': end,
        'x_offset': 0.0,
        'y_offset': 0.0,
    }


class MapGenMaze(base_generator.MapGenBase):
    def gen_map(self, random_seed=None):
        map_data = gen_map_maze_base(
            random_seed=random_seed,
            path_width=self.args.map_maze_path_width,
            wide_path_width=self.args.map_maze_wide_path_width,
            height_cells=self.args.map_maze_cell_dims[0],
            width_cells=self.args.map_maze_cell_dims[1],
            all_wide=self.args.map_maze_all_wide)
        map_data['resolution'] = self.args.base_resolution

        self.hr_grid = map_data['occ_grid'].copy()
        self.grid = map_data['occ_grid'].copy()
        self.tmp_start = map_data['start']
        self.tmp_goal = map_data['end']

        return self.hr_grid, self.grid, map_data

    def get_start_goal_poses(self, min_separation=1, max_separation=1e10):
        """Loop through the points in the grid and get a pair of poses, subject to a
certain condition. This is a more specific pose generation procedure because we
want to demonstrate the more specific features of this particular map.

        Returns:
        did_succeed (Bool)
        robot_pose (NamedTuple('PoseT', 'x y yaw'))
        goal_pose (NamedTuple('PoseT', 'x y yaw'))

        """
        start = Pose(x=self.tmp_start[0],
                     y=self.tmp_start[1],
                     yaw=2 * math.pi * random.random())
        goal = Pose(x=self.tmp_goal[0],
                    y=self.tmp_goal[1],
                    yaw=2 * math.pi * random.random())

        return (True, start, goal)
