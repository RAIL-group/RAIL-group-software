import math
import numpy as np
import random

from common import Pose
from environments import base_generator
from environments.utils import calc


def gen_map_jshaped_base(resolution=0.4,
                         random_seed=None):
    def draw_horizontal(grid, x1, y1, x2, y2, width=10, value=1):
        grid[x1:x1 + width, y1:y2] = value

    def draw_vertical(grid, x1, y1, x2, y2, width=10, value=1):
        grid[x1:x2, y1:y1 + width] = value

    def goal_on_right_map(width=10):
        grid = np.zeros((100, 110))
        draw_horizontal(grid, x1=80, y1=10, x2=width, y2=50)
        draw_horizontal(grid, x1=80, y1=51, x2=width, y2=100)
        draw_vertical(grid, x1=10, y1=10, x2=80, y2=width)
        draw_vertical(grid, x1=10, y1=90, x2=80, y2=width)
        draw_horizontal(grid, x1=10, y1=10, x2=width, y2=100)
        draw_vertical(grid, x1=10, y1=50, x2=70, y2=width)
        draw_vertical(grid, x1=30, y1=30, x2=60, y2=width)
        draw_horizontal(grid, x1=60, y1=30, x2=width, y2=50)
        return 1 - grid

    def goal_on_left_map(width=10):
        grid = np.zeros((100, 110))
        draw_horizontal(grid, x1=80, y1=10, x2=width, y2=59)
        draw_vertical(grid, x1=10, y1=10, x2=80, y2=width)
        draw_horizontal(grid, x1=80, y1=60, x2=width, y2=100)
        draw_vertical(grid, x1=10, y1=90, x2=80, y2=width)
        draw_horizontal(grid, x1=10, y1=10, x2=width, y2=50)
        draw_horizontal(grid, x1=10, y1=60, x2=width, y2=100)
        draw_vertical(grid, x1=10, y1=50, x2=70, y2=width)
        draw_vertical(grid, x1=30, y1=30, x2=60, y2=width)
        draw_horizontal(grid, x1=60, y1=30, x2=width, y2=50)
        return 1 - grid

    def apply_color_path_right(map, width=10, color=.5):
        draw_horizontal(map, x1=80, y1=10, x2=width, y2=50, value=color)
        draw_vertical(map, x1=10, y1=10, x2=80, y2=width, value=color)
        draw_horizontal(map, x1=10, y1=10, x2=width, y2=55, value=color)
        return map

    def apply_color_path_right2(map, width=10, color=.5):
        draw_horizontal(map, x1=80, y1=51, x2=width, y2=100, value=color)
        draw_vertical(map, x1=10, y1=90, x2=80, y2=width, value=color)
        draw_horizontal(map, x1=10, y1=55, x2=width, y2=100, value=color)
        return map

    def apply_color_path_left(map, width=10, color=.5):
        draw_horizontal(map, x1=80, y1=60, x2=width, y2=100, value=color)
        draw_vertical(map, x1=10, y1=90, x2=80, y2=width, value=color)
        draw_horizontal(map, x1=10, y1=55, x2=width, y2=100, value=color)
        return map

    def apply_color_path_left2(map, width=10, color=.5):
        draw_horizontal(map, x1=80, y1=10, x2=width, y2=59, value=color)
        draw_vertical(map, x1=10, y1=10, x2=80, y2=width, value=color)
        draw_horizontal(map, x1=10, y1=10, x2=width, y2=55, value=color)
        return map

    def apply_color_j(map, width=10, color=.5):
        draw_vertical(map, x1=30, y1=30, x2=40, y2=width, value=color)
        return map

    semantic_labels = {
        'wall': 1,
        'hallway': 2,
        'blue': 3,
        'red': 4,
    }
    # Initialize the random generator
    random.seed(random_seed)
    np.random.seed(random_seed)
    rando = np.random.rand()
    map_type = ''  # l or r
    goal_color = None  # r or b
    other_color = None  # r or b
    # print(f"Random seed: {random_seed} - rando: {rando}")
    if rando > .5:
        print("Left")
        map_type = 'l'
        grid = goal_on_left_map()

    else:
        print("Right")
        map_type = 'r'
        grid = goal_on_right_map()

    rando = np.random.rand()
    if rando > .5:
        print("Goal looks blue")
        goal_color = semantic_labels['blue']
        other_color = semantic_labels['red']
    else:
        print("Goal looks red")
        goal_color = semantic_labels['red']
        other_color = semantic_labels['blue']

    if map_type == 'l':
        semantic_grid = apply_color_path_left(
            grid.copy(), color=other_color)
        semantic_grid = apply_color_path_left2(
            semantic_grid.copy(), color=goal_color)
    elif map_type == 'r':
        semantic_grid = apply_color_path_right(
            grid.copy(), color=other_color)
        semantic_grid = apply_color_path_right2(
            semantic_grid.copy(), color=goal_color)

    semantic_grid = apply_color_j(
        semantic_grid.copy(), color=goal_color)
    semantic_grid[semantic_grid == 0] = semantic_labels['hallway']
    assert len(np.argwhere(semantic_grid == 0)) == 0

    start = (34, 44)
    # rando = np.random.rand()
    # if rando > .5:
    #     print("Flipped the map horizontally")
    #     start = (74, 44)
    #     grid = np.flip(grid, 1)
    #     semantic_grid = np.flip(semantic_grid, 1)

    end = (54, 85)

    occ_grid = np.transpose(grid)
    semantic_grid = np.transpose(semantic_grid)

    wall_class_index = {
        'hallway': semantic_labels['hallway'],
        'blue': semantic_labels['blue'],
        'red': semantic_labels['red']
    }
    polys, walls = calc.split_semantic_grid_to_polys(occ_grid,
                                                     semantic_grid,
                                                     wall_class_index,
                                                     resolution,
                                                     do_compute_walls=True)
    # start = (55, 15)  # Shows fork view
    return {
        'occ_grid': occ_grid,
        'semantic_grid': semantic_grid,
        'semantic_labels': semantic_labels,
        "polygons": polys,
        "walls": walls,
        'start': start,
        'end': end,
        'x_offset': 0.0,
        'y_offset': 0.0,
        "wall_class": wall_class_index
    }


class MapGenJshaped(base_generator.MapGenBase):
    def gen_map(self, random_seed=None):
        map_data = gen_map_jshaped_base(
            random_seed=random_seed,
            resolution=self.args.base_resolution)
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
                     yaw=0)
        goal = Pose(x=self.tmp_goal[0],
                    y=self.tmp_goal[1],
                    yaw=2 * math.pi * random.random())

        return (True, start, goal)
