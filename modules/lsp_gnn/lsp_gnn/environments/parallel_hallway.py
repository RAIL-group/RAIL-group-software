import numpy as np
import scipy
from skimage.morphology import binary_dilation

import gridmap
from environments import base_generator
from environments.utils import calc
from common import Pose


L_TMP = 100
L_UNSET = -1
L_BKD = 0
L_CLUTTER = 1
L_DOOR = 2
L_HALL = 3
L_ROOM = 4
L_ROOM2 = 5
L_UNK = 6

HALLWAY_ROOM_SPACE = 1
DOOR_SIZE = 6

# Default parameters for the map
GRID_SIZE = (200, 210)
NUM_OF_HALLWAYS = 3
BOUNDARY_THRESHOLD = 30
MIN_SPACING_HALLWAYS = 30
HALLWAY_WIDTH = 5
ROOM_WIDTH = 20
ROOM_LENGTH_RANGE = (28, 30)

semantic_labels = {
    'background': L_BKD,
    'clutter': L_CLUTTER,
    'door': L_DOOR,
    'hallway': L_HALL,
    'room': L_ROOM,
    'room2': L_ROOM2,
    'other': L_UNK,
    'block': L_BKD
}


def inflate_lines_to_create_hallways(grid, hallway_inflation_scale):
    """Inflate the lines by a kernel

    Args:
        grid (array of shape m x n): Grid with lines
        hallway_inflation_scale (int, optional): Number pixel to grow in
        8-neighbor direction. Defaults to 5.

    Returns:
        grid_with_hallways: grid with inflated lines as hallways
    """
    original_grid = np.zeros_like(grid)
    original_grid[grid == L_UNK] = 1
    kernel_dim = 2 * hallway_inflation_scale + 1
    hallway_inflation_kernel = np.ones((kernel_dim, kernel_dim), dtype=int)

    grid_with_hallway = scipy.ndimage.convolve(
        original_grid, hallway_inflation_kernel)
    grid_with_hallway[grid_with_hallway > 0] = semantic_labels['hallway']
    grid_with_hallway[grid_with_hallway == 0] = semantic_labels['background']

    return grid_with_hallway


def add_rooms(line_segments, grid_with_hallway,
              hallway_inflation_scale, room_b, room_l_range):
    grid_with_room = grid_with_hallway.copy()
    rooms_coords = []
    for line in line_segments:
        start, end = line
        is_horizontal = start[0] == end[0]
        axis = int(is_horizontal)
        if start[axis] > end[axis]:
            start, end = end, start

        if is_horizontal:
            # add rooms on horizontal hallway end points
            # end 1
            room_l = room_l_range[0]
            room_p1 = (start[0] - int(room_l / 2), start[1] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE)
            room_p2 = (room_p1[0] + room_l, room_p1[1] + room_b)
            room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['hallway'])
                    or np.any(room_slice == semantic_labels['room2'])):
                grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = semantic_labels['room']
                rooms_coords.append((room_p1, room_p2))
                # add door
                door_p1 = (start[0] - int(DOOR_SIZE / 2), start[1] - hallway_inflation_scale - HALLWAY_ROOM_SPACE)
                door_p2 = (door_p1[0] + DOOR_SIZE, door_p1[1] + HALLWAY_ROOM_SPACE)
                grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = semantic_labels['door']

            # end 2
            room_l = room_l_range[0]
            room_q1 = (end[0] - int(room_l / 2), end[1] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1)
            room_q2 = (room_q1[0] + room_l, room_q1[1] + room_b)
            room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['hallway'])
                    or np.any(room_slice == semantic_labels['room2'])):
                grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                rooms_coords.append((room_q1, room_q2))
                # add door
                door_q1 = (end[0] - int(DOOR_SIZE / 2), end[1] + hallway_inflation_scale + 1)
                door_q2 = (door_q1[0] + DOOR_SIZE, door_q1[1] + HALLWAY_ROOM_SPACE)
                grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']

            # add rooms along horizontal hallway
            # step = np.random.randint(1, 10)
            y = start[1]
            while y < end[1] - hallway_inflation_scale:
                room_l = room_l_range[0]
                room_q1 = (start[0] + hallway_inflation_scale + 1 + HALLWAY_ROOM_SPACE, y)
                room_q2 = (room_q1[0] + room_b, room_q1[1] + room_l)
                room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])
                        or np.any(room_slice == semantic_labels['room2'])):
                    grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                    rooms_coords.append((room_q1, room_q2))
                    # add door
                    door_q1 = (room_q1[0] - HALLWAY_ROOM_SPACE, (room_q1[1] + room_q2[1]) // 2 - DOOR_SIZE // 2)
                    door_q2 = (room_q1[0], (room_q1[1] + room_q2[1]) // 2 + DOOR_SIZE // 2)
                    grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']
                step = np.random.randint(1, 10)
                y += step

    return grid_with_room, rooms_coords


def swap_room_color(seed):
    is_passage_red = seed % 2
    if is_passage_red == 0:
        return {'hallway': L_HALL, 'blue': L_ROOM, 'red': L_ROOM2}
    return {'hallway': L_HALL, 'blue': L_ROOM2, 'red': L_ROOM}


def add_blocks(grid):
    grid_with_furnitures = grid.copy()
    rooms_mask = np.ones_like(grid)
    rooms_mask[grid == semantic_labels['room']] = 0
    rooms_mask[grid == semantic_labels['room2']] = 0

    dialated_non_room_area = binary_dilation(rooms_mask, footprint=np.ones((15, 15)))
    grid_with_furnitures[dialated_non_room_area == False] = semantic_labels['block']  # noqa
    return grid_with_furnitures


def generate_parallel_lines(
        grid_size,
        num_of_lines,
        spacing_between_lines,
        boundary_threshold,
        max_iter=1000):
    ii = 0
    xy_lower_bound = boundary_threshold + 1
    y_upper_bound = grid_size[1] - boundary_threshold - 1
    grid = np.ones(grid_size, dtype=int)
    final_semantic_grid = grid.copy() * L_TMP
    intermediate_semantic_grid = grid.copy() * L_TMP
    line_segments = []
    step = grid_size[0] // num_of_lines - 5
    offset = spacing_between_lines // 2

    while ii < num_of_lines:
        # Randomly pick a point that is at a safe distance from the
        # boundaries before the inflation
        random_point = np.random.randint(
            step * ii + offset, min(step * (ii + 1) - offset, y_upper_bound)
        )

        # intermediate_semantic_grid[random_point[0], xy_lower_bound:random_point[1] + 1] = L_UNK
        intermediate_semantic_grid[random_point, xy_lower_bound:y_upper_bound + 1] = L_UNK
        final_semantic_grid = intermediate_semantic_grid.copy()
        grid[final_semantic_grid == L_UNK] = 0

        line_segments.append(([random_point, xy_lower_bound], [
            random_point, y_upper_bound]))

        ii += 1

    return final_semantic_grid, line_segments, num_of_lines


def get_sorted_line_segments(line_segments):
    result = []
    for line in line_segments:
        line = list(line)
        line.sort(key=lambda x: x[1])
        result.append(line)
        result.sort(key=lambda x: x[0][0])
    return result


def add_connecting_rooms(
        grid, line_segments, hallway_inflation_scale, room_length=10):
    grid_with_hallway = grid.copy()
    sorted_line_segments = get_sorted_line_segments(line_segments)
    rooms_coords = []
    for idx, current_line in enumerate(sorted_line_segments[:-1]):
        next_line = sorted_line_segments[idx + 1]
        low = max(current_line[0][1], next_line[0][1]) + room_length // 2 + 5
        high = min(current_line[1][1], next_line[1][1]) - room_length // 2 - 5
        if low < high:
            point_y = np.random.randint(low, high)
        else:
            raise RuntimeError('low >= high!')
        point_x = [current_line[0][0] + hallway_inflation_scale + 1,
                   next_line[0][0] - hallway_inflation_scale - 1]
        room_p1 = [point_x[0] + 1, point_y - room_length - 1]
        room_p2 = [point_x[1], point_y + room_length + 1]
        grid_with_hallway[room_p1[0]:room_p2[0],
                          room_p1[1]:room_p2[1]] = semantic_labels['room2']
        rooms_coords.append((room_p1, room_p2))
        # add door
        grid_with_hallway[
            room_p1[0] - HALLWAY_ROOM_SPACE:room_p1[0],
            (room_p1[1] + room_p2[1]) // 2 - DOOR_SIZE // 2:(room_p1[1] + room_p2[1]) // 2 + DOOR_SIZE // 2
        ] = semantic_labels['door']
        # add door on the other side
        grid_with_hallway[
            room_p2[0]:room_p2[0] + HALLWAY_ROOM_SPACE,
            (room_p1[1] + room_p2[1]) // 2 - DOOR_SIZE // 2:(room_p1[1] + room_p2[1]) // 2 + DOOR_SIZE // 2
        ] = semantic_labels['door']

    return grid_with_hallway, rooms_coords


def gen_map_parallel(random_seed, resolution, grid_size,
                     num_of_hallways, boundary_threshold,
                     min_spacing_hallways, hallway_width,
                     room_width, room_length_range):
    hallways_count = 0
    grid_with_lines, line_segments, hallways_count = \
        generate_parallel_lines(
            grid_size=grid_size, num_of_lines=num_of_hallways,
            spacing_between_lines=min_spacing_hallways,
            boundary_threshold=boundary_threshold)

    grid_with_hallway = inflate_lines_to_create_hallways(
        grid_with_lines, hallway_inflation_scale=hallway_width)

    grid_with_special_rooms, special_rooms_coords = add_connecting_rooms(
        grid_with_hallway, line_segments,
        hallway_inflation_scale=hallway_width,
        room_length=room_length_range[0] // 2)

    grid_with_rooms, rooms_coords = add_rooms(
        line_segments, grid_with_special_rooms,
        hallway_inflation_scale=hallway_width, room_b=room_width,
        room_l_range=room_length_range)
    rooms_coords += special_rooms_coords

    # Place blocks to obscure the view inside rooms
    grid = add_blocks(grid_with_rooms)

    occupancy_grid = (grid <= L_CLUTTER).astype(float)

    # Gets the wall_class_index for the current seed with change
    # like even seed red, odd seed blue connections
    wall_class_index = swap_room_color(seed=random_seed)
    polys, walls = calc.split_semantic_grid_to_polys(occupancy_grid,
                                                     grid,
                                                     wall_class_index,
                                                     resolution=resolution,
                                                     do_compute_walls=True)

    return {
        "occ_grid": occupancy_grid.copy(),
        "semantic_grid": grid.copy(),
        "semantic_labels": semantic_labels,
        "polygons": polys,
        "walls": walls,
        "x_offset": 0.0,
        "y_offset": 0.0,
        "resolution": resolution,
        "hallways_count": hallways_count,
        "wall_class": wall_class_index
    }


class MapGenParallel(base_generator.MapGenBase):
    def gen_map(self, random_seed=None):
        map_data = gen_map_parallel(
            random_seed, resolution=self.args.base_resolution,
            grid_size=GRID_SIZE,
            num_of_hallways=NUM_OF_HALLWAYS,
            boundary_threshold=BOUNDARY_THRESHOLD,
            min_spacing_hallways=MIN_SPACING_HALLWAYS,
            hallway_width=HALLWAY_WIDTH,
            room_width=ROOM_WIDTH,
            room_length_range=ROOM_LENGTH_RANGE)

        self.hr_grid = map_data["occ_grid"].copy()
        self.grid = map_data["occ_grid"].copy()
        self.semantic_grid = map_data['semantic_grid'].copy()
        return self.hr_grid, self.grid, map_data

    def get_start_goal_poses(self, min_separation=220, max_separation=1e10, num_attemps=10000):
        inflation_radius = self.args.inflation_radius_m / self.args.base_resolution
        inflated_grid = gridmap.utils.inflate_grid(self.grid, inflation_radius)
        hallway_cells = np.column_stack(np.where(
            self.semantic_grid == semantic_labels['hallway']))
        free_cells = np.column_stack(np.where(
            inflated_grid == gridmap.constants.FREE_VAL))
        nrows, ncols = hallway_cells.shape
        dtype = {'names' : ['f{}'.format(i) for i in range(ncols)],
                 'formats' : ncols * [hallway_cells.dtype]}
        usable_cells = np.intersect1d(hallway_cells.view(dtype), free_cells.view(dtype))
        usable_cells = usable_cells.view(hallway_cells.dtype).reshape(-1, ncols)

        start_cell_pool = []
        goal_cell_pool = []
        for free_cell in usable_cells:
            if free_cell[0] <= 50:
                start_cell_pool.append(free_cell)
            elif free_cell[0] >= 140:
                goal_cell_pool.append(free_cell)

        for _ in range(num_attemps):
            rand_start_index = np.random.choice(
                np.arange(len(start_cell_pool)), size=1, replace=False)[0]
            rand_goal_index = np.random.choice(
                np.arange(len(goal_cell_pool)), size=1, replace=False)[0]
            start = start_cell_pool[rand_start_index]
            goal = goal_cell_pool[rand_goal_index]

            cost_grid, get_path = gridmap.planning. \
                compute_cost_grid_from_position(inflated_grid, goal)
            did_plan, _ = get_path([start[0], start[1]],
                                   do_sparsify=False,
                                   do_flip=False)
            path_cost = cost_grid[start[0], start[1]]
            if path_cost >= min_separation and path_cost <= max_separation and did_plan:
                start = Pose(x=start[0], y=start[1])
                goal = Pose(x=goal[0], y=goal[1])
                return (True, start, goal)
        return (False, None, None)
