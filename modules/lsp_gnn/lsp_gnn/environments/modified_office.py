import numpy as np
from scipy.ndimage import label
import scipy
from skimage.morphology import skeletonize
import sknw

from environments import base_generator
from environments.utils import calc
from common import Pose
import gridmap


L_TMP = 100
L_UNSET = -1
L_BKD = 0
L_CLUTTER = 1
L_DOOR = 2
L_HALL = 3
L_ROOM = 4
L_ROOM2 = 5
L_UNK = 6

# default parameters
RESOLUTION = 0.5  # this value yields a reasonable scale of office environment in sim
INFLATION_RADIUS_M = RESOLUTION * 1.5

# the default parameters below have units in terms of grid cells
GRID_SIZE = (500, 300)
NUM_OF_HALLWAYS = 5
BOUNDARY_THRESHOLD = 30
MIN_SPACING_HALLWAYS = 30
HALLWAY_WIDTH = 5
ROOM_WIDTH = 20
ROOM_LENGTH_RANGE = (25, 35)
ROOM_DOOR_SPACE = 1
HALLWAY_ROOM_SPACE = 1
DOOR_SIZE = 8
MAX_TABLES_PER_ROOM = 2
TABLE_SIZE_RANGE = (4, 8)
TABLE_WALL_BUFFER = 3

semantic_labels = {
    'background': L_BKD,
    'clutter': L_CLUTTER,
    'door': L_DOOR,
    'hallway': L_HALL,
    'room': L_ROOM,
    'room2': L_ROOM2,
    'other': L_UNK,
}


def generate_random_lines(num_of_lines=NUM_OF_HALLWAYS,
                          grid_size=GRID_SIZE,
                          spacing_between_lines=MIN_SPACING_HALLWAYS,
                          boundary_threshold=BOUNDARY_THRESHOLD,
                          max_iter=10000):
    """Generate random horizontal and vertical lines in a grid.

    Args:
        seed (int): Random seed
        num_of_lines (integer): Number of lines
        grid_size (tuple): Size of the grid
        spacing_between_lines (int): Spacing between two parallel lines
        boundary_threshold (int): Minimum spacing between lines and boundary of grid
        max_iter (int): Maximum number of iterations to run

    Returns:
        final_semantic_grid (2D array): Grid with horizontal and vertical lines
        line_segments: list of line segments, each segment as ((start_x, start_y), (end_x, end_y))
    """

    def _check_if_connected(semantic_grid, grid):
        # 8-neighbor structure/kernel for connected components
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        grid[semantic_grid == L_UNK] = 0
        new_grid = 1 - grid.copy()
        _, num_features = label(new_grid, structure=s)
        if num_features > 1:
            return False
        else:
            return True

    # keep track of the boundaries within which parallel line can't be drawn in row and col
    row = set()
    col = set()
    # space between parallel hallways
    space_between_parallel = spacing_between_lines
    # lower bound of both x and y
    xy_lower_bound = 0 + boundary_threshold + 1
    x_upper_bound = grid_size[0] - boundary_threshold - 1
    y_upper_bound = grid_size[1] - boundary_threshold - 1

    grid = np.ones(grid_size, dtype=int)
    final_semantic_grid = grid.copy() * L_TMP
    intermediate_semantic_grid = grid.copy() * L_TMP
    line_segments = []

    for _ in range(max_iter):
        # Randomly pick a point that is at a safe distance from the
        # boundaries before the inflation
        random_point = np.random.randint(
            xy_lower_bound, [x_upper_bound, y_upper_bound]
        )

        # finds the distance from every boundaries
        distance_to_bounds = [x_upper_bound - random_point[0],
                              random_point[0] - xy_lower_bound + 1,
                              random_point[1] - xy_lower_bound + 1,
                              y_upper_bound - random_point[1]]

        # direction specifies where the line should proceed
        direction = np.argmax(distance_to_bounds)
        sorted_direction = np.argsort(distance_to_bounds)[::-1]

        for direction in sorted_direction:

            if direction == 0:  # Draws from top to bottom (top left is (0,0))
                if random_point[1] in col:
                    continue
                intermediate_semantic_grid[random_point[0]:x_upper_bound + 1,
                                           random_point[1]] = L_UNK

                line_connected = _check_if_connected(
                    intermediate_semantic_grid, grid.copy())
                if line_connected:
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == L_UNK] = 0
                    lb = max(random_point[1] -
                             space_between_parallel, xy_lower_bound)
                    ub = min(
                        random_point[1] + space_between_parallel + 1, y_upper_bound)
                    lb_buffer = max(
                        random_point[0] - space_between_parallel, xy_lower_bound)

                    line_segments.append(((random_point[0], random_point[1]),
                                          (x_upper_bound, random_point[1])))

                    col.update(range(lb, ub))
                    row.update(range(lb_buffer, random_point[0]))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

            elif direction == 1:  # Draws from bottom to top
                if random_point[1] in col:
                    continue
                intermediate_semantic_grid[xy_lower_bound:random_point[0] + 1, random_point[1]] = L_UNK

                line_connected = _check_if_connected(
                    intermediate_semantic_grid, grid.copy())
                if line_connected:
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == L_UNK] = 0
                    lb = max(random_point[1] -
                             space_between_parallel, xy_lower_bound)
                    ub = min(
                        random_point[1] + space_between_parallel + 1, y_upper_bound)
                    ub_buffer = min(
                        random_point[0] + space_between_parallel, x_upper_bound)

                    line_segments.append(((random_point[0], random_point[1]),
                                          (xy_lower_bound, random_point[1])))

                    col.update(range(lb, ub))
                    row.update(range(random_point[0], ub_buffer))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

            elif direction == 2:  # Draws from right to left
                if random_point[0] in row:
                    continue
                intermediate_semantic_grid[random_point[0], xy_lower_bound:random_point[1] + 1] = L_UNK
                line_connected = _check_if_connected(
                    intermediate_semantic_grid, grid.copy())
                if line_connected:
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == L_UNK] = 0
                    lb = max(random_point[0] -
                             space_between_parallel, xy_lower_bound)
                    ub = min(
                        random_point[0] + space_between_parallel + 1, x_upper_bound)
                    ub_buffer = min(
                        random_point[1] + space_between_parallel, y_upper_bound)

                    line_segments.append(((random_point[0], random_point[1]),
                                          (random_point[0], xy_lower_bound)))
                    row.update(range(lb, ub))
                    col.update(range(random_point[1], ub_buffer))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

            elif direction == 3:  # Draws from left to right
                if random_point[0] in row:
                    continue
                intermediate_semantic_grid[random_point[0], random_point[1]:y_upper_bound + 1] = L_UNK
                line_connected = _check_if_connected(
                    intermediate_semantic_grid, grid.copy())
                if line_connected:
                    final_semantic_grid = intermediate_semantic_grid.copy()
                    grid[final_semantic_grid == L_UNK] = 0
                    grid[final_semantic_grid == L_UNK] = 0
                    lb = max(random_point[0] -
                             space_between_parallel, xy_lower_bound)
                    ub = min(
                        random_point[0] + space_between_parallel + 1, x_upper_bound)
                    lb_buffer = max(
                        random_point[1] - space_between_parallel, xy_lower_bound)

                    line_segments.append(((random_point[0], random_point[1]),
                                          (random_point[0], y_upper_bound)))

                    row.update(range(lb, ub))
                    col.update(range(lb_buffer, random_point[1]))
                    break
                else:
                    intermediate_semantic_grid = final_semantic_grid.copy()

        if len(line_segments) >= num_of_lines:
            break
    else:
        print(f"Needed to generate {num_of_lines} lines but only generated {len(line_segments)} lines.")

    return final_semantic_grid, line_segments


def inflate_lines_to_create_hallways(grid, hallway_inflation_scale=HALLWAY_WIDTH):
    """Inflate the lines by a kernel.

    Args:
        grid (2D array): Grid with lines
        hallway_inflation_scale (int): Number pixel to grow in 8-neighbor direction

    Returns:
        grid_with_hallway (2D array): grid with inflated lines as hallways
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


def add_rooms(grid_with_hallway,
              line_segments,
              hallway_inflation_scale=HALLWAY_WIDTH,
              room_b=ROOM_WIDTH,
              room_l_range=ROOM_LENGTH_RANGE):
    """Add rooms to the hallway grid (along the hallways).

    Args:
        grid_with_hallway (2D array): Grid with hallways (can contain special rooms too)
        line_segments (list): List of line segments corresponding to hallways,
                              each segment as ((start_x, start_y), (end_x, end_y))
        hallway_inflation_scale (int): Hallway width (same as kernel size used to create hallways from lines)
        room_b (int): Width/breadth of room
        room_l_range (tuple):  Tuple of two integers representing (minimum, maximum) lengths of rooms

    Returns:
        grid_with_room (2D array): Grid with rooms
        rooms_coords (list): List of room coordinates, each coordinate as ((start_x, start_y), (end_x, end_y))
    """
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
            room_l = np.random.randint(*room_l_range)
            room_p1 = (start[0] - int(room_l / 2), start[1] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE)
            room_p2 = (room_p1[0] + room_l, room_p1[1] + room_b)
            room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['room2'])
                    or np.any(room_slice == semantic_labels['hallway'])):
                grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = semantic_labels['room']
                rooms_coords.append((room_p1, room_p2))
                # add door
                door_p1 = (start[0] - int(DOOR_SIZE / 2), start[1] - hallway_inflation_scale - HALLWAY_ROOM_SPACE)
                door_p2 = (door_p1[0] + DOOR_SIZE, door_p1[1] + HALLWAY_ROOM_SPACE)
                grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = semantic_labels['door']

            # end 2
            room_l = np.random.randint(*room_l_range)
            room_q1 = (end[0] - int(room_l / 2), end[1] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1)
            room_q2 = (room_q1[0] + room_l, room_q1[1] + room_b)
            room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['room2'])
                    or np.any(room_slice == semantic_labels['hallway'])):
                grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                rooms_coords.append((room_q1, room_q2))
                # add door
                door_q1 = (end[0] - int(DOOR_SIZE / 2), end[1] + hallway_inflation_scale + 1)
                door_q2 = (door_q1[0] + DOOR_SIZE, door_q1[1] + HALLWAY_ROOM_SPACE)
                grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']

            # add rooms along horizontal hallway
            for y in range(start[1], end[1] - hallway_inflation_scale, 1):
                room_l = np.random.randint(*room_l_range)
                room_p1 = (start[0] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE, y)
                room_p2 = (room_p1[0] + room_b, room_p1[1] + room_l)
                room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['room2'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = semantic_labels['room']
                    rooms_coords.append((room_p1, room_p2))
                    # add door
                    door_p1 = (room_p2[0], room_p2[1] - ROOM_DOOR_SPACE - DOOR_SIZE)
                    door_p2 = (room_p2[0] + HALLWAY_ROOM_SPACE, room_p2[1] - ROOM_DOOR_SPACE)
                    # correction for door extending beyond hallway end 2
                    door_check_slice = grid_with_room[door_p2[0]:door_p2[0] + 1, door_p1[1]:door_p2[1]]
                    overflow_len = len(np.where(door_check_slice == semantic_labels['background'])[1])
                    if overflow_len > 0:
                        door_p1 = (room_p1[0] + room_b, room_p1[1] + ROOM_DOOR_SPACE)
                        door_p2 = (door_p1[0] + HALLWAY_ROOM_SPACE, door_p1[1] + DOOR_SIZE)
                    grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = semantic_labels['door']

                room_l = np.random.randint(*room_l_range)
                room_q1 = (start[0] + hallway_inflation_scale + 1 + HALLWAY_ROOM_SPACE, y)
                room_q2 = (room_q1[0] + room_b, room_q1[1] + room_l)
                room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['room2'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                    rooms_coords.append((room_q1, room_q2))
                    # add door
                    door_q1 = (room_q1[0] - HALLWAY_ROOM_SPACE, room_q1[1] + ROOM_DOOR_SPACE)
                    door_q2 = (room_q1[0], room_q1[1] + ROOM_DOOR_SPACE + DOOR_SIZE)
                    grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']

        else:
            # add rooms on vertical hallway end points
            # end 1
            room_l = np.random.randint(*room_l_range)
            room_p1 = (start[0] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE, start[1] - int(room_l / 2))
            room_p2 = (room_p1[0] + room_b, room_p1[1] + room_l)
            room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['room2'])
                    or np.any(room_slice == semantic_labels['hallway'])):
                grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = semantic_labels['room']
                rooms_coords.append((room_p1, room_p2))
                # add door
                door_p1 = (start[0] - hallway_inflation_scale - HALLWAY_ROOM_SPACE, start[1] - int(DOOR_SIZE / 2))
                door_p2 = (door_p1[0] + HALLWAY_ROOM_SPACE, door_p1[1] + DOOR_SIZE)
                grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = semantic_labels['door']

            # end 2
            room_l = np.random.randint(*room_l_range)
            room_q1 = (end[0] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1, end[1] - int(room_l / 2))
            room_q2 = (room_q1[0] + room_b, room_q1[1] + room_l)
            room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
            if not (np.any(room_slice == semantic_labels['room'])
                    or np.any(room_slice == semantic_labels['room2'])
                    or np.any(room_slice == semantic_labels['hallway'])):
                grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                rooms_coords.append((room_q1, room_q2))
                # add door
                door_q1 = (end[0] + hallway_inflation_scale + 1, end[1] - int(DOOR_SIZE / 2))
                door_q2 = (door_q1[0] + HALLWAY_ROOM_SPACE, door_q1[1] + DOOR_SIZE)
                grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']

            # add rooms along vertical hallway
            for x in range(start[0], end[0] - hallway_inflation_scale, 1):
                room_l = np.random.randint(*room_l_range)
                room_p1 = (x, start[1] - hallway_inflation_scale - room_b - HALLWAY_ROOM_SPACE)
                room_p2 = (room_p1[0] + room_l, room_p1[1] + room_b)
                room_slice = grid_with_room[room_p1[0] - 1:room_p2[0] + 1, room_p1[1] - 1:room_p2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['room2'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_room[room_p1[0]:room_p2[0], room_p1[1]:room_p2[1]] = semantic_labels['room']
                    rooms_coords.append((room_p1, room_p2))
                    # add door
                    door_p1 = (room_p2[0] - ROOM_DOOR_SPACE - DOOR_SIZE, room_p2[1])
                    door_p2 = (room_p2[0] - ROOM_DOOR_SPACE, room_p2[1] + HALLWAY_ROOM_SPACE)
                    # correction for door extending beyond hallway end 2
                    door_check_slice = grid_with_room[door_p1[0]:door_p2[0], door_p2[1]:door_p2[1] + 1]
                    overflow_len = len(np.where(door_check_slice == semantic_labels['background'])[0])
                    if overflow_len > 0:
                        door_p1 = (room_p1[0] + ROOM_DOOR_SPACE, room_p1[1] + room_b)
                        door_p2 = (door_p1[0] + DOOR_SIZE, door_p1[1] + HALLWAY_ROOM_SPACE)
                    grid_with_room[door_p1[0]:door_p2[0], door_p1[1]:door_p2[1]] = semantic_labels['door']

                room_l = np.random.randint(*room_l_range)
                room_q1 = (x, start[1] + hallway_inflation_scale + 1 + HALLWAY_ROOM_SPACE)
                room_q2 = (room_q1[0] + room_l, room_q1[1] + room_b)
                room_slice = grid_with_room[room_q1[0] - 1:room_q2[0] + 1, room_q1[1] - 1:room_q2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['room2'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_room[room_q1[0]:room_q2[0], room_q1[1]:room_q2[1]] = semantic_labels['room']
                    rooms_coords.append((room_q1, room_q2))
                    # add door
                    door_q1 = (room_q1[0] + ROOM_DOOR_SPACE, room_q1[1] - HALLWAY_ROOM_SPACE)
                    door_q2 = (room_q1[0] + ROOM_DOOR_SPACE + DOOR_SIZE, room_q1[1])
                    grid_with_room[door_q1[0]:door_q2[0], door_q1[1]:door_q2[1]] = semantic_labels['door']

    return grid_with_room, rooms_coords


def add_special_rooms(grid_with_hallway,
                      intersections,
                      hallway_inflation_scale=HALLWAY_WIDTH,
                      room_length_range=ROOM_LENGTH_RANGE):
    """Add special rooms to the hallway grid wherever possible to connect two hallways.

    Args:
        grid_with_hallway (2D array): Grid with hallways
        intersections (list): List of intersection points
        hallway_inflation_scale (int): Hallway width (same as kernel size used to create hallways from lines)
        room_l_range (tuple): Tuple of two integers representing (minimum, maximum) lengths of rooms

    Returns:
        grid_with_sp_room (2D array): Grid with special rooms
        rooms_coords (list): List of special room coordinates, each coordinate as ((start_x, start_y), (end_x, end_y))
    """
    def _check_intersection_or_hallway_end(side_point, extended_point):
        check_point_start = side_point[0]
        check_point_end = side_point[1]
        hallway_end_check_start = extended_point[0]
        hallway_end_check_end = extended_point[1]

        another_intersection_met, hallway_end = False, False

        if grid_with_sp_room[check_point_start[0], check_point_start[1]] == semantic_labels['hallway']:
            another_intersection_met = True

        if grid_with_sp_room[check_point_end[0], check_point_end[1]] == semantic_labels['hallway']:
            another_intersection_met = True

        if grid_with_sp_room[hallway_end_check_start[0], hallway_end_check_start[1]] == semantic_labels['background']:
            hallway_end = True
        if grid_with_sp_room[hallway_end_check_end[0], hallway_end_check_end[1]] == semantic_labels['background']:
            hallway_end = True

        return another_intersection_met, hallway_end

    grid_with_sp_room = grid_with_hallway.copy()
    intersections = np.round(intersections).astype(int)
    intersection_with_distance = []
    for i, inter in enumerate(intersections):
        x, y = inter[0], inter[1]
        for next_point in intersections[i + 1:]:
            if next_point[0] == x or next_point[1] == y:
                intersection_with_distance.append(
                    [[inter, next_point], np.linalg.norm(inter - next_point)])
    intersection_with_distance.sort(key=lambda x: x[1])

    # if intersection distance less than a certain room size; remove the intersection
    min_room_length, max_room_length = room_length_range[0], room_length_range[1]
    min_intersection_distance = min_room_length + 3 * hallway_inflation_scale
    max_intersection_distance = max_room_length + 5 * hallway_inflation_scale

    intersection_with_distance = [intersection for intersection in intersection_with_distance if (
        intersection[1] >= min_intersection_distance and intersection[1] < max_intersection_distance)]

    rooms_coords = []
    for intersection in intersection_with_distance:
        # find whether the line is horizontal or vertical.
        start, end = intersection[0]
        is_horizontal = start[0] == end[0]
        axis = int(is_horizontal)
        if start[axis] > end[axis]:
            start, end = end, start
        distance = {}
        if (is_horizontal):
            '''
            find the minimum distance along the hallway in which the room can be expanded
            in the ascending direction
            '''
            another_intersection_met = False
            hallway_end = False
            distance_ascending = hallway_inflation_scale
            while (not (another_intersection_met or hallway_end)):
                distance_ascending += 1
                poi_ascending = start[0] + distance_ascending

                check_point_start = [poi_ascending,
                                     start[1] + hallway_inflation_scale + 1]
                check_point_end = [poi_ascending,
                                   end[1] - hallway_inflation_scale - 1]

                hallway_end_check_start = [poi_ascending + 1, start[1]]
                hallway_end_check_end = [poi_ascending + 1, end[1]]

                side_points = [check_point_start, check_point_end]
                extended_points = [
                    hallway_end_check_start, hallway_end_check_end]

                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    side_points, extended_points)

            distance['ascending'] = distance_ascending
            '''
            find the minimum distance along the hallway in which the room can be expanded
            in the descending direction
            '''
            another_intersection_met = False
            hallway_end = False
            distance_descending = hallway_inflation_scale
            while (not (another_intersection_met or hallway_end)):
                distance_descending += 1
                poi_descending = start[0] - distance_descending

                check_point_start = [poi_descending,
                                     start[1] + hallway_inflation_scale + 1]
                check_point_end = [poi_descending,
                                   end[1] - hallway_inflation_scale - 1]

                hallway_end_check_start = [poi_descending - 1, start[1]]
                hallway_end_check_end = [poi_descending - 1, end[1]]

                side_points = [check_point_start, check_point_end]
                extended_points = [
                    hallway_end_check_start, hallway_end_check_end]

                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    side_points, extended_points)
            distance['descending'] = distance_descending

            # Add "special room" in the ascending direction
            if distance['ascending'] > max_room_length:
                room_p1 = [start[0] + distance['ascending'] - max_room_length,
                           start[1] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1]
                room_p2 = [end[0] + distance['ascending'], end[1] - hallway_inflation_scale - HALLWAY_ROOM_SPACE]
                room_slice = grid_with_sp_room[room_p1[0] -
                                               1:room_p2[0] + 1, room_p1[1]:room_p2[1]]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_sp_room[room_p1[0]:room_p2[0],
                                      room_p1[1]:room_p2[1]] = semantic_labels['room2']
                    rooms_coords.append((room_p1, room_p2))
                    # add doors
                    grid_with_sp_room[room_p1[0] + ROOM_DOOR_SPACE:room_p1[0] + ROOM_DOOR_SPACE + DOOR_SIZE,
                                      room_p1[1] - HALLWAY_ROOM_SPACE:room_p1[1]] = semantic_labels['door']
                    grid_with_sp_room[room_p2[0] - ROOM_DOOR_SPACE - DOOR_SIZE:room_p2[0] - ROOM_DOOR_SPACE,
                                      room_p2[1]:room_p2[1] + HALLWAY_ROOM_SPACE] = semantic_labels['door']

            # Add "special room" in the descending direction
            if distance['descending'] > max_room_length:
                room_q1 = [start[0] - distance['descending'] + 1, start[1] +
                           hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1]
                room_q2 = [end[0] - distance['descending'] + max_room_length,
                           end[1] - hallway_inflation_scale - HALLWAY_ROOM_SPACE]
                room_slice = grid_with_sp_room[room_q1[0] -
                                               1:room_q2[0] + 1, room_q1[1]:room_q2[1]]
                if not (np.any(room_slice == semantic_labels['room2'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_sp_room[room_q1[0]:room_q2[0],
                                      room_q1[1]:room_q2[1]] = semantic_labels['room']
                    rooms_coords.append((room_q1, room_q2))
                    # add doors
                    grid_with_sp_room[room_q1[0] + ROOM_DOOR_SPACE:room_q1[0] + ROOM_DOOR_SPACE + DOOR_SIZE,
                                      room_q1[1] - HALLWAY_ROOM_SPACE:room_q1[1]] = semantic_labels['door']
                    grid_with_sp_room[room_q2[0] - ROOM_DOOR_SPACE - DOOR_SIZE:room_q2[0] - ROOM_DOOR_SPACE,
                                      room_q2[1]:room_q2[1] + HALLWAY_ROOM_SPACE] = semantic_labels['door']

        else:
            '''
            find the minimum distance along the hallway in which the room can be expanded
            in the ascending direction
            '''
            another_intersection_met = False
            hallway_end = False
            distance_ascending = hallway_inflation_scale
            while (not (another_intersection_met or hallway_end)):
                distance_ascending += 1
                poi_ascending = start[1] + distance_ascending
                check_point_start = [
                    start[0] + hallway_inflation_scale + 1, poi_ascending]
                check_point_end = [
                    end[0] - hallway_inflation_scale - 1, poi_ascending]

                hallway_end_check_start = [start[0], poi_ascending + 1]
                hallway_end_check_end = [end[0], poi_ascending + 1]
                side_points = [check_point_start, check_point_end]
                extended_points = [
                    hallway_end_check_start, hallway_end_check_end]
                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    side_points, extended_points)

            distance['ascending'] = distance_ascending
            '''
            find the minimum distance along the hallway in which the room can be expanded
            in the descending direction
            '''
            another_intersection_met = False
            hallway_end = False
            distance_descending = hallway_inflation_scale
            while (not (another_intersection_met or hallway_end)):
                distance_descending += 1
                poi_descending = start[1] - distance_descending
                check_point_start = [
                    start[0] + hallway_inflation_scale + 1, poi_descending]
                check_point_end = [
                    end[0] - hallway_inflation_scale - 1, poi_descending]

                hallway_end_check_start = [start[0], poi_descending - 1]
                hallway_end_check_end = [end[0], poi_descending - 1]

                side_points = [check_point_start, check_point_end]
                extended_points = [
                    hallway_end_check_start, hallway_end_check_end]
                another_intersection_met, hallway_end = _check_intersection_or_hallway_end(
                    side_points, extended_points)
            distance['descending'] = distance_descending

            # Add "special room" in the ascending direction
            if distance['ascending'] > max_room_length:
                room_p1 = [start[0] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1,
                           start[1] + distance['ascending'] - max_room_length]
                room_p2 = [end[0] - hallway_inflation_scale -
                           HALLWAY_ROOM_SPACE, end[1] + distance['ascending'] - 1]
                room_slice = grid_with_sp_room[room_p1[0]:room_p2[0], room_p1[1] - 1:room_p2[1] + 1]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_sp_room[room_p1[0]:room_p2[0],
                                      room_p1[1]:room_p2[1]] = semantic_labels['room2']
                    rooms_coords.append((room_p1, room_p2))
                    # add doors
                    grid_with_sp_room[room_p1[0] - HALLWAY_ROOM_SPACE:room_p1[0],
                                      room_p1[1] + ROOM_DOOR_SPACE:room_p1[1] + ROOM_DOOR_SPACE + DOOR_SIZE] = (
                        semantic_labels['door'])
                    grid_with_sp_room[room_p2[0]:room_p2[0] + HALLWAY_ROOM_SPACE,
                                      room_p2[1] - ROOM_DOOR_SPACE - DOOR_SIZE:room_p2[1] - ROOM_DOOR_SPACE] = (
                        semantic_labels['door'])
            # Add "special room" in the descending direction
            if distance['descending'] > max_room_length:
                room_q1 = [start[0] + hallway_inflation_scale + HALLWAY_ROOM_SPACE + 1,
                           start[1] - distance['descending']]
                room_q2 = [end[0] - hallway_inflation_scale - HALLWAY_ROOM_SPACE,
                           end[1] - distance['descending'] + max_room_length]
                room_slice = grid_with_sp_room[room_q1[0] +
                                               1:room_q2[0] - 1, room_q1[1]:room_q2[1]]
                if not (np.any(room_slice == semantic_labels['room'])
                        or np.any(room_slice == semantic_labels['hallway'])):
                    grid_with_sp_room[room_q1[0]:room_q2[0],
                                      room_q1[1]:room_q2[1]] = semantic_labels['room2']
                    rooms_coords.append((room_q1, room_q2))
                    # add doors
                    grid_with_sp_room[room_q1[0] - HALLWAY_ROOM_SPACE:room_q1[0],
                                      room_q1[1] + ROOM_DOOR_SPACE:room_q1[1] + ROOM_DOOR_SPACE + DOOR_SIZE] = (
                        semantic_labels['door'])
                    grid_with_sp_room[room_q2[0]:room_q2[0] + HALLWAY_ROOM_SPACE,
                                      room_q2[1] - ROOM_DOOR_SPACE - DOOR_SIZE:room_q2[1] - ROOM_DOOR_SPACE] = (
                        semantic_labels['door'])

    return grid_with_sp_room, rooms_coords


def add_tables(grid_with_rooms,
               rooms_coords,
               max_tables_per_room=MAX_TABLES_PER_ROOM,
               table_size_range=TABLE_SIZE_RANGE,
               table_wall_buffer=TABLE_WALL_BUFFER):
    """Add tables to grid with rooms.

    Args:
        grid_with_rooms (2D array): Grid with rooms
        rooms_coords (list): List of all room coordinates, each coordinate as ((start_x, start_y), (end_x, end_y))
        max_tables_per_room (int): Maximum number of tables to per room
        table_size_range (tuple): Tuple of two integers representing (minimum, maximum) lengths of tables
        table_wall_buffer (int): Minimum spacing between table and room walls

    Returns:
        grid_with_tables (2D array): Grid with tables
        table_poses_sizes (list): List of tuples containing table center coordinates and lengths,
                                  each tuple as (center_x, center_y, length_x, length_y)
    """
    grid_with_tables = grid_with_rooms.copy()
    table_poses_sizes = []
    for room_p1, room_p2 in rooms_coords:
        for _ in range(max_tables_per_room):
            size_x, size_y = np.random.choice(np.arange(table_size_range[0], table_size_range[1] + 1, 2),
                                              size=2)
            table_x = np.random.randint(room_p1[0] + int(size_x / 2) + table_wall_buffer,
                                        room_p2[0] - int(size_x / 2) - table_wall_buffer)
            table_y = np.random.randint(room_p1[1] + int(size_y / 2) + table_wall_buffer,
                                        room_p2[1] - int(size_y / 2) - table_wall_buffer)
            table_p1 = (table_x - int(size_x / 2), table_y - int(size_y / 2))
            table_p2 = (table_x + int(size_x / 2), table_y + int(size_y / 2))
            table_slice = grid_with_tables[table_p1[0]:table_p2[0],
                                           table_p1[1]:table_p2[1]]
            if not np.any(table_slice == semantic_labels['clutter']):
                grid_with_tables[table_p1[0]:table_p2[0],
                                 table_p1[1]:table_p2[1]] = semantic_labels['clutter']
                table_poses_sizes.append((table_x, table_y, size_x, size_y))

    return grid_with_tables, table_poses_sizes


def determine_intersections(hallway_mask):
    """Returns a dictionary containing intersection points and deadend points of hallways."""
    sk = skeletonize(hallway_mask)
    graph = sknw.build_sknw(sk)
    vertex_data = graph.nodes()
    counter = {id: 0
               for id in vertex_data}
    edges = graph.edges()
    for s, e in edges:
        counter[s] += 1
        counter[e] += 1
    pendant_vertices = [key
                        for key in counter
                        if counter[key] == 1]
    intersection_vertices = list(set(vertex_data) - set(pendant_vertices))
    intersections = np.array([vertex_data[i]['o']
                             for i in intersection_vertices])
    pendants = np.array([vertex_data[i]['o']
                        for i in pendant_vertices])
    return {
        'intersections': intersections,
        'deadends': pendants
    }


def count_loops_in_hallways(hallway_mask):
    grid = 1 - hallway_mask
    _, num_comp = label(grid)
    return num_comp - 1


def swap_room_color(seed):
    # is_passage_red = seed % 2
    # if is_passage_red == 0:
    #     return {'hallway': L_HALL, 'blue': L_ROOM, 'red': L_ROOM2}
    return {'hallway': L_HALL, 'blue': L_ROOM2, 'red': L_ROOM}


def gen_map_office2(random_seed,
                    resolution=RESOLUTION,
                    grid_size=GRID_SIZE,
                    num_of_hallways=NUM_OF_HALLWAYS,
                    boundary_threshold=BOUNDARY_THRESHOLD,
                    min_spacing_hallways=MIN_SPACING_HALLWAYS,
                    hallway_width=HALLWAY_WIDTH,
                    room_width=ROOM_WIDTH,
                    room_length_range=ROOM_LENGTH_RANGE):
    np.random.seed(random_seed)
    loop_count = 2
    counting_loop = 100
    while loop_count > 0:
        grid_with_lines, line_segments = generate_random_lines(
            # seed=random_seed,
            grid_size=grid_size,
            num_of_lines=num_of_hallways,
            spacing_between_lines=min_spacing_hallways,
            boundary_threshold=boundary_threshold)
        grid_with_hallway = inflate_lines_to_create_hallways(
            grid_with_lines, hallway_inflation_scale=hallway_width)
        loop_count = count_loops_in_hallways(grid_with_hallway == semantic_labels['hallway'])

        counting_loop -= 1
        if counting_loop == 0:
            raise ValueError

    # features = determine_intersections(grid_with_hallway == semantic_labels['hallway'])
    # grid_with_special_rooms, special_rooms_coords = add_special_rooms(grid_with_hallway,
    #                                                                   intersections=features['intersections'],
    #                                                                   hallway_inflation_scale=hallway_width,
    #                                                                   room_length_range=room_length_range)
    grid_with_rooms, rooms_coords = add_rooms(grid_with_hallway,
                                              line_segments,
                                              hallway_inflation_scale=hallway_width,
                                              room_b=room_width,
                                              room_l_range=room_length_range)
    # rooms_coords += special_rooms_coords

    occupancy_grid = (grid_with_rooms <= L_CLUTTER).astype(float)
    wall_class_index = swap_room_color(seed=random_seed)
    polys, walls = calc.split_semantic_grid_to_polys(occupancy_grid,
                                                     grid_with_rooms,
                                                     wall_class_index,
                                                     resolution=resolution,
                                                     do_compute_walls=True)

    grid, table_poses_sizes = add_tables(grid_with_rooms, rooms_coords)
    occupancy_grid = (grid <= L_CLUTTER).astype(float)

    return {
        "occ_grid": occupancy_grid.copy(),
        "semantic_grid": grid.copy(),
        "semantic_labels": semantic_labels,
        "polygons": polys,
        "walls": walls,
        "x_offset": 0.0,
        "y_offset": 0.0,
        "resolution": resolution,
        # "features": features,
        "tables": table_poses_sizes,
        "wall_class": wall_class_index
    }


class MapGenOffice2(base_generator.MapGenBase):
    def gen_map(self, random_seed=None):
        self.map_data = gen_map_office2(random_seed,
                                        resolution=self.args.base_resolution,
                                        grid_size=GRID_SIZE,
                                        num_of_hallways=NUM_OF_HALLWAYS,
                                        boundary_threshold=BOUNDARY_THRESHOLD,
                                        min_spacing_hallways=MIN_SPACING_HALLWAYS,
                                        hallway_width=HALLWAY_WIDTH,
                                        room_width=ROOM_WIDTH,
                                        room_length_range=ROOM_LENGTH_RANGE)

        self.hr_grid = self.map_data["occ_grid"].copy()
        self.grid = self.map_data["occ_grid"].copy()

        return self.hr_grid, self.grid, self.map_data

    def get_start_goal_poses(self, min_separation=150, max_separation=1e10, num_attemps=1000):
        inflation_radius = self.args.inflation_radius_m / self.args.base_resolution
        inflated_grid = gridmap.utils.inflate_grid(self.grid, inflation_radius)
        free_cells = np.column_stack(np.where(inflated_grid == gridmap.constants.FREE_VAL))
        for _ in range(num_attemps):
            rand_indices = np.random.choice(np.arange(len(free_cells)), size=2, replace=False)
            start, goal = free_cells[rand_indices]
            cost_grid, _ = gridmap.planning.compute_cost_grid_from_position(inflated_grid, goal)
            path_cost = cost_grid[start[0], start[1]]
            if path_cost >= min_separation and path_cost <= max_separation:
                start = Pose(x=start[0], y=start[1])
                goal = Pose(x=goal[0], y=goal[1])
                return (True, start, goal)
        else:
            raise RuntimeError("Could not find a pair of poses that "
                               "connect during start/goal pose generation.")
