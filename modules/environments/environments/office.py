import numpy as np
import random
import scipy.ndimage

from . import base_generator
from .clutter_generator import add_clutter, add_forest
from .utils import calc


GRID_SIZE = 2000
NUM_HALLWAYS = 10

L_TMP = 100
L_UNSET = -1
L_BKD = 0
L_CLUTTER = 1
L_DOOR = 2
L_HALL = 3
L_ROOM = 4
L_UNK = 5

DEFAULT_PARAMETERS = {
    'base_resolution': 0.1,
    'inflation_radius_m': 0.25,
    'planning_downsample_factor': 1,
}


class Direction:
    def __init__(self, init=None):
        if init is not None:
            if init + 4 < 0:
                raise ValueError("Direction index should not be so far negative.")
            self.index = (init + 4) % 4
        else:
            self.index = random.randint(0, 3)

    def update_direction(self):
        if random.random() >= 0.2:
            return
        else:
            if random.random() >= 0.5:
                self.index = (4 + self.index - 1) % 4
            else:
                self.index = (4 + self.index + 1) % 4

    def get_vec(self, offset=0):
        val = (self.index + offset + 4) % 4
        if val == 0:
            return np.array([1, 0])
        elif val == 1:
            return np.array([0, 1])
        elif val == 2:
            return np.array([-1, 0])
        elif val == 3:
            return np.array([0, -1])
        else:
            raise ValueError("Direction.index should never have this value")


def downsample_grid(hr_grid, downsample_factor):
    grid = scipy.ndimage.maximum_filter(hr_grid, size=downsample_factor)
    grid = grid[::downsample_factor, ::downsample_factor]
    return grid


def initialize_grid():
    return L_UNSET * np.ones([GRID_SIZE, GRID_SIZE])


def fill_region(grid, fill_vec, start_cell, dir_vec, fill_val=L_HALL):
    fmx = int(min(start_cell[0], start_cell[0] + fill_vec[0]))
    fmy = int(min(start_cell[1], start_cell[1] + fill_vec[1]))
    fMx = int(max(start_cell[0], start_cell[0] + fill_vec[0]))
    fMy = int(max(start_cell[1], start_cell[1] + fill_vec[1]))

    # Bounds check
    if fmx < 0 or fmy < 0 or fMx > grid.shape[0] or fmy > grid.shape[1]:
        return False, None

    # Collision check
    if fill_val != L_BKD and grid[fmx:fMx, fmy:fMy].max().max() > L_UNSET:
        return False, None

    # Fill the region
    grid[fmx:fMx, fmy:fMy] = fill_val

    return True, grid


def add_hall(grid, start_cell, direction, length, width):
    fill_vec = length * direction.get_vec(0) + width * direction.get_vec(1)
    did_succeed, grid = fill_region(
        grid, fill_vec, start_cell, direction.get_vec(0), fill_val=L_HALL
    )
    return did_succeed, grid


def create_empty_hallways(grid, resolution, num_hallways=NUM_HALLWAYS):
    # Some initializations
    current_direction = Direction()
    current_cell = np.array([grid.shape[0] / 2, grid.shape[1] / 2])

    # Add a hallway and get a new cell point
    for _ in range(num_hallways):
        hall_length = random.randint(int(8 / resolution), int(16 / resolution))
        hall_width = int(round(3.0 / resolution))
        did_succeed, grid = add_hall(
            grid, current_cell, current_direction, hall_length, hall_width
        )

        # If adding fails, return failure
        if not did_succeed:
            return False, None

        # Update the current cell
        current_cell += hall_length * current_direction.get_vec(0)

        new_direction = Direction(current_direction.index)
        new_direction.update_direction()

        # Update current cell to account for new direction
        if new_direction.index != current_direction.index:
            current_cell += hall_width * current_direction.get_vec(1)

        current_direction = new_direction

    return True, grid


def add_walls(grid, wall_fill_val=L_BKD, space_val=L_HALL):
    """This is done with a simple convolution filter"""
    kernel = np.ones([3, 3])

    free_grid_mask = grid == space_val
    inflated_mask = scipy.ndimage.filters.convolve(
        free_grid_mask, kernel, mode="constant", cval=0
    )

    grid[np.logical_and(inflated_mask > 0, grid == L_UNSET)] = wall_fill_val

    return grid


def create_offices(grid, resolution, num_offices):
    for _ in range(num_offices):
        # Find a random wall point
        idx = np.where(grid == L_BKD)
        idx_idx = random.randint(0, idx[0].size - 1)
        start_x = idx[0][idx_idx]
        start_y = idx[1][idx_idx]

        # Find the direction of the hallway
        if grid[start_x + 1, start_y + 0] == L_HALL:
            hall_direction = Direction(0)
        elif grid[start_x + 0, start_y + 1] == L_HALL:
            hall_direction = Direction(1)
        elif grid[start_x - 1, start_y + 0] == L_HALL:
            hall_direction = Direction(2)
        elif grid[start_x + 0, start_y - 1] == L_HALL:
            hall_direction = Direction(3)
        else:
            # Likely a corner point for the wall
            continue

        # Some parameters
        door_width = int(1.5 / resolution)
        office_width = int(random.uniform(7.0, 9.0) / resolution)
        office_depth = int(random.uniform(5.0, 7.0) / resolution)

        # Check that the door fits
        door_vector = hall_direction.get_vec(1)
        door_fits = True
        for ii in range(-1, door_width + 1):
            if (
                grid[start_x + ii * door_vector[0], start_y + ii * door_vector[1]]
                != L_BKD
            ):
                door_fits = False
        if not door_fits:
            continue

        # Try to fill the office
        office_offset = random.randint(1, office_width - door_width - 1)
        fill_vec = office_width * hall_direction.get_vec(
            1
        ) + office_depth * hall_direction.get_vec(2)
        start_cell = np.array([start_x, start_y])
        start_cell -= office_offset * hall_direction.get_vec(1)
        # Correction for 'lopsided' function
        if hall_direction.get_vec(2).max() > 0:
            start_cell += hall_direction.get_vec(2)

        did_succeed, new_grid = fill_region(
            grid, fill_vec, start_cell, hall_direction, fill_val=L_ROOM
        )

        if not did_succeed:
            continue
        else:
            grid = new_grid

        # Create the door
        for ii in range(door_width):
            grid[start_x + ii * door_vector[0], start_y + ii * door_vector[1]] = L_DOOR

        # Put walls around the office (temporary label value)
        grid = add_walls(grid, wall_fill_val=L_TMP, space_val=L_ROOM)

    # Replace the temporary value with background/wall
    grid[grid == L_TMP] = L_BKD

    return grid


def gen_map_office(resolution, random_seed=None):
    # Initialize the random generator
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create the empty hallways
    while True:
        grid = initialize_grid()
        did_succeed, grid = create_empty_hallways(grid, resolution, NUM_HALLWAYS)
        if did_succeed:
            break

    grid = add_walls(grid, wall_fill_val=L_BKD)
    grid = create_offices(grid, resolution, num_offices=400)

    # Trim the edges of the grid
    col_max = np.amax(grid, axis=0)
    row_max = np.amax(grid, axis=1)

    # Create clutter
    grid = add_clutter(grid, resolution_m=resolution)
    grid = add_clutter(
        grid,
        label=L_HALL,
        resolution_m=resolution,
        do_insert_central_clutter=False,
        num_wall_clutter=30,
    )

    grid = grid[row_max > 0, :]
    grid = grid[:, col_max > 0]

    # Pad the edges
    grid = np.pad(grid, pad_width=2, mode="constant", constant_values=L_UNSET)

    semantic_labels = {
        'background': L_BKD,
        'clutter': L_CLUTTER,
        'door': L_DOOR,
        'hallway': L_HALL,
        'room': L_ROOM,
        'other': L_UNK,
    }

    wall_class_index = {
        'hallway': L_HALL,
        'room': L_ROOM,
    }

    occupancy_grid = (grid <= L_CLUTTER).astype(float)

    polys, walls = calc.split_semantic_grid_to_polys(occupancy_grid,
                                                     grid,
                                                     wall_class_index,
                                                     resolution,
                                                     do_compute_walls=True)

    return {
        "occ_grid": (grid <= L_CLUTTER).astype(float),
        "semantic_grid": grid.copy(),
        "semantic_labels": semantic_labels,
        "polygons": polys,
        "walls": walls,
        "x_offset": 0.0,
        "y_offset": 0.0,
    }


def gen_map_ring(args, ring_height_m=50.0, ring_width_m=50.0, random_seed=None):
    # Initialize the random generator
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create the empty ring
    grid = initialize_grid()
    sx = grid.shape[0] // 2
    sy = grid.shape[1] // 2
    resolution = args.base_resolution
    hall_width = int(round(3.0 / resolution))
    ring_width = int(round(ring_width_m / resolution))
    ring_height = int(round(ring_height_m / resolution))
    grid[
        sx - hall_width : sx + ring_width + hall_width,
        sx - hall_width : sx + ring_height + hall_width,
    ] = L_HALL
    grid[sx : sx + ring_width, sx : sx + ring_height] = L_UNSET

    # Add walls
    grid = add_walls(grid, wall_fill_val=L_BKD)

    # Now, add the center portion and put a temporary wall around it
    vest_width = int(round(8 / resolution))
    vest_height = int(round(5 / resolution))
    connect_width = int(round(2.0 / resolution))
    cx = sx + ring_width // 2
    grid[cx - vest_width // 2 : cx + vest_width // 2, sy : sy + vest_height] = L_HALL
    grid[
        cx - vest_width // 2 : cx + vest_width // 2,
        sy + ring_height - vest_height : sy + ring_height,
    ] = L_HALL
    grid[
        cx - connect_width // 2 : cx + connect_width // 2, sy : sy + ring_height
    ] = L_HALL

    # With a random prob, block the center of the path
    do_block = random.random() >= args.map_ring_connection_prob
    if do_block is True:
        grid[
            cx - connect_width / 2 : cx + connect_width / 2,
            sy + ring_height / 2 - 2 : sy + ring_height / 2 + 2,
        ] = L_UNSET

    grid = add_walls(grid, wall_fill_val=L_TMP)

    grid = create_offices(grid, args.base_resolution, num_offices=400)
    grid[grid == L_TMP] = L_BKD

    # Create clutter
    grid = add_clutter(
        grid, resolution_m=args.base_resolution / args.planning_downsample_factor
    )
    grid = add_clutter(
        grid,
        label=L_HALL,
        resolution_m=args.base_resolution / args.planning_downsample_factor,
        do_insert_central_clutter=False,
        num_wall_clutter=30,
    )

    # Trim the edges of the grid
    col_max = np.amax(grid, axis=0)
    row_max = np.amax(grid, axis=1)
    grid = grid[row_max > 0, :]
    grid = grid[:, col_max > 0]

    # Pad the edges
    grid = np.pad(grid, pad_width=2, mode="constant", constant_values=L_UNSET)

    # Create the start goal pose generators (each should generate a PoseT
    # NamedTuple that places the pose in a room near one end of the connection
    # or the other).
    label_grid = grid.copy()

    grid = (grid <= L_CLUTTER).astype(float)
    return grid, downsample_grid(grid, args.planning_downsample_factor), label_grid


def gen_map_lattice(args, ring_height_m=50.0, ring_width_m=50.0, random_seed=None):
    # Initialize the random generator
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create the empty ring
    grid = initialize_grid()
    sx = grid.shape[0] / 2
    sy = grid.shape[1] / 2
    resolution = args.base_resolution
    hall_width = int(round(3.0 / resolution))
    wider_hall_width = int(round(5.0 / resolution))
    ring_width = int(round(ring_width_m / resolution))
    ring_height = int(round(ring_height_m / resolution))

    # Add the 2 main hallways
    grid[
        sx - hall_width : sx + ring_width + hall_width,
        sx - hall_width : sx + ring_height + hall_width,
    ] = L_HALL
    grid[
        sx - hall_width : sx + ring_width + hall_width, sx : sx + ring_height
    ] = L_UNSET

    # Add walls
    grid = add_walls(grid, wall_fill_val=L_BKD)

    # Add a few connecting hallways
    nh = 5

    ri = int(random.random() * nh)
    dm = ring_width * 1.0 / (nh - 1)
    for ii in range(nh):
        cx = int(sx + dm * ii + 0.0 * dm)
        if ii == ri:
            grid[
                cx - wider_hall_width / 2 : cx + wider_hall_width / 2,
                sy : sy + ring_height,
            ] = L_HALL
        else:
            grid[
                cx - hall_width / 2 : cx + hall_width / 2, sy : sy + ring_height
            ] = L_HALL
            grid[
                cx - hall_width / 2 : cx + hall_width / 2,
                sy + ring_height / 2 - 2 : sy + ring_height / 2 + 2,
            ] = L_UNSET

    grid = add_walls(grid, wall_fill_val=L_TMP)

    # Trim the edges of the grid
    col_max = np.amax(grid, axis=0)
    row_max = np.amax(grid, axis=1)
    grid = grid[row_max > 0, :]
    grid = grid[:, col_max > 0]

    # Pad the edges
    grid = np.pad(grid, pad_width=2, mode="constant", constant_values=L_UNSET)

    # Create the start goal pose generators (each should generate a PoseT
    # NamedTuple that places the pose in a room near one end of the connection
    # or the other).
    label_grid = grid.copy()

    grid = (grid <= L_CLUTTER).astype(float)
    return grid, downsample_grid(grid, args.planning_downsample_factor), label_grid


def gen_map_forest(args, ring_height_m=50.0, ring_width_m=50.0, random_seed=None):
    # Initialize the random generator
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create the empty ring
    grid = initialize_grid()
    sx = grid.shape[0] / 2
    resolution = args.base_resolution
    ring_width = int(round(ring_width_m / resolution))
    ring_height = int(round(ring_height_m / resolution))

    # Add some pillars
    mask = np.zeros(grid.shape)
    mask[sx : sx + ring_width, sx : sx + ring_height] = 2
    grid[mask > 1] = L_ROOM
    grid = add_forest(
        grid,
        mask > 1,
        label=L_HALL,
        resolution_m=args.base_resolution / args.planning_downsample_factor,
    )

    # Trim the edges of the grid
    col_max = np.amax(grid, axis=0)
    row_max = np.amax(grid, axis=1)
    grid = grid[row_max > 0, :]
    grid = grid[:, col_max > 0]

    # Pad the edges
    grid = np.pad(grid, pad_width=20, mode="constant", constant_values=L_UNSET)

    # Create the start goal pose generators (each should generate a PoseT
    # NamedTuple that places the pose in a room near one end of the connection
    # or the other).
    label_grid = grid.copy()

    grid = (grid <= L_CLUTTER).astype(float)
    return grid, downsample_grid(grid, args.planning_downsample_factor), label_grid


class MapGenOffice(base_generator.MapGenBase):
    def gen_map(self, random_seed=None):
        map_data = gen_map_office(
            resolution=self.args.base_resolution, random_seed=random_seed
        )
        map_data["resolution"] = self.args.base_resolution

        self.hr_grid = map_data["occ_grid"].copy()
        self.grid = map_data["occ_grid"].copy()

        return self.hr_grid, self.grid, map_data
