# from .env_generation_base import *
import cv2
import math
import numpy as np
import random
import scipy.ndimage
from gridmap.utils import inflate_grid

L_TMP = 100
L_UNSET = -1
L_BKD = 0
L_CLUTTER = 1
L_DOOR = 2
L_HALL = 3
L_ROOM = 4
L_UNK = 5


def get_room_orientation(labels, room_label):
    is_room = labels == room_label
    nonzero_x = np.where(np.amax(is_room, axis=1))
    nonzero_y = np.where(np.amax(is_room, axis=0))

    xm = max(nonzero_x[0][0] - 1, 0)
    xM = min(nonzero_x[0][-1] + 2, labels.shape[0])
    ym = max(nonzero_y[0][0] - 1, 0)
    yM = min(nonzero_y[0][-1] + 2, labels.shape[1])
    sub_is_room = is_room[xm:xM, ym:yM]

    try:
        from MinimumBoundingBox import MinimumBoundingBox

        coords = np.where(sub_is_room)
        coords = tuple(zip(*coords))
        bounding_rect = MinimumBoundingBox(coords)
        angle = (bounding_rect.unit_vector_angle + math.pi / 8) % (
            math.pi / 4
        ) - math.pi / 8
        return (angle, 1)
    except:  # noqa: E722
        return (0.0, 0)


def add_clutter_to_room(
    labels,
    grid,
    room_label,
    inflation_radius_m,
    resolution_m,
    do_insert_central_clutter,
    num_wall_clutter,
    rmat,
):
    is_room = labels == room_label
    nonzero_x = np.where(np.amax(is_room, axis=1))
    nonzero_y = np.where(np.amax(is_room, axis=0))

    xm = max(nonzero_x[0][0] - 1, 0)
    xM = min(nonzero_x[0][-1] + 2, labels.shape[0])
    ym = max(nonzero_y[0][0] - 1, 0)
    yM = min(nonzero_y[0][-1] + 2, labels.shape[1])

    sub_grid_labels = labels[xm:xM, ym:yM]
    sub_grid = grid[xm:xM, ym:yM]

    inflation_radius = inflation_radius_m / resolution_m
    inflated_door_sub_grid = inflate_grid(sub_grid, inflation_radius, L_DOOR)

    # Inflate doors
    sub_grid_not_near_door = np.logical_not(inflated_door_sub_grid)

    # Try to add a bit of furnature
    fw = int(round(1.5 / resolution_m))
    fh = int(round(1.0 / resolution_m))
    counter = 0

    # Add furnature anywhere in the room
    while counter < 300 and do_insert_central_clutter:
        counter += 1
        trial = np.zeros([xM - xm, yM - ym])
        xr = random.randint(0, xM - xm)
        yr = random.randint(0, yM - ym)

        pts = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]])
        pts = np.matmul(rmat, pts.T).T + np.array([[xr, yr]])
        pts = np.round(pts).astype(int)
        cv2.fillPoly(trial, [pts], 1)

        try:
            is_touching_room = np.logical_and(
                trial == 1, sub_grid_labels == room_label
            ).any()
            is_outside_room = np.logical_and(
                trial == 1, sub_grid_labels != room_label
            ).any()
            is_near_door = np.logical_and(trial == 1, sub_grid_not_near_door == 0).any()
            if is_touching_room and not is_outside_room and not is_near_door:
                # Success! Add the furnature
                labels[xm:xM, ym:yM][trial == 1] = 0
                grid[xm:xM, ym:yM][trial == 1] = L_CLUTTER
                break
        except:  # noqa: E722
            pass

    # Now add some clutter near the walls
    clutter_counter = 0
    while counter < 25 * num_wall_clutter:
        counter += 1
        trial = np.zeros([xM - xm, yM - ym])

        fw = int(round(0.35 / resolution_m))
        fh = int(round(0.35 / resolution_m))

        xr = random.randint(0, xM - xm)
        yr = random.randint(0, yM - ym)

        pts = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]])
        pts = np.matmul(rmat, pts.T).T + np.array([[xr, yr]])
        pts = np.round(pts).astype(int)
        cv2.fillPoly(trial, [pts], 1)

        # If all of the points are not near a door and at least one of the points is in a wall
        is_near_door = np.logical_and(trial == 1, sub_grid_not_near_door == 0).any()
        is_touching_room = np.logical_and(
            trial == 1, sub_grid_labels == room_label
        ).any()
        is_touching_wall = np.logical_and(trial == 1, sub_grid_labels == 0).any()
        is_in_another_room = np.logical_and(
            trial == 1,
            np.logical_not(
                np.logical_or(sub_grid == L_BKD, sub_grid_labels == room_label)
            ),
        ).any()
        if (
            not is_near_door
            and is_touching_wall
            and not is_in_another_room
            and is_touching_room
        ):
            # Success! Add the furnature
            grid[xm:xM, ym:yM][trial == 1] = L_CLUTTER
            clutter_counter += 1
            if clutter_counter >= num_wall_clutter:
                break


def add_forest(
    grid, mask, label=L_CLUTTER, resolution_m=0.04, radius_m=1.4, num_pillars=250
):

    inflation_radius_m = 1.5
    nonzero_x = np.where(np.amax(mask, axis=1))
    nonzero_y = np.where(np.amax(mask, axis=0))

    xm = max(nonzero_x[0][0] - 1, 0)
    xM = min(nonzero_x[0][-1] + 2, grid.shape[0])
    ym = max(nonzero_y[0][0] - 1, 0)
    yM = min(nonzero_y[0][-1] + 2, grid.shape[1])

    sub_mask = mask[xm:xM, ym:yM]

    inflation_radius = inflation_radius_m / resolution_m

    # Generate pillar circle
    rad = radius_m / resolution_m
    kernel_size = int(1 + 2 * math.ceil(rad))
    cind = int(math.ceil(rad))
    y, x = np.ogrid[-cind : kernel_size - cind, -cind : kernel_size - cind]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[y * y + x * x <= rad * rad] = 1

    counter = 0
    trial = np.zeros([xM - xm, yM - ym])
    while counter < num_pillars:
        counter += 1
        mid_trial = np.zeros(trial.shape).astype(bool)
        xr = random.randint(0, xM - xm - kernel_size)
        yr = random.randint(0, yM - ym - kernel_size)
        mid_trial[xr : xr + kernel_size, yr : yr + kernel_size] = kernel

        is_inside_room = not np.logical_and(mid_trial, np.logical_not(sub_mask)).any()
        inflated_trial = inflate_grid(trial, inflation_radius, 1)
        is_overlapping = np.logical_and(mid_trial, inflated_trial).any()
        if is_inside_room and not is_overlapping:
            trial = np.logical_or(mid_trial, trial)

    grid[xm:xM, ym:yM][trial == 1] = L_CLUTTER
    return grid


def add_clutter(
    grid,
    label=L_ROOM,
    resolution_m=0.04,
    do_insert_central_clutter=True,
    num_wall_clutter=6,
    rot_deg=None,
):
    # Parameter definitions
    inflation_radius_m = 1.0

    # Get rooms after masking by doors
    room_grid = grid == label
    # Group the rooms into connected components
    labels, nb = scipy.ndimage.label(room_grid)

    if rot_deg is None:
        angle = 0.0
        angle_count = 0.00001
        # Get the average angle
        for ii in range(1, nb + 1):
            a, c = get_room_orientation(labels, room_label=ii)
            angle += a
            angle_count += c

        avg_angle = angle / angle_count
        sr = math.sin(avg_angle)
        cr = math.cos(avg_angle)
        rmat = np.array([[cr, sr], [-sr, cr]])
    else:
        avg_angle = (rot_deg - 24.5) * math.pi / 180.0
        sr = math.sin(avg_angle)
        cr = math.cos(avg_angle)
        rmat = np.array([[cr, sr], [-sr, cr]])
        rmat = np.array([[cr, sr], [-sr, cr]])

    # Loop through the rooms and add a bit of clutter to each
    for ii in range(1, nb + 1):
        add_clutter_to_room(
            labels,
            grid,
            room_label=ii,
            inflation_radius_m=inflation_radius_m,
            resolution_m=resolution_m,
            do_insert_central_clutter=do_insert_central_clutter,
            num_wall_clutter=num_wall_clutter,
            rmat=rmat,
        )

    return grid
