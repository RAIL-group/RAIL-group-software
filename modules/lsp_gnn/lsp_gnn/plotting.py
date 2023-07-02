import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion

import lsp
from lsp.constants import (COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL,
                           OBSTACLE_THRESHOLD)

FOOT_PRINT = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]


def plot_pose(ax, pose, color='black', filled=True):
    if filled:
        ax.scatter(pose.x, pose.y, color=color, s=10, label='point')
    else:
        ax.scatter(pose.x,
                   pose.y,
                   color=color,
                   s=10,
                   label='point',
                   facecolors='none')


def plot_path(ax, path, style='m:'):
    if path is not None and len(path) == 2:
        ax.plot(path[0, :], path[1, :], style)


def plot_pose_path(ax, poses, style='m'):
    path = np.array([[p.x, p.y] for p in poses]).T
    plot_path(ax, path, style)


def plot_grid(ax,
              grid_map,
              known_map,
              frontiers=None,
              cmap_name='viridis'):
    grid = lsp.utils.plotting.make_plotting_grid(grid_map, known_map)
    if frontiers is not None:
        cmap = plt.get_cmap(cmap_name)
        for frontier in frontiers:
            color = cmap(frontier.prob_feasible)
            grid[frontier.points[0, :], frontier.points[1, :], 0] = color[0]
            grid[frontier.points[0, :], frontier.points[1, :], 1] = color[1]
            grid[frontier.points[0, :], frontier.points[1, :], 2] = color[2]

    ax.imshow(np.transpose(grid, (1, 0, 2)))


def plot_semantic_grid_with_frontiers(
    ax, grid_map, known_map, frontiers,
        semantic_grid=None, semantic_labels=None, cmap_name='viridis'):
    grid = make_plotting_grid(
        grid_map, known_map, semantic_grid, semantic_labels)
    cmap = plt.get_cmap(cmap_name)
    for frontier in frontiers:
        color = cmap(frontier.prob_feasible)
        grid[frontier.points[0, :], frontier.points[1, :], 0] = color[0]
        grid[frontier.points[0, :], frontier.points[1, :], 1] = color[1]
        grid[frontier.points[0, :], frontier.points[1, :], 2] = color[2]
    ax.imshow(np.transpose(grid, (1, 0, 2)))


def make_plotting_grid(grid_map, known_map=None,
                       semantic_grid=None, semantic_labels=None):
    # print(semantic_labels)
    # if known_map is not None:
    #     grid_map = known_map
    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    collision = grid_map >= OBSTACLE_THRESHOLD
    # Take one pixel boundary of the region collision
    thinned = erosion(collision, footprint=FOOT_PRINT)
    boundary = np.logical_xor(collision, thinned)
    free = np.logical_and(grid_map < OBSTACLE_THRESHOLD, grid_map >= FREE_VAL)
    red = np.logical_and(free, semantic_grid == semantic_labels['red'])
    grid[:, :, 0][red] = 1
    grid[:, :, 1][red] = 0
    grid[:, :, 2][red] = 0
    blue = np.logical_and(free, semantic_grid == semantic_labels['blue'])
    grid[:, :, 0][blue] = 0
    grid[:, :, 1][blue] = 0
    grid[:, :, 2][blue] = 1
    hall = np.logical_and(free, semantic_grid == semantic_labels['hallway'])
    grid[:, :, 0][hall] = 1
    grid[:, :, 1][hall] = 1
    grid[:, :, 2][hall] = 1
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    if known_map is not None:
        known_collision = known_map == COLLISION_VAL
        unobserved = grid_map == UNOBSERVED_VAL
        unknown_obstacle = np.logical_and(known_collision, unobserved)
        grid[:, :, 0][unknown_obstacle] = 0.65
        grid[:, :, 1][unknown_obstacle] = 0.65
        grid[:, :, 2][unknown_obstacle] = 0.75

    return grid
