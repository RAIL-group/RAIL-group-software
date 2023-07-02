import numpy as np
import matplotlib.pyplot as plt
import scipy

from lsp.constants import (COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL,
                           OBSTACLE_THRESHOLD)


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


def plot_path(ax, path, style='b:'):
    if path is not None:
        ax.plot(path[0, :], path[1, :], style)


def plot_pose_path(ax, poses, style='b'):
    path = np.array([[p.x, p.y] for p in poses]).T
    plot_path(ax, path, style)


def make_plotting_grid(grid_map, known_map=None):
    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    collision = grid_map >= OBSTACLE_THRESHOLD
    free = np.logical_and(grid_map < OBSTACLE_THRESHOLD, grid_map >= FREE_VAL)
    grid[:, :, 0][free] = 1
    grid[:, :, 0][collision] = 0
    grid[:, :, 1][free] = 1
    grid[:, :, 1][collision] = 0
    grid[:, :, 2][free] = 1
    grid[:, :, 2][collision] = 0

    if known_map is not None:
        known_collision = known_map == COLLISION_VAL
        unobserved = grid_map == UNOBSERVED_VAL
        unknown_obstacle = np.logical_and(known_collision, unobserved)
        grid[:, :, 0][unknown_obstacle] = 0.65
        grid[:, :, 1][unknown_obstacle] = 0.65
        grid[:, :, 2][unknown_obstacle] = 0.75

    return grid


def plot_grid_with_frontiers(ax,
                             grid_map,
                             known_map,
                             frontiers,
                             cmap_name='viridis'):
    grid = make_plotting_grid(grid_map, known_map)

    cmap = plt.get_cmap(cmap_name)
    for frontier in frontiers:
        color = cmap(frontier.prob_feasible)
        grid[frontier.points[0, :], frontier.points[1, :], 0] = color[0]
        grid[frontier.points[0, :], frontier.points[1, :], 1] = color[1]
        grid[frontier.points[0, :], frontier.points[1, :], 2] = color[2]

    ax.imshow(np.transpose(grid, (1, 0, 2)))


def make_known_grid(known_map):
    kernel = np.ones((3, 3)) / 9
    grid = scipy.ndimage.convolve(known_map, kernel)
    walls = (known_map == 1) & (grid < 1)
    grid_map = known_map.copy()
    grid_map += 1
    grid_map[known_map == 0] = 0
    grid_map[known_map == 1] = 1
    grid_map[walls] = 2

    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    grid[:, :, 0][grid_map == 0] = 1
    grid[:, :, 0][grid_map == 2] = 0
    grid[:, :, 1][grid_map == 0] = 1
    grid[:, :, 1][grid_map == 2] = 0
    grid[:, :, 2][grid_map == 0] = 1
    grid[:, :, 2][grid_map == 2] = 0

    grid[:, :, 0][grid_map == 1] = 0.65
    grid[:, :, 1][grid_map == 1] = 0.65
    grid[:, :, 2][grid_map == 1] = 0.75

    return grid
