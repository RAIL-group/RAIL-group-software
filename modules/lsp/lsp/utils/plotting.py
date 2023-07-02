"""
This file contains various plotting functions.
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import scipy.stats
from skimage.measure import LineModelND, ransac
from ..constants import (COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL,
                         OBSTACLE_THRESHOLD)


def plot_grid(ax, grid, cost, path, robot_pose, goal_pose):
    ax.clear()
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    ax.scatter(goal_pose[1] - 0.5,
               goal_pose[0] - 0.5,
               color='green',
               s=10,
               label='point')
    ax.scatter(robot_pose[1] - 0.5,
               robot_pose[0] - 0.5,
               color='blue',
               s=10,
               label='point')
    if path is not None:
        ax.set_title("Net Cost: %f" % cost)
        ax.imshow(grid, origin='lower')

    if path is not None:
        ax.plot(path[1, :] - 0.5, path[0, :] - 0.5)


def _convert_path_to_map_frame(path, map_data):
    return np.stack((
        map_data['resolution'] * path[0, :] + map_data['x_offset'],
        map_data['resolution'] * path[1, :] + map_data['y_offset'],
    ))


def _colored_map_grid(grid_map, known_map=None):
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


def plot_grid_to_scale(
        ax,
        grid,
        map_data=None,
        cmap=None):

    grid_shape = grid.shape
    if map_data is not None:
        xlimits = [
            map_data['x_offset'],
            map_data['x_offset'] + map_data['resolution'] * grid_shape[0]]
        ylimits = [
            map_data['y_offset'],
            map_data['y_offset'] + map_data['resolution'] * grid_shape[1]]
        extent = ylimits + xlimits
    else:
        extent = [0, grid_shape[1], 0, grid_shape[0]]

    ax.imshow(grid, extent=extent, origin='lower', cmap=cmap)


def plot_navigation_data(
        ax,
        observed_map_grid=None,
        known_map_grid=None,
        robot_pose=None,
        goal_pose=None,
        map_data=None,
        robot_poses=None,
        robot_poses_m=None,
        planned_path=None,
        planned_path_m=None):

    # Plot the base grid
    color_map = _colored_map_grid(observed_map_grid, known_map_grid)
    plot_grid_to_scale(ax, color_map, map_data)

    # Plot the points
    if robot_pose is not None:
        ax.scatter(robot_pose.y, robot_pose.x,
                   color='blue', s=10, label='point')
    if goal_pose is not None:
        ax.scatter(goal_pose.y, goal_pose.x,
                   color='green', s=10, label='point')

    # Plot the paths
    if robot_poses is not None and map_data is not None:
        rxs = [map_data['x_offset'] + map_data['resolution'] * p.x for p in robot_poses]
        rys = [map_data['y_offset'] + map_data['resolution'] * p.y for p in robot_poses]
        ax.plot(rys, rxs, 'b')
    elif robot_poses is not None:
        rxs = [p.x for p in robot_poses]
        rys = [p.y for p in robot_poses]
        ax.plot(rys, rxs, 'b')
    elif robot_poses_m is not None:
        rxs = [p.x for p in robot_poses_m]
        rys = [p.y for p in robot_poses_m]
        ax.plot(rys, rxs, 'b')

    if planned_path is not None:
        planned_path = np.array(planned_path)
        if map_data is not None:
            planned_path_m = _convert_path_to_map_frame(planned_path, map_data)
        else:
            ax.plot(planned_path[1, :], planned_path[0, :], 'b:')
    if planned_path_m is not None:
        planned_path_m = np.array(planned_path_m)
        ax.plot(planned_path_m[1, :], planned_path_m[0, :], 'b:')


def plot_results_data(data_file, ax=None):
    data = np.loadtxt(data_file)

    ele = [v for v in range(1000, 1424) if v not in data[:, 0]]
    print(f"Missing elements: {ele}")

    # Use only the success cases
    data = data[data[:, 1] == 1]

    # Compute some helper plot bounds
    limit_f = max(data[:, 8])
    limit_n = max(data[:, 7])
    limit = max(limit_f, limit_n) + 100

    # Compute the RANSAC fit
    to_fit_data = np.column_stack([data[:, 7], data[:, 8]])
    model_robust, inliers = ransac(to_fit_data,
                                   LineModelND,
                                   min_samples=min(800, data.shape[0] - 1),
                                   residual_threshold=200,
                                   max_trials=1000)
    line_y = model_robust.predict_y([0, limit + 1000])
    slope = (line_y[1] - line_y[0]) / (limit + 1000 - 0)

    if ax is None:
        fig = plt.figure(figsize=(2, 4), dpi=300)
        ax = fig.add_subplot(1, 1, 1)

    p_bad = matplotlib.collections.PatchCollection([
        matplotlib.patches.Rectangle([0, 0], 9000, 9000, alpha=0.2, angle=45)
    ])
    p_bad.set_color([1.0, 0.9, 0.9])
    ax.add_collection(p_bad)
    p_good = matplotlib.collections.PatchCollection([
        matplotlib.patches.Rectangle([0, 0], 9000, -9000, alpha=0.2, angle=45)
    ])
    p_good.set_color([0.9, 1.0, 0.9])
    ax.add_collection(p_good)

    xy = np.vstack([data[:, 7], data[:, 8]])
    z = scipy.stats.gaussian_kde(xy)(xy)
    colors = cm.get_cmap('viridis')((z - z.min()) / (z.max() - z.min()))
    outliers = np.logical_not(inliers)
    good_outliers = np.logical_and(outliers, slope * data[:, 7] > data[:, 8])
    bad_outliers = np.logical_and(outliers, slope * data[:, 7] < data[:, 8])
    plt.scatter(data[outliers, 7],
                data[outliers, 8],
                s=10,
                facecolors='none',
                edgecolors=colors[outliers])
    plt.scatter(data[inliers, 7], data[inliers, 8], c=z[inliers], s=10)
    # Print out some stats
    num_outliers = np.sum(outliers.astype(float)) / outliers.shape[0]
    num_good_outliers = np.sum(
        good_outliers.astype(float)) / good_outliers.shape[0]
    num_bad_outliers = np.sum(
        bad_outliers.astype(float)) / bad_outliers.shape[0]
    f_dist = sum(data[:, 8])
    n_dist = sum(data[:, 7])

    # Center line (boundary between good/bad)
    plt.plot([0, 1.1 * limit], [0, 1.1 * limit], 'k--')

    # # RANSAC fit line
    # plt.plot([0, limit + 1000], line_y)

    # Format the plot
    ax.set_xlabel("Dijkstra Heuristic Net Distance (Baseline)")
    ax.set_ylabel("Frontier Planning Net Distance (Ours)")
    ax.axis([0, limit, 0, limit])
    plt.colorbar(ticks=[],
                 label="Low Data Density            High Data Density")
    plt.savefig(os.path.join(args.data_base_dir, args.results_base_name) +
                '_fig.png',
                dpi=300)

    # Write to data file
    fname = os.path.join(args.data_base_dir,
                         args.results_base_name) + '_summary.txt'
    with open(fname, 'w') as f:
        f.write(f"Num Points: {outliers.shape[0]}\n")
        f.write(f"% Outliers: {num_outliers}\n")
        f.write(f"% Bad Outliers: {num_bad_outliers}\n")
        f.write(f"% Good Outliers: {num_good_outliers}\n")
        f.write(
            f"Net Dist: (Frontier {f_dist:.1f})/(Naive {n_dist:.1f}) = {f_dist/n_dist}\n"
        )
        f.write(f"Slope of fit line: {slope}\n")


np.seterr(divide='ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot a grid file')
    parser.add_argument('--zip_file',
                        type=str,
                        help='File with grid and path data.',
                        required=False,
                        default=None)
    parser.add_argument('--data_file',
                        type=str,
                        help='File with grid and path data.',
                        required=False,
                        default=None)
    parser.add_argument('--frontier_pred_files',
                        type=str,
                        help='File with grid and path data.',
                        nargs='+',
                        required=False,
                        default=None)
    parser.add_argument('--data_base_dir',
                        type=str,
                        help='Data directory (results in /data/results)',
                        default='/data/')
    parser.add_argument('--results_base_name',
                        type=str,
                        help='Filename where figure will be written.',
                        required=True)
    args = parser.parse_args()

    if args.data_file is not None:
        plot_results_data(args.data_file)
    else:
        data = np.load(args.zip_file)
        grid = data['grid']
        path = data['path']
        cost = data['cost']
        robot_pose = data['robot_pose']
        goal_pose = data['goal_pose']

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot_grid(ax, grid, cost, path, robot_pose, goal_pose)
        plt.show()


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
