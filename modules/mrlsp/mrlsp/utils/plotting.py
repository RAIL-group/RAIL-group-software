import lsp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

colors = {
    '0': 'blue',
    '1': 'red',
    '2': 'cyan',
    '3': 'magenta',
    '4': 'yellow',
    '5': 'black',
    '6': 'white',
    '7': 'green'
}

style = {
    '0': 'b:',
    '1': 'r:',
    '2': 'c:',
    '3': 'm:',
    '4': 'y:',
    '5': 'k:',
    '6': 'w:',
    '7': 'g:'
}


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


def plot_grid_with_frontiers(ax,
                             grid_map,
                             known_map,
                             frontiers,
                             cmap_name='viridis'):
    grid = lsp.utils.plotting.make_plotting_grid(grid_map, known_map)

    cmap = plt.get_cmap(cmap_name)
    for frontier in frontiers:
        color = cmap(frontier.prob_feasible)
        grid[frontier.points[0, :], frontier.points[1, :], 0] = color[0]
        grid[frontier.points[0, :], frontier.points[1, :], 1] = color[1]
        grid[frontier.points[0, :], frontier.points[1, :], 2] = color[2]

    ax.imshow(np.transpose(grid, (1, 0, 2)))


def plot_mrlsp(args,
               timestamp,
               num_robots,
               robots,
               goal_pose,
               subgoals,
               pano_image,
               robot_grid,
               known_map=None,
               paths=None,
               cols=1):

    # create a main figure
    if num_robots == 1:
        plt.ion()
        plt.figure(1, figsize=(6, 6))
        plt.clf()
        if pano_image[0] is not None:
            ax = plt.subplot(211)
            plt.imshow(pano_image[0])
            ax = plt.subplot(212)
        else:
            ax = plt.subplot(111)
        plot_pose(ax, robots[0].pose, color='blue')
        plot_grid_with_frontiers(ax, robot_grid, known_map,
                                 subgoals)
        plot_pose(ax, goal_pose[0], color='green', filled=False)
        if paths[0] != []:
            plot_path(ax, paths[0])
        plot_pose_path(ax, robots[0].all_poses)
        plt.show()
        plt.pause(0.01)

    else:
        image_file = Path(args.save_dir) / \
            f'img_{args.map_type}_{args.num_robots}robots_{args.current_seed}_{timestamp}.png'
        num_image = num_robots
        plt.ion()
        fig = plt.figure(num=1, figsize=(12, num_image * 2))
        gs = fig.add_gridspec(num_image, 2)
        plt.clf()
        # plot pano images for each robot
        for k in range((num_image)):
            ax = fig.add_subplot(gs[k, 0])  # noqa: F841
            plt.imshow(pano_image[k])
            plt.title("robot" + str(k + 1), color=colors[str(k)])

        ax1 = fig.add_subplot(gs[:num_image - 1, 1])

        plot_grid_with_frontiers(ax1, robot_grid, known_map,
                                 subgoals)
        for k in range(num_robots):
            plt.text(robots[k].pose.x,
                     robots[k].pose.y,
                     k + 1,
                     color=colors[str(k)])
            plot_pose(ax1, robots[k].pose, color=colors[str(k)])
            plot_pose(ax1, goal_pose[k], color=colors[str(k)], filled=False)
            # print(paths[k])
            if paths[k] != []:
                plot_path(ax1, paths[k], style=style[str(k)])
            plot_pose_path(ax1, robots[k].all_poses, style[str(k)][:-1])
        # plt.savefig(image_file)
        plt.show()
        plt.pause(0.1)


def plot_final_figure(args,
                      num_robots,
                      robots,
                      goal_pose,
                      subgoals,
                      robot_grid,
                      known_map=None,
                      paths=None,
                      planner=None):
    image_file = Path(args.save_dir) / \
        f'planner_{planner}_{args.map_type}_cost_{args.current_seed}_r{num_robots}.png'
    plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(111)
    plot_grid_with_frontiers(ax1, robot_grid, known_map,
                             subgoals)
    for k in range(num_robots):
        plt.text(robots[k].pose.x,
                 robots[k].pose.y,
                 k + 1,
                 color=colors[str(k)])
        plot_pose(ax1, robots[k].pose, color=colors[str(k)])
        plot_pose(ax1, goal_pose[k], color=colors[str(k)], filled=False)
        if paths[k] != []:
            plot_path(ax1, paths[k], style=style[str(k)])
        plot_pose_path(ax1, robots[k].all_poses, style[str(k)][:-1])
    plt.savefig(image_file)
