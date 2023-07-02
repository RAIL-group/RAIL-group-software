import gridmap
from gridmap.constants import COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL
import numpy as np
import matplotlib.cm


def _get_linewidth_from_cpd(cpd):
    return max(0.5, 2 + np.log10(float(cpd)))


def plot_map(ax, planner, robot_poses=None):
    """Plots the partial map from the planner, including the robot pose and
    goal. The optional argument 'robot_poses' should be a list of all robot
    poses, used to plot the trajectory the robot has followed so far."""
    partial_grid = planner.observed_map
    robot_pose = planner.robot_pose
    goal_pose = planner.goal

    plotting_grid = np.zeros(partial_grid.shape)
    plotting_grid[partial_grid == COLLISION_VAL] = 0.0
    plotting_grid[partial_grid == FREE_VAL] = 1.0
    plotting_grid[partial_grid == UNOBSERVED_VAL] = 0.7
    ax.imshow(plotting_grid, origin='lower', cmap='gray')
    ax.set_title("The robot's partial map of the world\n"
                 "Robot (R), Goal (G), and Subgoals\n"
                 "Also includes the agent's plan.")

    if robot_poses is not None:
        xs = [p.x for p in robot_poses]
        ys = [p.y for p in robot_poses]
        ax.plot(ys, xs, 'b')

    ax.text(goal_pose.y,
            goal_pose.x,
            "G",
            bbox={
                "boxstyle": "round",
                "edgecolor": "black",
                "facecolor": "green",
                "linewidth": 2
            },
            ha='center',
            va='center',
            transform=ax.transData)

    ax.text(robot_pose.y,
            robot_pose.x,
            "R",
            bbox={
                "boxstyle": "round",
                "edgecolor": "black",
                "facecolor": "white",
                "linewidth": 2
            },
            ha='center',
            va='center',
            transform=ax.transData)


def plot_map_with_plan(ax, planner, subgoal_ind_dict, policy, subgoal_props, do_include_policy=True, robot_poses=None):
    partial_grid = planner.observed_map
    inflated_partial_grid = planner.inflated_grid
    robot_pose = planner.robot_pose
    goal_pose = planner.goal

    plotting_grid = np.zeros(partial_grid.shape)
    plotting_grid[partial_grid == COLLISION_VAL] = 0.0
    plotting_grid[partial_grid == FREE_VAL] = 1.0
    plotting_grid[partial_grid == UNOBSERVED_VAL] = 0.7
    ax.imshow(plotting_grid, origin='lower', cmap='gray')
    ax.set_title("The robot's partial map of the world\n"
                 "Robot (R), Goal (G), and Subgoals\n"
                 "Also includes the agent's plan.")

    # Set up
    ind_subgoal_dict = {ind: s for s, ind in subgoal_ind_dict.items()}
    subgoals = [s for s in subgoal_ind_dict.keys()]
    num_subgoals = len(subgoals)

    cmap = matplotlib.cm.get_cmap('viridis')
    colors = [cmap(ii / num_subgoals) for ii in range(num_subgoals)]

    # Plot the paths
    planning_grid = inflated_partial_grid.copy()
    planning_grid[partial_grid == UNOBSERVED_VAL] = COLLISION_VAL
    for subgoal in subgoals:
        planning_grid[subgoal.points[0, :],
                      subgoal.points[1, :]] = FREE_VAL

    poses = (
        [[robot_pose.x, robot_pose.y]] +
        [ind_subgoal_dict[ind].get_frontier_point() for ind in policy])
    probs = [subgoal_props[ind].prob_feasible for ind in policy]
    path_dat = []
    cpd = 1.0
    cpds = []
    for ii, (ps, pe, prob) in enumerate(zip(poses, poses[1:], probs)):
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            planning_grid, [pe[0], pe[1]], use_soft_cost=True)
        did_plan, path = get_path([ps[0], ps[1]],
                                  do_sparsify=True,
                                  do_flip=True)
        path_dat.append([path, colors[ii], float(cpd)])
        cpd *= (1 - prob)
        cpds.append(float(cpd))

    # Plot in reverse to 'first' paths go 'on top'
    if not do_include_policy:
        path_dat = path_dat[:1]
        path_dat[0][2] = 1.0

    for path, color, cpd in path_dat[::-1]:
        if len(path) < 2:
            continue

        ax.plot(path[1, :] - 0.5,
                path[0, :] - 0.5,
                color=color,
                alpha=0.8,
                linewidth=_get_linewidth_from_cpd(cpd))

    # Plot the paths to the goal
    planning_grid = inflated_partial_grid.copy()
    planning_grid[partial_grid == FREE_VAL] = COLLISION_VAL
    for subgoal in subgoals:
        planning_grid[subgoal.points[0, :],
                      subgoal.points[1, :]] = FREE_VAL

    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        planning_grid, [goal_pose.x, goal_pose.y],
        use_soft_cost=True)
    path_dat = []

    for color, cpd, ps in zip(colors, cpds, poses[1:]):
        did_plan, path = get_path([ps[0], ps[1]],
                                  do_sparsify=True,
                                  do_flip=False)
        path_dat.append([path, color, cpd])

    # Plot in reverse to 'first' paths go 'on top'
    if not do_include_policy:
        path_dat = path_dat[:1]
        path_dat[0][2] = 1.0

    if robot_poses is not None:
        xs = [p.x for p in robot_poses]
        ys = [p.y for p in robot_poses]
        ax.plot(ys, xs, 'b')

    for path, color, cpd in path_dat[::-1]:
        if len(path) < 2:
            continue
        ax.plot(path[1, :] - 0.5,
                path[0, :] - 0.5,
                color=color,
                alpha=0.6,
                linestyle='dotted',
                linewidth=_get_linewidth_from_cpd(cpd))

    # Plot the badges
    for ii, ind in reversed(list(enumerate(policy))):
        subgoal = ind_subgoal_dict[ind]
        s_cent = subgoal.get_frontier_point()
        label = str(ii)
        ax.text(s_cent[1],
                s_cent[0],
                label,
                bbox={"boxstyle": "round",
                      "edgecolor": colors[ii],
                      "facecolor": "white",
                      "linewidth": 2},
                transform=ax.transData)

    ax.text(goal_pose.y,
            goal_pose.x,
            "G",
            bbox={
                "boxstyle": "round",
                "edgecolor": "black",
                "facecolor": "green",
                "linewidth": 2
            },
            ha='center',
            va='center',
            transform=ax.transData)

    ax.text(robot_pose.y,
            robot_pose.x,
            "R",
            bbox={
                "boxstyle": "round",
                "edgecolor": "black",
                "facecolor": "white",
                "linewidth": 2
            },
            ha='center',
            va='center',
            transform=ax.transData)
