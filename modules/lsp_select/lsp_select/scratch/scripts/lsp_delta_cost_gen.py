import environments
import lsp
import lsp_select
import numpy as np
import matplotlib.pyplot as plt
import gridmap
from pathlib import Path

from lsp.planners import KnownSubgoalPlanner, LearnedSubgoalPlanner
from lsp_select.planners import LSPCycleGAN


def maze_eval(args):
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements,
        min_breadcrumb_signed_distance=4.0 * args.base_resolution)
    builder = environments.simulated.WorldBuildingUnityBridge
    robot = lsp.robot.Turtlebot_Robot(pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=map_data)

    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)
        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)
        simulator.frontier_grouping_inflation_radius = simulator.inflation_radius
        planning_loop = lsp.planners.PlanningLoop(goal,
                                                  known_map,
                                                  simulator,
                                                  unity_bridge=None,
                                                  robot=robot,
                                                  args=args,
                                                  verbose=True)
        known_planner = KnownSubgoalPlanner(goal=goal, known_map=known_map, args=args)
        planners = [LearnedSubgoalPlanner(goal=goal, args=args),
                    LSPCycleGAN(goal=goal, args=args)]

        delta_path_array = []
        delta_cost_array = []
        # last_subgoal_for_path = None
        min_delta_cost = 0
        min_delta_path = []
        total_delta_cost = 0
        total_delta_path = []
        count_identical_subgoals = 0
        subgoal_data = {'subgoals': [], 'subgoal_centroids': []}
        recorded_subgoals = []
        for counter, step_data in enumerate(planning_loop):
            # Update the planner objects
            known_planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])

            selected_subgoals = [known_planner.compute_selected_subgoal()]
            for planner in planners:
                planner.update(
                    {'image': step_data['image']},
                    step_data['robot_grid'],
                    step_data['subgoals'],
                    step_data['robot_pose'],
                    step_data['visibility_mask'])
                selected_subgoal = planner.compute_selected_subgoal()
                selected_subgoals.append(selected_subgoal)
            selected_planner_idx = 0
            planner = planners[selected_planner_idx]
            subgoal_centroids = []

            # if goal is not visible, chosen subgoal is not None
            if None not in selected_subgoals:
                for subgoal in selected_subgoals:
                    subgoal_centroids.extend(subgoal.get_frontier_point())
                if counter == 0:
                    is_identical_to_last = False
                else:
                    last_subgoal_m2 = subgoal_data['subgoals'][-1][2]
                    is_identical_to_last = check_identical_subgoals(selected_subgoals[2],
                                                                    last_subgoal_m2,
                                                                    planner.inflated_grid,
                                                                    step_data['subgoals'],
                                                                    step_data['robot_pose'],
                                                                    goal)
                subgoal_data['subgoals'].append(selected_subgoals)
                subgoal_data['subgoal_centroids'].append(subgoal_centroids)
                if selected_subgoals[0] != selected_subgoals[2]:
                    recorded_subgoals.append(selected_subgoals[2])
                    distances = lsp.core.compute_distances(step_data['subgoals'],
                                                           step_data['robot_grid'],
                                                           step_data['robot_pose'],
                                                           goal)
                    delta_cost = distances['robot'][selected_subgoals[2]]
                    delta_cost_array.append(delta_cost)
                    delta_path_array.append(get_path_in_known_grid(planner.inflated_grid,
                                                                   step_data['robot_pose'],
                                                                   selected_subgoals[2].get_frontier_point(),
                                                                   planner.subgoals))
                    if is_identical_to_last:
                        del recorded_subgoals[-2]
                        count_identical_subgoals += 1
                        # temp_last_subgoal_for_path = subgoal_data['subgoals'][-count_identical_subgoals:]
                        temp_delta_cost_array = delta_cost_array[-count_identical_subgoals:]
                        temp_delta_path_array = delta_path_array[-count_identical_subgoals:]
                        min_idx = np.argmin(temp_delta_cost_array)
                        min_delta_cost = temp_delta_cost_array[min_idx]
                        min_delta_path = temp_delta_path_array[min_idx]
                        # min_last_subgoal = temp_last_subgoal_for_path[min_idx]
                    else:
                        count_identical_subgoals = 0
                        total_delta_cost += min_delta_cost
                        total_delta_path.append(min_delta_path)
                        min_delta_cost = 0
                        min_delta_path = []

                    subgoal_centroids.append(delta_cost)

                else:
                    delta_cost = 0
                    delta_cost_array.append(delta_cost)
                    delta_path_array.append(None)
                    subgoal_centroids.append(delta_cost)
                subgoal_centroids.append(int(is_identical_to_last))

            planning_loop.set_chosen_subgoal(selected_subgoals[selected_planner_idx + 1])
            if args.do_plot:
                plt.ion()
                plt.figure(1)
                plt.clf()
                if step_data['image'] is not None:
                    ax = plt.subplot(211)
                    plt.imshow(step_data['image'])
                    ax = plt.subplot(212)
                else:
                    ax = plt.subplot(111)
                lsp_select.utils.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_select.utils.plotting.plot_grid_with_frontiers(
                    ax, planner.observed_map, known_map, planner.subgoals)
                lsp_select.utils.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_select.utils.plotting.plot_path(ax, planning_loop.current_path)
                if planning_loop.chosen_subgoal is not None:
                    model2_centroid = selected_subgoals[2].get_centroid().astype(int)
                    path2 = get_path_in_known_grid(planner.inflated_grid,
                                                   robot.pose,
                                                   model2_centroid,
                                                   planner.subgoals,
                                                   do_not_mask=planning_loop.chosen_subgoal)
                    # point = selected_subgoals[1].points[:, 0]
                    plt.title(f'{delta_cost=:.2f},{model2_centroid},{is_identical_to_last=}')
                    lsp_select.utils.plotting.plot_path(ax, path2, style='r:')
                lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)
                # if min_delta_path is not None and len(min_delta_path):
                #     lsp_select.utils.plotting.plot_path(ax, min_delta_path, style='c:')

                plt.show()
                plt.pause(0.1)
                # input()
        masked_grid = lsp.core.mask_grid_with_frontiers(planner.inflated_grid, step_data['subgoals'])

        path_points = [[pose.x, pose.y]]
        for subgoal in recorded_subgoals:
            path_points.append(subgoal.get_frontier_point())
        path_points.append([goal.x, goal.y])
        path_points = np.array(path_points)
        plt.ioff()
        plt.figure(1)
        ax = plt.subplot(111)
        for i in range(len(path_points) - 1):
            lsp_select.utils.plotting.plot_grid_with_frontiers(
                ax, planner.observed_map, known_map, planner.subgoals)
            start = path_points[i]
            end = path_points[i + 1]
            _, get_path = gridmap.planning.compute_cost_grid_from_position(
                masked_grid, end, use_soft_cost=True)
            _, path2 = get_path(start, do_sparsify=True, do_flip=True)
            print(path2)
            if len(path2) == 2:
                lsp_select.utils.plotting.plot_path(ax, path2, style='r-')
        lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)

        plt.show()

        return subgoal_data


def check_identical_subgoals(current_subgoal, last_subgoal, inflated_grid, current_all_subgoals, robot_pose, goal_pose):
    masked_grid_s1 = lsp.core.mask_grid_with_frontiers(inflated_grid, current_all_subgoals, do_not_mask=current_subgoal)
    masked_grid_s2 = lsp.core.mask_grid_with_frontiers(masked_grid_s1, [last_subgoal])
    _, get_path = gridmap.planning.compute_cost_grid_from_position(
        masked_grid_s2, [goal_pose.x, goal_pose.y], use_soft_cost=False)
    did_plan, _ = get_path([robot_pose.x, robot_pose.y],
                           do_sparsify=False,
                           do_flip=False)
    return not did_plan


def get_path_in_known_grid(inflated_grid, pose, goal, subgoals, do_not_mask=None):
    planning_grid = lsp.core.mask_grid_with_frontiers(inflated_grid, subgoals, do_not_mask=do_not_mask)
    _, get_path = gridmap.planning.compute_cost_grid_from_position(
        planning_grid, goal, use_soft_cost=True)
    _, path = get_path([pose.x, pose.y], do_sparsify=True, do_flip=True)
    return path


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--generator_network_file', type=str)
    parser.add_argument('--disable_cyclegan', action='store_true')
    parser.add_argument('--env', choices=['envA', 'envB', 'envC'], default='envA')
    args = parser.parse_args()
    args.current_seed = args.seed[0]
    subgoal_data = maze_eval(args)
    np.savetxt(Path(args.save_dir) / f'{args.env}_{args.current_seed}.txt',
               subgoal_data['subgoal_centroids'],
               fmt='%.0f')
