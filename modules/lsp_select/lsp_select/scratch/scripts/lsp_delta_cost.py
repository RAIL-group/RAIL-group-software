import numpy as np
import environments
import lsp
import lsp_select
import matplotlib.pyplot as plt
from lsp.planners import LearnedSubgoalPlanner
from lsp_select.planners import LSPCycleGAN
import gridmap

from lsp.constants import FREE_VAL, UNOBSERVED_VAL, COLLISION_VAL


def maze_eval(args):
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    # goal = corrupt_robot_pose(known_map, args) if args.env == 'envC' else goal
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

        planners = [LearnedSubgoalPlanner(goal=goal, args=args),
                    LSPCycleGAN(goal=goal, args=args)]
        spidx = 0  # selected planner index

        delta_cost_array = []

        alt_visited_subgoals = set()

        navigation_data = {'policies': [[] for _ in planners],
                           'steps': []}

        for counter, step_data in enumerate(planning_loop):
            policies = []  # stores (subgoal, policy) for each planner
            for pid, planner in enumerate(planners):
                planner.update(
                    {'image': step_data['image']},
                    step_data['robot_grid'],
                    step_data['subgoals'],
                    step_data['robot_pose'],
                    step_data['visibility_mask'])
                selected_subgoal = planner.compute_selected_subgoal()
                latest_ordering = planner.latest_ordering if selected_subgoal is not None else []
                policies.append([selected_subgoal, latest_ordering])
                navigation_data['policies'][pid].append(latest_ordering)

            chosen_subgoal, chosen_policy = policies[spidx]
            planning_loop.set_chosen_subgoal(chosen_subgoal)

            is_goal_visible = chosen_subgoal is None

            navigation_data['steps'].append(planner.robot_pose,
                                            planner.subgoals,
                                            planner.inflated_grid,
                                            is_goal_visible)
            _, _, sel_cost = get_subgoals_path_cost(chosen_policy, chosen_policy,
                                                    planners[spidx].subgoals,
                                                    planners[spidx].robot_pose,
                                                    planners[spidx].inflated_grid,
                                                    is_goal_visible)

            for i, planner in enumerate(planners):
                if i == spidx:
                    continue
                other_policy = policies[i][1]
                alt_subgoals, alt_path, alt_cost = get_subgoals_path_cost(chosen_policy, other_policy,
                                                                          planner.subgoals,
                                                                          planner.robot_pose,
                                                                          planner.inflated_grid,
                                                                          is_goal_visible,
                                                                          visited_subgoals=alt_visited_subgoals)
                alt_visited_subgoals.update(alt_subgoals)
                delta_cost = max(0, alt_cost - sel_cost)
                delta_cost_array.append(delta_cost)

            planner = planners[spidx]
            if args.do_plot:
                plt.ion()
                plt.figure(1)
                plt.clf()
                plt.subplot(211)
                plt.imshow(step_data['image'])
                ax = plt.subplot(212)
                lsp_select.utils.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_select.utils.plotting.plot_grid_with_frontiers(
                    ax, planner.observed_map, known_map, planner.subgoals)
                lsp_select.utils.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_select.utils.plotting.plot_path(ax, planning_loop.current_path)
                lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)
                for i, s in enumerate(alt_subgoals):
                    point = s.get_frontier_point()
                    plt.scatter(*point)
                    plt.text(*point, str(i))
                lsp_select.utils.plotting.plot_path(ax, alt_path, style='r:')

                plt.title(f'{type(planner).__name__}, {delta_cost=:.2f}')
                plt.xlabel(f'seed={args.current_seed}')
                plt.pause(0.01)
                input()

        delta_cost_array = np.array(delta_cost_array, dtype=float)
        delta_cost_array[delta_cost_array == 0] = np.inf
        total_delta_cost = 0
        for sub_cost in np.split(delta_cost_array, np.where(delta_cost_array == np.inf)[0]):
            if len(sub_cost) > 1:
                total_delta_cost += np.min(sub_cost)
        print(f'{total_delta_cost=}')


def check_identical_subgoals(current_subgoal, last_subgoal, inflated_grid, current_all_subgoals, robot_pose, goal_pose):
    masked_grid_s1 = lsp.core.mask_grid_with_frontiers(inflated_grid, current_all_subgoals, do_not_mask=current_subgoal)
    masked_grid_s2 = lsp.core.mask_grid_with_frontiers(masked_grid_s1, [last_subgoal])
    _, get_path = gridmap.planning.compute_cost_grid_from_position(
        masked_grid_s2, [goal_pose.x, goal_pose.y], use_soft_cost=False)
    did_plan, _ = get_path([robot_pose.x, robot_pose.y],
                           do_sparsify=False,
                           do_flip=False)
    return not did_plan


def get_subgoals_path_cost(policy1, policy2, all_subgoals, pose, inflated_grid,
                           is_goal_visible, visited_subgoals=set()):
    if is_goal_visible:
        return [], np.array([[], []]), 0
    alt_subgoals = compute_alt_subgoals(policy1, policy2, visited_subgoals=visited_subgoals)
    alt_path = compute_subgoals_path(alt_subgoals, all_subgoals, pose, inflated_grid)
    alt_cost = compute_cost_from_path(alt_path)
    return alt_subgoals, alt_path, alt_cost


def compute_alt_subgoals(policy1, policy2, visited_subgoals=set()):
    if policy1[0] == policy2[0]:
        return [policy1[0]]
    alt_subgoals = []
    for s in policy2:
        if s not in visited_subgoals:
            alt_subgoals.append(s)
        if s == policy1[0]:
            break
    return alt_subgoals


def compute_subgoals_path(path_subgoals, all_subgoals, pose, inflated_grid):
    planning_grid = lsp.core.mask_grid_with_frontiers(inflated_grid, all_subgoals)
    path_points = [[pose.x, pose.y]]
    for subgoal in path_subgoals:
        planning_grid[subgoal.points[0], subgoal.points[1]] = FREE_VAL
        planning_grid[planning_grid == UNOBSERVED_VAL] = COLLISION_VAL
        path_points.append(subgoal.get_frontier_point())
    path_points = np.array(path_points)

    alt_path = [[], []]
    for i in range(len(path_points) - 1):
        start = path_points[i]
        end = path_points[i + 1]
        _, get_path = gridmap.planning.compute_cost_grid_from_position(
            planning_grid, end, use_soft_cost=True)
        did_plan, path = get_path(start, do_sparsify=True, do_flip=True)
        if did_plan:
            alt_path[0].extend(path[0])
            alt_path[1].extend(path[1])
    return np.array(alt_path)


def compute_cost_from_path(path):
    return np.linalg.norm(path[:, 1:] - path[:, :-1], axis=0).sum()


def get_path_in_known_grid(inflated_grid, pose, goal, subgoals, do_not_mask=None):
    planning_grid = lsp.core.mask_grid_with_frontiers(inflated_grid, subgoals, do_not_mask=do_not_mask)
    _, get_path = gridmap.planning.compute_cost_grid_from_position(
        planning_grid, goal, use_soft_cost=True)
    _, path = get_path([pose.x, pose.y], do_sparsify=True, do_flip=True)
    return path


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--generator_network_file', type=str)
    parser.add_argument('--disable_cyclegan', action='store_true')
    parser.add_argument('--planner', choices=['lsp', 'lspcyclegan', 'nonlearned'])
    parser.add_argument('--env', choices=['envA', 'envB', 'envC'])
    args = parser.parse_args()

    args.current_seed = args.seed[0]
    maze_eval(args)
