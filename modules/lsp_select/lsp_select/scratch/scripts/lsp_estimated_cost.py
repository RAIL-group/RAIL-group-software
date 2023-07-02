import numpy as np
import environments
import lsp
import lsp_select
import matplotlib.pyplot as plt
from lsp.planners import LearnedSubgoalPlanner
from lsp_select.planners import LSPCycleGAN
import gridmap
from lsp_select.utils.misc import is_feasible_subgoal
from lsp.constants import FREE_VAL, UNOBSERVED_VAL, COLLISION_VAL
from pathlib import Path


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

        planners = [planner(goal=goal, args=args) for planner in args.planners]
        spidx = 0  # selected planner index

        navigation_data = {'policies': [[] for _ in planners],
                           'steps': [],
                           'spidx': spidx,
                           'start_pose': pose,
                           'goal_pose': goal,
                           'net_motion': None,
                           'correct_subgoals': [],
                           'final_masked_grid': None,
                           'known_path': [],
                           'known_cost': None}

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
                # latest_ordering = planner.latest_ordering if selected_subgoal is not None else []
                if selected_subgoal is None:
                    latest_ordering = []
                else:
                    # latest_ordering = planner.latest_ordering
                    latest_ordering = get_full_policy(planner.latest_ordering,
                                                      planner.subgoals,
                                                      planner.inflated_grid,
                                                      planner.robot_pose,
                                                      goal)
                policies.append([selected_subgoal, latest_ordering])
                navigation_data['policies'][pid].append(latest_ordering)

            chosen_subgoal, chosen_policy = policies[spidx]
            planning_loop.set_chosen_subgoal(chosen_subgoal)

            is_goal_visible = chosen_subgoal is None

            navigation_data['steps'].append([planner.robot_pose,
                                            planner.subgoals,
                                            planner.inflated_grid,
                                            is_goal_visible])

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
                for j, s in enumerate(planners[0].latest_ordering):
                    point = s.get_frontier_point()
                    plt.scatter(*point)
                    plt.text(*point, str(j))
                plt.title(f'{type(planner).__name__}')
                plt.xlabel(f'seed={args.current_seed}')
                plt.pause(0.01)

    step_data = navigation_data['steps']
    final_subgoals = step_data[-1][1]
    final_masked_grid = lsp.core.mask_grid_with_frontiers(step_data[-1][2], final_subgoals)
    known_path = get_path_in_known_grid(final_masked_grid, pose, goal, final_subgoals, do_not_mask=None)
    known_cost = compute_cost_from_path(known_path)
    correct_subgoals = []
    for i, (robot_pose, subgoals, inflated_grid, is_goal_visible) in enumerate(step_data):
        if is_goal_visible:
            break
        for subgoal in subgoals:
            if is_feasible_subgoal(subgoal, final_masked_grid, subgoals, robot_pose, goal):
                correct_subgoals.append(subgoal)
                break
    navigation_data['correct_subgoals'] = correct_subgoals
    navigation_data['final_masked_grid'] = final_masked_grid
    navigation_data['net_motion'] = robot.net_motion
    navigation_data['known_path'] = known_path
    navigation_data['known_cost'] = known_cost
    if args.do_plot:
        plt.ion()
        plt.figure(1)
        plt.clf()
        ax = plt.subplot(111)
        lsp_select.utils.plotting.plot_grid_with_frontiers(
            ax, planner.observed_map, known_map, planner.subgoals)
        lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)
        # lsp_select.utils.plotting.plot_path(ax, navigation_data['known_path'], style='g:')
        plt.title(f'net_motion={navigation_data["net_motion"]:.2f}, known_cost={navigation_data["known_cost"]:.2f}')
        plt.show()
        plt.pause(0.01)
        input()

    return navigation_data


def get_full_policy(latest_ordering, subgoals, inflated_grid, robot_pose, goal, num_frontiers_max=12):
    remaining_frontiers = [f for f in subgoals if f not in latest_ordering]
    distances = lsp.core.compute_distances(subgoals,
                                           inflated_grid,
                                           robot_pose,
                                           goal)
    while len(latest_ordering) < len(subgoals):
        remaining_frontiers = [f for f in subgoals if f not in latest_ordering]
        if len(remaining_frontiers) > num_frontiers_max:
            remaining_frontiers = lsp.core.get_top_n_frontiers(remaining_frontiers,
                                                               distances['goal'],
                                                               distances['robot'],
                                                               num_frontiers_max)
        # "move" the robot to the last frontier.
        distances['robot'] = {
            f: distances['frontier'][frozenset([f, latest_ordering[-1]])] for f in remaining_frontiers
        }
        _, updated_ordering = lsp.core.get_lowest_cost_ordering(remaining_frontiers, distances)
        latest_ordering += updated_ordering
        remaining_frontiers = [f for f in subgoals if f not in latest_ordering]
    return latest_ordering


def eval_alternate_policies(step_data, correct_subgoals, alternate_policies, start_pose, goal_pose):
    # final_subgoals = step_data[-1][1]
    # final_masked_grid = lsp.core.mask_grid_with_frontiers(step_data[-1][2], final_subgoals)
    # known_path = get_path_in_known_grid(final_masked_grid, start_pose, goal_pose, final_subgoals, do_not_mask=None)
    # known_cost = compute_cost_from_path(known_path)
    # correct_subgoals = []
    # for i, (robot_pose, subgoals, inflated_grid, is_goal_visible) in enumerate(step_data):
    #     if is_goal_visible:
    #         break
    #     for subgoal in subgoals:
    #         if is_feasible_subgoal(subgoal, final_masked_grid, subgoals, robot_pose, goal_pose):
    #             correct_subgoals.append(subgoal)
    #             break

    alt_visited_subgoals = set()
    delta_cost_array = []
    for i, (robot_pose, subgoals, inflated_grid, is_goal_visible) in enumerate(step_data):
        if is_goal_visible:
            break
        correct_subgoal = correct_subgoals[i]
        # plt.ion()
        # plt.figure(1)
        # plt.clf()
        # ax = plt.subplot(111)
        # lsp_select.utils.plotting.plot_pose(ax, robot_pose, color='blue')
        # lsp_select.utils.plotting.plot_grid_with_frontiers(
        #     ax, inflated_grid, None, subgoals)
        # plt.scatter(*correct_subgoal.get_frontier_point())
        # plt.show()
        # plt.pause(0.01)
        # input()
        _, _, sel_cost = get_subgoals_path_cost([correct_subgoal], [correct_subgoal],
                                                subgoals,
                                                robot_pose,
                                                inflated_grid,
                                                is_goal_visible)
        alternate_policy = [s for s in alternate_policies[i] if s not in alt_visited_subgoals]
        # alternate_policy = alternate_policies[i]
        # print(f'correct_subgoal:{correct_subgoal.get_frontier_point()}')
        # print(f'alternate_policy:\n {[s.get_frontier_point() for s in alternate_policy]}')
        alt_subgoals, alt_path, alt_cost = get_subgoals_path_cost([correct_subgoal], alternate_policy,
                                                                  subgoals,
                                                                  robot_pose,
                                                                  inflated_grid,
                                                                  is_goal_visible)
        # print([s.get_frontier_point() for s in alt_subgoals])
        delta_cost = max(0, alt_cost - sel_cost)
        delta_cost_array.append(delta_cost)

        if args.do_plot:
            plt.ion()
            plt.figure(1)
            plt.clf()
            ax = plt.subplot(111)
            lsp_select.utils.plotting.plot_pose(ax, robot_pose, color='blue')
            lsp_select.utils.plotting.plot_grid_with_frontiers(
                ax, inflated_grid, None, subgoals)
            lsp_select.utils.plotting.plot_pose(ax, goal_pose, color='green', filled=False)
            # lsp_select.utils.plotting.plot_path(ax, alt_path, style='r:')
            for j, s in enumerate(alternate_policies[i]):
                # if s == correct_subgoal:
                #     # break
                #     point = s.get_frontier_point()
                #     plt.scatter(*point, c='green')
            # for j, s in enumerate(alt_subgoals):  # noqa:E115
                # if s in alt_visited_subgoals:
                #     continue
                point = s.get_frontier_point()
                plt.scatter(*point, c='red')
                # plt.text(*point, str(j))
            correct_point = correct_subgoal.get_frontier_point()
            plt.scatter(*correct_point, c='green')
            # plt.text(*correct_point, '  true')
            plt.title(f'{delta_cost=:.2f}')
            plt.show()
            plt.pause(0.01)
            input()

        alt_visited_subgoals.update(alt_subgoals[:-1])

    return delta_cost_array


def aggregate_delta_costs(delta_costs, min_non_zero_len=2, min_zero_len=2):
    def remove_conjecutive_zeros(delta_costs, min_zero_len=1):
        for _ in range(min_zero_len - 1):
            final_arr = []
            was_prev_zero = False
            for i in range(len(delta_costs)):
                if delta_costs[i] == 0:
                    if was_prev_zero:
                        final_arr.append(delta_costs[i])
                        continue
                    else:
                        was_prev_zero = True
                else:
                    final_arr.append(delta_costs[i])
                    was_prev_zero = False
            delta_costs = np.array(final_arr)
        return delta_costs
    delta_costs = np.array(delta_costs, dtype=float)
    delta_costs = remove_conjecutive_zeros(delta_costs, min_zero_len)
    # print(delta_costs)
    delta_costs[delta_costs == 0] = np.inf
    aggregate_array = []
    for sub_cost in np.split(delta_costs, np.where(delta_costs == np.inf)[0]):
        if len(sub_cost) > min_non_zero_len:
            aggregate_array.append(np.min(sub_cost))
    aggregate_array = np.array(aggregate_array)
    # print(aggregate_array)
    return aggregate_array.sum()


# def aggregate_delta_costs(delta_costs, min_len=1):
#     delta_costs = np.array(delta_costs, dtype=float)
#     delta_costs[delta_costs == 0] = np.inf
#     total_delta_cost = 0
#     for sub_cost in np.split(delta_costs, np.where(delta_costs == np.inf)[0]):
#         if len(sub_cost) > min_len:
#             total_delta_cost += np.min(sub_cost)
#     return total_delta_cost


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
                           is_goal_visible):
    if is_goal_visible:
        return [], np.array([[], []]), 0
    alt_subgoals = compute_alt_subgoals(policy1, policy2)
    alt_path = compute_subgoals_path(alt_subgoals, all_subgoals, pose, inflated_grid)
    alt_cost = compute_cost_from_path(alt_path)
    return alt_subgoals, alt_path, alt_cost


def compute_alt_subgoals(policy1, policy2):
    if len(policy2) == 0 or policy1[0] == policy2[0]:
        return [policy1[0]]
    alt_subgoals = []
    for s in policy2:
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
        planning_grid, [goal.x, goal.y], use_soft_cost=True)
    _, path = get_path([pose.x, pose.y], do_sparsify=True, do_flip=True)
    return path


def generate_cost_data(args):
    args.planners = [LearnedSubgoalPlanner, LSPCycleGAN]
    navigation_data_1 = maze_eval(args)
    actual_cost_1 = navigation_data_1['net_motion']
    known_cost = navigation_data_1['known_cost']
    delta_cost_2_array = eval_alternate_policies(navigation_data_1['steps'],
                                                 navigation_data_1['correct_subgoals'],
                                                 navigation_data_1['policies'][1],
                                                 navigation_data_1['start_pose'],
                                                 navigation_data_1['goal_pose'])
    args.planners = args.planners[::-1]
    navigation_data_2 = maze_eval(args)
    actual_cost_2 = navigation_data_2['net_motion']
    delta_cost_1_array = eval_alternate_policies(navigation_data_2['steps'],
                                                 navigation_data_2['correct_subgoals'],
                                                 navigation_data_2['policies'][1],
                                                 navigation_data_2['start_pose'],
                                                 navigation_data_2['goal_pose'])
    return known_cost, actual_cost_1, actual_cost_2, delta_cost_1_array, delta_cost_2_array


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

    # args.current_seed = 923
    # costs_data = generate_cost_data(args)

    for seed in range(*args.seed):
        filepath = Path(args.save_dir) / f'actualcosts_{seed}.txt'
        if filepath.is_file():
            print(f'Data already exists for {seed=}.')
            continue

        print(f'Generating data for {seed=}.')
        args.current_seed = seed
        costs_data = generate_cost_data(args)
        np.savetxt(filepath, costs_data[:3], fmt='%.4f')
        np.savetxt(Path(args.save_dir) / f'deltacosts1_{seed}.txt', costs_data[3], fmt='%.4f')
        np.savetxt(Path(args.save_dir) / f'deltacosts2_{seed}.txt', costs_data[4], fmt='%.4f')

    files = Path(args.save_dir).glob('actualcosts_*.txt')
    actual_costs = np.array([np.loadtxt(f) for f in sorted(files)])
    known_cost = actual_costs[:, 0]
    actual_cost_1 = actual_costs[:, 1]
    actual_cost_2 = actual_costs[:, 2]

    files1 = Path(args.save_dir).glob('deltacosts1_*.txt')
    delta_costs_1 = [np.loadtxt(f) for f in sorted(files1)]
    files2 = Path(args.save_dir).glob('deltacosts2_*.txt')
    delta_costs_2 = [np.loadtxt(f) for f in sorted(files2)]

    delta_cost_1 = [aggregate_delta_costs(dc, 3, 2) for dc in delta_costs_1]
    delta_cost_2 = [aggregate_delta_costs(dc, 3, 2) for dc in delta_costs_2]

    if args.do_plot:
        plt.ioff()
        plt.subplot(121)
        # act_delta_cost_2 = np.maximum(0, actual_cost_2 - known_cost)
        # plt.scatter(actual_cost_2, known_cost + delta_cost_2)
        for i, (x, y) in enumerate(zip(actual_cost_2, known_cost + delta_cost_2), 900):
            plt.scatter(x, y, color='tab:blue', alpha=0.6)
            # plt.text(x, y, str(i), fontdict={'size': 8})
        plt.plot([0, 1400], [0, 1400], 'r:')

        # A1 = np.vstack([est_delta_cost_1, np.ones(len(est_delta_cost_1))]).T
        # m1, c1 = np.linalg.lstsq(A1, delta_cost_2, rcond=None)[0]
        # plt.plot(est_delta_cost_1, m1 * est_delta_cost_1 + c1, 'g', label='Fitted line')

        plt.xlabel(r'$P_B$')
        plt.ylabel(r'$\hat{P_B}$')
        plt.axis('equal')
        plt.title(r'LearnedSubgoalPlanner ($P_A$) in control')

        plt.subplot(122)
        # act_delta_cost_1 = np.maximum(0, actual_cost_1 - known_cost)
        # plt.scatter(actual_cost_1, known_cost + delta_cost_1)
        for i, (x, y) in enumerate(zip(actual_cost_1, known_cost + delta_cost_1), 900):
            plt.scatter(x, y, color='tab:blue', alpha=0.6)
            # plt.text(x, y, str(i), fontdict={'size': 8})
        plt.plot([0, 400], [0, 400], 'r:')

        # A2 = np.vstack([est_delta_cost_2, np.ones(len(est_delta_cost_2))]).T
        # m2, c2 = np.linalg.lstsq(A2, delta_cost_1, rcond=None)[0]
        # plt.plot(est_delta_cost_2, m2 * est_delta_cost_2 + c2, 'g', label='Fitted line')
        plt.xlabel(r'$P_A$')
        plt.ylabel(r'$\hat{P_A}$')
        plt.axis('equal')
        plt.title(r'LSPCycleGAN ($P_B$) in control')
        # plt.xlabel('True delta cost')
        # plt.ylabel('Estimated delta cost')
        # plt.axis('equal')
        # plt.title('LSPCycleGAN in control')

        # plt.subplot(223)
        # plt.scatter(act_delta_cost_1 - act_delta_cost_2, act_delta_cost_1 - delta_cost_2)
        # plt.xlabel(r'$\nabla P_A - \hat{\nabla} P_B$')
        # plt.ylabel(r'$\nabla P_A - \nabla P_B$')
        # plt.axis('equal')

        # plt.subplot(224)
        # plt.scatter(act_delta_cost_2 - act_delta_cost_1, act_delta_cost_2 - delta_cost_1)
        # plt.xlabel(r'$\nabla P_B - \hat{\nabla} P_A$')
        # plt.ylabel(r'$\nabla P_B - \nabla P_A$')
        # plt.axis('equal')
        plt.show()
