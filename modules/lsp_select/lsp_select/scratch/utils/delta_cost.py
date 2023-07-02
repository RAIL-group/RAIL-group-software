import numpy as np
import lsp
import copy
import gridmap
from lsp.constants import FREE_VAL, UNOBSERVED_VAL, COLLISION_VAL


def get_full_policy(latest_ordering, subgoals, inflated_grid, robot_pose, goal, num_frontiers_max=12):
    if latest_ordering is None and len(subgoals) == 1:
        latest_ordering = [list(subgoals)[0]]
    remaining_frontiers = [f for f in subgoals if f not in latest_ordering]
    distances = lsp.core.compute_distances(subgoals,
                                           inflated_grid,
                                           robot_pose,
                                           goal)
    while len(latest_ordering) < len(subgoals):
        distances_ = copy.copy(distances)
        remaining_frontiers = [f for f in subgoals if f not in latest_ordering]
        if len(remaining_frontiers) > num_frontiers_max:
            remaining_frontiers = lsp.core.get_top_n_frontiers(remaining_frontiers,
                                                               distances_['goal'],
                                                               distances_['robot'],
                                                               num_frontiers_max)
        # "move" the robot to the last frontier.
        distances_['robot'] = {
            f: distances_['frontier'][frozenset([f, latest_ordering[-1]])] for f in remaining_frontiers
        }
        _, updated_ordering = lsp.core.get_lowest_cost_ordering(remaining_frontiers, distances_)
        latest_ordering += updated_ordering
        remaining_frontiers = [f for f in subgoals if f not in latest_ordering]
    return latest_ordering


def eval_alternate_policies(step_data, correct_subgoals, alternate_policies, start_pose, goal_pose):
    alt_visited_subgoals = set()
    delta_cost_array = []
    for i, (robot_pose, subgoals, inflated_grid, is_goal_visible) in enumerate(step_data):
        if is_goal_visible:
            break
        correct_subgoal = correct_subgoals[i]
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

        # if args.do_plot:
        #     plt.ion()
        #     plt.figure(1)
        #     plt.clf()
        #     ax = plt.subplot(111)
        #     lsp_select.utils.plotting.plot_pose(ax, robot_pose, color='blue')
        #     lsp_select.utils.plotting.plot_grid_with_frontiers(
        #         ax, inflated_grid, None, subgoals)
        #     lsp_select.utils.plotting.plot_pose(ax, goal_pose, color='green', filled=False)
        #     lsp_select.utils.plotting.plot_path(ax, alt_path, style='r:')
        #     # for j, s in enumerate(alternate_policies[i]):
        #     #     if s == correct_subgoal:
        #     #         break
        #     #     point = s.get_frontier_point()
        #     #     plt.scatter(*point)
        #     for j, s in enumerate(alt_subgoals):
        #         # if s in alt_visited_subgoals:
        #         #     continue
        #         point = s.get_frontier_point()
        #         plt.scatter(*point)
        #         plt.text(*point, str(j))
        #     correct_point = correct_subgoal.get_frontier_point()
        #     plt.scatter(*correct_point, c='green')
        #     plt.text(*correct_point, '  true')
        #     plt.title(f'{delta_cost=:.2f}')
        #     plt.show()
        #     plt.pause(0.01)
        #     input()

        alt_visited_subgoals.update(alt_subgoals[:-1])

    return delta_cost_array


def aggregate_delta_costs(delta_costs, min_non_zero_len=3, min_zero_len=3):
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


def is_feasible_subgoal(subgoal, final_masked_grid, subgoals, robot_pose, goal_pose):
    masked_grid = lsp.core.mask_grid_with_frontiers(final_masked_grid, set(subgoals), do_not_mask=subgoal)
    _, get_path = gridmap.planning.compute_cost_grid_from_position(masked_grid,
                                                                   [goal_pose.x, goal_pose.y],
                                                                   use_soft_cost=False)
    did_plan, _ = get_path([robot_pose.x, robot_pose.y], do_sparsify=False, do_flip=False)

    return did_plan


def get_subgoal_labels(subgoals_data, final_masked_grid, goal_pose):
    prob_feasible_array = []
    for counter, data in enumerate(subgoals_data):
        subgoals = data['subgoals']
        robot_pose = data['robot_pose']
        for subgoal in subgoals:
            prob_feasible_true = is_feasible_subgoal(subgoal, final_masked_grid, subgoals, robot_pose, goal_pose)
            prob_feasible_array.append([counter, subgoal.prob_feasible, prob_feasible_true])
    return np.array(prob_feasible_array)


def compute_distances(frontiers, grid, robot_pose, goal_pose, downsample_factor=1):
    goal_distances = lsp.core.get_goal_distances(grid, goal_pose, frontiers=frontiers,
                                                 downsample_factor=downsample_factor)
    robot_distances = lsp.core.get_robot_distances(
        grid, robot_pose, frontiers=frontiers,
        downsample_factor=downsample_factor)
    frontier_distances = lsp.core.get_frontier_distances(
        grid, frontiers=frontiers, downsample_factor=downsample_factor)
    distances = {
        'frontier': frontier_distances,
        'robot': robot_distances,
        'goal': goal_distances,
    }
    return distances
