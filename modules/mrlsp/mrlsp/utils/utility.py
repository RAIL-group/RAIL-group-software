from common import Pose
import common
import random
import numpy as np
import lsp
import math
import gridmap
import copy
from gridmap.constants import (COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL)
from scipy.optimize import linear_sum_assignment
import itertools


# generate different start and goal poses for num_robots
def generate_start_and_goal(num_robots=1,
                            known_map=None,
                            same_start=False,
                            same_goal=False,
                            def_start=None,
                            def_goal=None):
    if (not same_start or not same_goal):

        start = [0 for i in range(num_robots)]
        goal = [0 for i in range(num_robots)]
        start_pose = [0 for i in range(num_robots)]
        goal_pose = [0 for i in range(num_robots)]
        for i in range(num_robots):
            while True:
                (x, y) = random.randint(0,
                                        len(known_map) - 1), random.randint(
                                            0,
                                            len(known_map[0]) - 1)
                if (known_map[x, y] == 0):
                    a = np.array([x, y])
                    start[i] = a
                    break

        for i in range(num_robots):
            while True:
                (x, y) = random.randint(0,
                                        len(known_map) - 1), random.randint(
                                            0,
                                            len(known_map[0]) - 1)
                if (known_map[x, y] == 0):
                    a = np.array([x, y])
                    goal[i] = a
                    break

        for i in range(num_robots):
            start_pose[i] = Pose(x=start[i][0],
                                 y=start[i][1],
                                 yaw=2 * math.pi * random.random())

            goal_pose[i] = Pose(x=goal[i][0],
                                y=goal[i][1],
                                yaw=2 * math.pi * random.random())
        print("not same start or goal")

    if same_start:
        print("same start")
        start_pose = [def_start for _ in range(num_robots)]

    if same_goal:
        print("same goal")
        goal_pose = [def_goal for _ in range(num_robots)]

    return start_pose, goal_pose


# define robot instance for num_robots
def robot(num_robots, start_pose, primitive_length, num_primitives, map_data):

    robot = [0 for i in range(num_robots)]

    for i in range(num_robots):
        robot[i] = lsp.robot.Turtlebot_Robot(start_pose[i],
                                             primitive_length=primitive_length,
                                             num_primitives=num_primitives,
                                             map_data=map_data)

    return robot


# Create lsp planner for num_robtos
def lsp_planner(args, num_robots, goal_pose):
    robot_lsp_planners = [0 for i in range(num_robots)]

    for i in range(num_robots):
        robot_lsp_planners[i] = lsp.planners.LearnedSubgoalPlanner(
            goal_pose[i], args)

    return robot_lsp_planners


# To update the current grid with the current observation
def update_grid(robot_grid, observed_grid):
    previous = robot_grid
    now = observed_grid
    current = lsp.constants.UNOBSERVED_VAL * np.ones_like(previous)

    for idx, _ in np.ndenumerate(current):
        # more like and gate
        if previous[idx] == UNOBSERVED_VAL and now[idx] == UNOBSERVED_VAL:
            current[idx] = UNOBSERVED_VAL
        elif previous[idx] == COLLISION_VAL and now[idx] == COLLISION_VAL:
            current[idx] = COLLISION_VAL
        elif previous[idx] == COLLISION_VAL and now[idx] == UNOBSERVED_VAL:
            current[idx] = COLLISION_VAL
        elif previous[idx] == UNOBSERVED_VAL and now[idx] == COLLISION_VAL:
            current[idx] = COLLISION_VAL
        else:
            current[idx] = FREE_VAL
    return current


# TODO: For every instance, send only pose not robot
def get_multirobot_distances(robot_grid, robots, goal_pose, subgoals):
    distances = {}
    distances['goal'] = lsp.core.get_goal_distances(robot_grid, goal_pose[0], frontiers=subgoals)
    distances['frontier'] = lsp.core.get_frontier_distances(robot_grid, frontiers=subgoals)
    for i, robot in enumerate(robots):
        distances[f'robot{i+1}'] = lsp.core.get_robot_distances(robot_grid, robot.pose, frontiers=subgoals)
    return distances


def get_all_distance(robot_grid, robots, goal_pose, subgoals):
    distances = []

    # goal_pose[0] is taken because all the goal is same
    # Incase of multiple goals, this has to go inside the for loop
    goal_dist = lsp.core.get_goal_distances(robot_grid,
                                            goal_pose[0],
                                            frontiers=subgoals)

    frontier_distances = lsp.core.get_frontier_distances(robot_grid,
                                                         frontiers=subgoals)

    for robot in robots:
        robot_dist = lsp.core.get_robot_distances(robot_grid,
                                                  robot.pose,
                                                  frontiers=subgoals)

        distance = {
            'robot': robot_dist,
            'goal': goal_dist,
            'frontier': frontier_distances
        }

        distances.append(distance)

    return distances


def get_top_n_frontiers(frontiers, distances, n):
    """This heuristic is for retrieving the 'best' N frontiers"""
    frontiers = [f for f in frontiers if f.prob_feasible > 0]

    h_prob = {s: s.prob_feasible for s in frontiers}
    fs_prob = sorted(list(frontiers),
                     key=lambda s: h_prob[s],
                     reverse=True)
    # take two subgoals with higher probability
    fs_collated = fs_prob[:2]
    # remaining subgoals for which gives the minimum cost
    subgoals = fs_prob[2:]

    order_heuristics = []
    order_heuristics.append({
        s: ii for ii, s in enumerate(subgoals)
    })
    order_heuristics.append({
        s: 1 - s.prob_feasible for s in subgoals
    })
    order_heuristics.append({
        s: distances['goal'][s] + distances['robot'][s] +
        s.prob_feasible * s.delta_success_cost +
        (1 - s.prob_feasible) * s.exploration_cost
        for s in subgoals
    })
    order_heuristics.append({
        s: distances['goal'][s] + distances['robot'][s]
        for s in subgoals
    })
    order_heuristics.append({
        s: distances['goal'][s] + distances['robot'][s] +
        s.delta_success_cost
        for s in subgoals
    })
    order_heuristics.append({
        s: distances['goal'][s] + distances['robot'][s] +
        s.exploration_cost
        for s in subgoals
    })

    heuristic_ordering_dat = []
    for heuristic in order_heuristics:
        ordered_subgoals = sorted(subgoals, reverse=False, key=lambda s: heuristic[s])
        ordering_cost = get_ordering_cost(ordered_subgoals, distances)
        heuristic_ordering_dat.append((ordering_cost, ordered_subgoals))
    subgoals = min(heuristic_ordering_dat, key=lambda hod: hod[0])[1]

    for s in subgoals:
        fs_collated.append(s)

    return fs_collated[0:n]


def get_top_n_frontiers_multirobot(num_robots, frontiers, distances, n):
    goal_dist = distances['goal']
    individual_distance = {}
    individual_distance['goal'] = goal_dist
    individual_distance['frontier'] = distances['frontier']
    best_frontiers = []
    seen = set()
    h_prob = {s: s.prob_feasible for s in frontiers}
    for i in range(num_robots):
        robot_dist = distances[f'robot{i+1}']
        individual_distance['robot'] = robot_dist
        bf = lsp.core.get_top_n_frontiers(frontiers, goal_dist, robot_dist, n)
        if len(bf) == 0:
            print("No frontiers returned by get_top_n_frontiers")
            fs_prob = sorted(list(frontiers),
                             key=lambda s: h_prob[s],
                             reverse=True)
            bf = fs_prob[:n]
        # bf = get_top_n_frontiers(frontiers, individual_distance, n)
        best_frontiers.append(bf)

    all_robot_best_frontiers = []
    for i in range(n):
        for bf in best_frontiers:
            if i < len(bf) and bf[i] not in seen:
                all_robot_best_frontiers.append(bf[i])
                seen.add(bf[i])

    top_frontiers = all_robot_best_frontiers[:n]
    return top_frontiers


def get_cost_dictionary(num_robots, distances, planner):
    cost_dictionary = [None for _ in range(num_robots)]
    for i in range(num_robots):
        cost_dictionary[i] = {
            subgoal: lsp.core.get_lowest_cost_ordering_beginning_with(
                subgoal, planner[i].subgoals, distances[i], do_sort=False)[0]
            for subgoal in planner[i].subgoals
        }
    subgoal_matrix = np.array([list(cd.keys()) for cd in cost_dictionary])
    cost_matrix = np.array([list(cd.values()) for cd in cost_dictionary])
    return cost_dictionary, subgoal_matrix, cost_matrix


def find_subgoal(robot_grid, robots, goal_pose, simulator):
    inflated_grid = lsp.constants.UNOBSERVED_VAL * np.ones_like(robot_grid)
    for robot in robots:
        observed_grid = simulator.get_inflated_grid(robot_grid, robot)
        observed_grid = gridmap.mapping.get_fully_connected_observed_grid(
            observed_grid, robot.pose)
        inflated_grid = update_grid(inflated_grid, observed_grid)
    subgoals = simulator.get_updated_frontier_set(inflated_grid, None, set())
    return subgoals, inflated_grid


def get_ordering_cost(subgoals, distances):
    fstate = None
    for s in subgoals:
        fstate = lsp.core.FState(s, distances, fstate)
    return fstate.cost


def limit_total_subgoals(num_robots, frontiers, distances, n):
    subgoals = set([copy.copy(s) for s in frontiers])
    extra_subgoals = set()
    # This is done to limit the frontiers beyond n
    # all the subgoals that are of low cost, are stored and returned in extra_subgoals
    if len(subgoals) > n:
        print(f"More than {n} subgoals, limiting to {n} subgoals")
        top_frontiers = set(
            get_top_n_frontiers_multirobot(num_robots, subgoals, distances, n))
        extra_subgoals = subgoals - top_frontiers
        subgoals = top_frontiers
    else:
        print("Printing subgoal probabilities:")
        print([s.prob_feasible for s in subgoals])
        chosen_subgoal = set([s for s in subgoals if s.prob_feasible != 0])
        print("Number of subgoals with non-zero probability: ", len(chosen_subgoal))
        extra_subgoals = subgoals - chosen_subgoal
        subgoals = chosen_subgoal
    return subgoals, extra_subgoals


def get_path_middle_point(known_map, start, goal, args):
    """This function returns the middle point on the path from goal to the
    robot starting position"""
    inflation_radius = args.inflation_radius_m / args.base_resolution
    inflated_mask = gridmap.utils.inflate_grid(
        known_map, inflation_radius=inflation_radius)
    # Now sample the middle point
    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        inflated_mask, [goal.x, goal.y])
    _, path = get_path([start.x, start.y], do_sparsify=False, do_flip=False)
    row, col = path.shape
    x = path[0][col // 2]
    y = path[1][col // 2]
    new_start_pose = common.Pose(x=x, y=y, yaw=2 * np.pi * np.random.rand())
    return new_start_pose


def get_frontier_time_by_triangle_formation(a, b, c, time_travelled):
    epsilon = 0.1
    if a + b < c:
        new_time = time_travelled * c / (a + b)
        return new_time
    elif b > a + c:
        new_time = time_travelled * c / (a + b)
        return new_time
    elif a > b + c:
        new_time = time_travelled * c / (a + b)
        return new_time
    elif abs((a + b) - c) <= epsilon:
        # the frontier is horizontally aligned (in positive direction)
        new_time = c - time_travelled
        return new_time
    elif abs((a + c) - b) <= epsilon:
        # the frontier is horizontally aligned (in negative direction)
        new_time = c + time_travelled
        return new_time
    elif abs((b + c) - a) <= epsilon:
        # the all the frontier lies in the same line
        new_time = abs(c - time_travelled)
        return new_time

    if a > 0:
        x = (c * c - b * b + a * a) / (2 * a)
        y = np.sqrt(c * c - x * x)
        frontier_point = np.array([x, y])
    else:
        raise AssertionError('First side of triangle not > 0')

    new_point = np.array([time_travelled, 0])
    new_time = np.linalg.norm(new_point - frontier_point)
    if math.isnan(new_time):
        print(a, b, c, time_travelled)
        raise AssertionError('new time is \'nan\'!!')
    return new_time


def find_action_index(action, all_actions):
    n = len(action)
    for i, a in enumerate(all_actions):
        result = [a[j] == action[j] for j in range(n)]
        # pdb.set_trace()
        if (all(result)):
            return i
    return None


def copy_dictionary(all_time):
    new_all_time = {}
    for time in all_time:
        new_all_time[time] = all_time[time].copy()
    return new_all_time


def find_action_list_from_cost_matrix_using_lsa(cost_matrix, subgoal_matrix):
    cost = cost_matrix
    num_robots = len(cost_matrix)
    left_robot = num_robots
    assigned_robot = 0
    joint_action = [None for i in range(num_robots)]
    count = 0
    while (left_robot != 0 and count < num_robots + 1):
        # find the lowest cost for the first 'k' robots, where k is the number of subgoals
        n_rows, n_cols = linear_sum_assignment(cost)
        for i, row in enumerate(n_rows):
            # assign the action to the robot if it is not previously assigned, i.e., not None
            if joint_action[row] is None:
                joint_action[row] = subgoal_matrix[row][n_cols[i]]
                assigned_robot += 1
                # replace the cost by a 'high number' so that it it doesn't get selected when doing lsa
                cost[row] = 1e11
            # decrement the left robot so that it loops and assigns to the remaining robot.
        left_robot = num_robots - assigned_robot
        count += 1
    # for every none items in the joint action, randomly assign a subgoal in the joint action that's not none
    if None in joint_action:
        non_none_items = [item for item in joint_action if item is not None]
        none_idx = [idx for idx, val in enumerate(joint_action) if val is None]
        for idx in none_idx:
            joint_action[idx]= np.random.choice(non_none_items)
    return joint_action


def get_action_combination(iterables, repeat, same_action=False):
    '''If same action = false i.e., permutations without replacement
    If no.of frontiers > no. of robots, permute the frontiers'''
    if not same_action and len(iterables) >= repeat:
        actions = list(itertools.permutations(iterables, repeat))
        return actions
    else:
        '''same action = true i.e., permutations with replacement
        If no.of frontiers < no.of robots, make sure some robot explore the 'same' frontier'''
        actions = list(itertools.product(iterables, repeat=repeat))
        '''same_action = false i.e, you want robot all robot to prevent exploring 'same' frontier
        as much as possible. Hence maximum allowed same action = no. of frontiers / robot'''
        if not same_action and len(iterables) > 1:
            final_action = []
            same_action_max = int(np.ceil(repeat / len(iterables)))
            for action in actions:
                max_action_same = any([action.count(action[i]) > same_action_max for i in range(len(action))])
                if max_action_same:
                    continue
                final_action.append(action)
            return final_action
        return actions
