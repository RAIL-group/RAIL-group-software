import numpy as np
from .planner import BaseLSPPlanner
import lsp
import mrlsp
from mrlsp.utils.utility import (get_all_distance,
                                 find_action_list_from_cost_matrix_using_lsa)


class BaselineLSA(BaseLSPPlanner):
    '''Baseline 1: Linear sum assignment using LSP'''
    def __init__(self, robots, goal, args):
        super(BaselineLSA, self).__init__(robots, goal, args)

    def compute_selected_subgoal(self):
        # If goal in range, return None as action
        if lsp.core.goal_in_range(grid=self.robot_grid,
                                  robot_pose=None, goal_pose=self.goal_pose, frontiers=self.subgoals):
            joint_action = [None for i in range(len(self.robots))]
            return joint_action
        distances = get_all_distance(self.inflated_grid, self.robots, [self.goal_pose], self.subgoals)
        # find cost for robot to reach goal from all the subgoal, and create a cost matrix.
        cost_dictionary = [None for _ in range(len(self.robots))]
        for i in range(len(self.robots)):
            cost_dictionary[i] = {
                subgoal: lsp.core.get_lowest_cost_ordering_beginning_with(
                    subgoal, self.subgoals, distances[i], do_sort=False)[0]
                for subgoal in self.subgoals
            }
        subgoal_matrix = np.array([list(cd.keys()) for cd in cost_dictionary])
        cost_matrix = np.array([list(cd.values()) for cd in cost_dictionary])
        # from the cost matrix, return the list of subgoals that has the least cost.
        joint_action = find_action_list_from_cost_matrix_using_lsa(cost_matrix, subgoal_matrix)

        return joint_action


class BaselineLSAExcludingAction(BaseLSPPlanner):
    '''Baseline 2: Linear sum assignment excluding action using LSP'''
    def __init__(self, robots, goal, args):
        super(BaselineLSAExcludingAction, self).__init__(robots, goal, args)

    def compute_selected_subgoal(self):
        if lsp.core.goal_in_range(grid=self.robot_grid,
                                  robot_pose=None, goal_pose=self.goal_pose, frontiers=self.subgoals):
            joint_action = [None for i in range(len(self.robots))]
            return joint_action
        actions_combination = mrlsp.utils.utility.get_action_combination(self.subgoals, len(self.robots))
        distances = get_all_distance(self.inflated_grid, self.robots, [self.goal_pose], self.subgoals)
        cost_of_action_combination = []
        for action in actions_combination:
            subgoals_except_in_action = [sg for sg in self.subgoals if sg not in action]
            final_cost = 0
            for i, a in enumerate(action):
                cost = lsp.core.get_lowest_cost_ordering_beginning_with(
                    a,
                    subgoals=subgoals_except_in_action,
                    distances=distances[i],
                    do_sort=False
                )[0]
                final_cost += cost
            cost_of_action_combination.append(final_cost)
        joint_action = actions_combination[np.argmin(cost_of_action_combination)]
        return joint_action
