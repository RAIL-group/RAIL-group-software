import numpy as np
from .planner import MRPlanner
from mrlsp.utils.utility import get_multirobot_distances, find_action_list_from_cost_matrix_using_lsa


class OptimisticPlanner(MRPlanner):
    def __init__(self, robots, goal, args):
        super(OptimisticPlanner, self).__init__(robots, goal)
        self.selected_subgoal = None
        self.observed_map = None
        self.args = args

    def compute_selected_subgoal(self):
        '''TODO: Right now, goal pose is sent as list: Just because every other function use list of goalsl,
        The functionality that is thought to be extended, where goal is multiple'''
        distances_mr = get_multirobot_distances(self.inflated_grid, self.robots, [self.goal_pose], self.subgoals)
        # find cost for robot to reach goal from all the subgoal, and create a cost matrix.
        cost_dictionary = [None for _ in range(len(self.robots))]
        for i in range(len(self.robots)):
            cost_dictionary[i] = {
                subgoal: distances_mr[f'robot{i+1}'][subgoal] + distances_mr['goal'][subgoal]
                for subgoal in self.subgoals
            }
        subgoal_matrix = np.array([list(cd.keys()) for cd in cost_dictionary])
        cost_matrix = np.array([list(cd.values()) for cd in cost_dictionary])
        # from the cost matrix, return the list of subgoals that has the least cost.
        joint_action = find_action_list_from_cost_matrix_using_lsa(cost_matrix, subgoal_matrix)

        return joint_action
