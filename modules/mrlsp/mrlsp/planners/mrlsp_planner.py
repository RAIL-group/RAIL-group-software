from .planner import BaseLSPPlanner
import lsp
import mrlsp
import mrlsp_accel
import itertools
import numpy as np


class MRLearnedSubgoalPlanner(BaseLSPPlanner):
    '''Multi-robot Learned Subgoal Planner'''

    def __init__(self, robots, goal, args):
        super(MRLearnedSubgoalPlanner, self).__init__(robots, goal, args)

    def compute_selected_subgoal_old(self):
        # If goal in range, return None as action
        if lsp.core.goal_in_range(grid=self.robot_grid,
                                  robot_pose=None, goal_pose=self.goal_pose, frontiers=self.subgoals):
            joint_action = [None for i in range(len(self.robots))]
            return joint_action

        joint_action = mrlsp.pouct.find_best_joint_action(
            self.subgoals, self.distances_mr, num_robots=len(self.robots), num_iterations=self.args.iterations)

        return joint_action

    def compute_selected_subgoal(self):

        # If goal in range, return None as action
        if lsp.core.goal_in_range(grid=self.robot_grid,
                                  robot_pose=None, goal_pose=self.goal_pose, frontiers=self.subgoals):
            joint_action = [None for i in range(len(self.robots))]
            return joint_action

        # for cpp
        num_robots = len(self.robots)
        unexplored_frontiers = list(self.subgoals)
        s_dict = {hash(s): s for s in unexplored_frontiers}
        s_cpp = [
            mrlsp_accel.FrontierDataMR(s.prob_feasible, s.delta_success_cost,
                                       s.exploration_cost, hash(s),
                                       s.is_from_last_chosen) for s in unexplored_frontiers
        ]
        rd_cpp = {(i, hash(s)): self.distances_mr[f'robot{i+1}'][s]
                  for i in range(num_robots) for s in unexplored_frontiers}
        gd_cpp = {hash(s): self.distances_mr['goal'][s] for s in unexplored_frontiers}
        fd_cpp = {(hash(sp[0]), hash(sp[1])): self.distances_mr['frontier'][frozenset(sp)]
                  for sp in itertools.permutations(unexplored_frontiers, 2)}

        joint_action_hash = mrlsp_accel.find_best_joint_action_accel(
            num_robots, s_cpp, rd_cpp, gd_cpp, fd_cpp, self.args.iterations)
        joint_action = [s_dict[s] for s in joint_action_hash]
        # ''' Use LSA here '''
        num_robots = len(self.robots)
        cost_dictionary = [None for _ in range(num_robots)]
        for i in range(num_robots):
            cost_dictionary[i] = {
                subgoal: self.distances_mr[f'robot{i+1}'][subgoal] + self.distances_mr['goal'][subgoal]
                for subgoal in joint_action
            }
        subgoal_matrix = np.array([list(cd.keys()) for cd in cost_dictionary])
        cost_matrix = np.array([list(cd.values()) for cd in cost_dictionary])
        # from the cost matrix, return the list of subgoals that has the least cost.
        action = mrlsp.utils.utility.find_action_list_from_cost_matrix_using_lsa(cost_matrix, subgoal_matrix)
        return action
