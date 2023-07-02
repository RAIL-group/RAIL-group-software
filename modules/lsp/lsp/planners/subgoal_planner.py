import copy
import gridmap
import lsp
import numpy as np
import torch

from .planner import Planner

NUM_MAX_FRONTIERS = 12


class BaseSubgoalPlanner(Planner):
    def __init__(self, goal, args, verbose=False):
        super(BaseSubgoalPlanner, self).__init__(goal)
        self.subgoals = set()
        self.selected_subgoal = None
        self.observed_map = None
        self.args = args
        self.verbose = verbose

        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        if self.inflation_radius >= np.sqrt(5):
            self.downsample_factor = 2
        else:
            self.downsample_factor = 1

        self.update_counter = 0

    def update(self, observation, observed_map, subgoals, robot_pose,
               visibility_mask):
        """Updates the internal state with the new grid/pose/laser scan.

        This function also computes a few necessary items, like which
        frontiers have recently been updated and computes their properties
        from the known grid.
        """
        self.update_counter += 1
        self.observation = observation
        self.observed_map = observed_map
        self.robot_pose = robot_pose

        # Store the inflated grid after ensuring that the unreachable 'free
        # space' is set to 'unobserved'. This avoids trying to plan to
        # unreachable space and avoids having to check for this everywhere.
        inflated_grid = self._get_inflated_occupancy_grid()
        self.inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
            inflated_grid, robot_pose)

        # Compute the new frontiers and update stored frontiers
        new_subgoals = set([copy.copy(s) for s in subgoals])
        self.subgoals = lsp.core.update_frontier_set(
            self.subgoals,
            new_subgoals,
            max_dist=2.0 / self.args.base_resolution,
            chosen_frontier=self.selected_subgoal)

        # Also check that the goal is not inside the frontier
        lsp.core.update_frontiers_goal_in_frontier(self.subgoals, self.goal)

        # Update the subgoal inputs
        self._update_subgoal_inputs(observation['image'], robot_pose, self.goal)

        # Once the subgoal inputs are set, compute their properties
        self._update_subgoal_properties(robot_pose, self.goal)

    def _update_subgoal_inputs(self, image, robot_pose, goal_pose):
        # Loop through subgoals and get the 'input data'
        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue

            # Compute the data that will be passed to the neural net
            input_data = lsp.utils.learning_vision.get_oriented_input_data(
                image, robot_pose, goal_pose, subgoal)

            # Store the input data alongside each subgoal
            subgoal.nn_input_data = input_data

    def _update_subgoal_properties(self, robot_pose, goal_pose):
        raise NotImplementedError("Method for abstract class")


class KnownSubgoalPlanner(BaseSubgoalPlanner):

    def __init__(self, goal, known_map, args, verbose=False, do_compute_weightings=False):
        super(KnownSubgoalPlanner, self).__init__(goal, args, verbose)

        self.known_map = known_map
        self.inflated_known_grid = gridmap.utils.inflate_grid(
            known_map, inflation_radius=self.inflation_radius)
        self.do_compute_weightings = do_compute_weightings

        # Compute cost grid
        _, self.get_path = gridmap.planning.compute_cost_grid_from_position(
            self.inflated_known_grid, [goal.x, goal.y], use_soft_cost=True)

    def _update_subgoal_properties(self, robot_pose, goal_pose):
        new_subgoals = [s for s in self.subgoals if not s.props_set]
        lsp.core.update_frontiers_properties_known(self.inflated_known_grid,
                                                   self.inflated_grid,
                                                   self.subgoals, new_subgoals,
                                                   robot_pose, goal_pose,
                                                   self.downsample_factor)

        if self.do_compute_weightings:
            lsp.core.update_frontiers_weights_known(self.inflated_known_grid,
                                                    self.inflated_grid,
                                                    self.subgoals, new_subgoals,
                                                    robot_pose, goal_pose,
                                                    self.downsample_factor)

        self.updated_subgoals = [subgoal for subgoal in new_subgoals
                                 if not subgoal.is_obstructed]

        if self.verbose:
            for subgoal in self.updated_subgoals:
                lsp.utils.command_line.print_frontier_data(
                    subgoal, num_leading_spaces=16, print_weights=True)

    def get_subgoal_training_data(self):
        data = []
        for subgoal in self.updated_subgoals:
            datum = subgoal.nn_input_data
            datum['is_feasible'] = subgoal.prob_feasible
            datum['delta_success_cost'] = subgoal.delta_success_cost
            datum['exploration_cost'] = subgoal.exploration_cost
            datum['positive_weighting'] = subgoal.positive_weighting
            datum['negative_weighting'] = subgoal.negative_weighting
            data.append(datum)

        return data

    def compute_selected_subgoal(self):
        """Use the known map to compute the selected subgoal."""

        if not self.subgoals:
            return None

        # Compute the plan
        did_plan, path = self.get_path([self.robot_pose.x, self.robot_pose.y],
                                       do_sparsify=False,
                                       do_flip=True,
                                       bound=None)
        if did_plan is False:
            print("Plan did not succeed...")
            raise NotImplementedError("Not sure what to do here yet")
        if np.argmax(self.observed_map[path[0, -1], path[1, -1]] >= 0):
            return None

        # Determine the chosen subgoal
        ind = np.argmax(self.observed_map[path[0, :], path[1, :]] < 0)
        return min(self.subgoals,
                   key=lambda s: s.get_distance_to_point((path.T)[ind]))


class LearnedSubgoalPlanner(BaseSubgoalPlanner):

    def __init__(self, goal, args, device=None, verbose=False):
        super(LearnedSubgoalPlanner, self).__init__(goal, args)

        if device is not None:
            self.device = device
        else:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")

        self.subgoal_property_net = lsp.learning.models.VisLSPOriented.get_net_eval_fn(
            args.network_file, device=self.device)

    def _update_subgoal_properties(self, robot_pose, goal_pose):

        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue

            [prob_feasible, delta_success_cost, exploration_cost] = \
                self.subgoal_property_net(subgoal.nn_input_data)

            subgoal.set_props(prob_feasible=prob_feasible,
                              delta_success_cost=delta_success_cost,
                              exploration_cost=exploration_cost,
                              last_observed_pose=robot_pose)

        for subgoal in self.subgoals:
            if not self.args.silence and subgoal.prob_feasible > 0.0:
                print(" " * 20 + "PLAN  (%.2f %.2f) | %.6f | %7.2f | %7.2f" %
                      (subgoal.get_centroid()[0], subgoal.get_centroid()[1],
                       subgoal.prob_feasible, subgoal.delta_success_cost,
                       subgoal.exploration_cost))

    def compute_selected_subgoal(self):
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose, self.goal,
                                                  self.subgoals)

        if is_goal_in_range:
            print("Goal in Range")
            return None

        # Compute chosen frontier
        min_cost, frontier_ordering = (
            lsp.core.get_best_expected_cost_and_frontier_list(
                self.inflated_grid,
                self.robot_pose,
                self.goal,
                self.subgoals,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                downsample_factor=self.downsample_factor,
                do_correct_low_prob=True))
        if min_cost is None or min_cost > 1e8 or frontier_ordering is None:
            print("Problem with planning.")
            self.latest_ordering = None
            self.selected_subgoal = None
            return None

        self.latest_ordering = frontier_ordering
        self.selected_subgoal = frontier_ordering[0]
        return self.selected_subgoal
