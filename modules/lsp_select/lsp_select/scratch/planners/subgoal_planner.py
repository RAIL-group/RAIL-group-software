import copy
import torch
import numpy as np
import gridmap
import lsp
import lsp_select
from lsp.planners.subgoal_planner import BaseSubgoalPlanner
from lsp.planners.subgoal_planner import LearnedSubgoalPlanner as LSP
from lsp.planners import Planner, DijkstraPlanner
from sklearn.metrics import log_loss
from lsp_select.utils import delta_cost, simulators
from lsp.planners.subgoal_planner import NUM_MAX_FRONTIERS


class LearnedSubgoalPlannerLaser(BaseSubgoalPlanner):

    def __init__(self, goal, args, device=None, verbose=False):
        super(LearnedSubgoalPlannerLaser, self).__init__(goal, args)

        if device is not None:
            self.device = device
        else:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")

        self.subgoal_property_net = lsp_select.learning.models.LaserLSP.get_net_eval_fn(
            args.network_file, device=self.device)

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
        self._update_subgoal_inputs(observation['ranges'], robot_pose, self.goal)

        # Once the subgoal inputs are set, compute their properties
        self._update_subgoal_properties(robot_pose, self.goal)

    def _update_subgoal_inputs(self, ranges, robot_pose, goal_pose):
        # Loop through subgoals and get the 'input data'
        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue

            # Compute the data that will be passed to the neural net
            input_data = lsp_select.utils.learning_laser.get_frontier_data_vector(
                ranges[::4] * self.args.base_resolution, robot_pose, goal_pose, subgoal)

            # Store the input data alongside each subgoal
            subgoal.nn_input_data = input_data

    def _update_subgoal_properties(self, robot_pose, goal_pose):

        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue

            [prob_feasible, delta_success_cost, exploration_cost] = \
                self.subgoal_property_net(subgoal.nn_input_data)
            delta_success_cost = 0 if delta_success_cost < 0 else delta_success_cost
            exploration_cost = 0 if exploration_cost < 0 else exploration_cost
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


class LearnedSubgoalPlanner(BaseSubgoalPlanner):

    def __init__(self, goal, args, device=None, verbose=False):
        super(LearnedSubgoalPlanner, self).__init__(goal, args)

        if device is not None:
            self.device = device
        else:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")

        self.subgoal_property_net = lsp.learning.models.VisLSPOriented.get_net_eval_fn_std(
            args.network_file, device=self.device)

    def _update_subgoal_properties(self, robot_pose, goal_pose):

        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue

            [prob_feasible, delta_success_cost, exploration_cost, delta_success_cost_std, exploration_cost_std] = \
                self.subgoal_property_net(subgoal.nn_input_data)

            subgoal.set_props(prob_feasible=prob_feasible,
                              delta_success_cost=delta_success_cost,
                              exploration_cost=exploration_cost,
                              delta_success_cost_std=delta_success_cost_std,
                              exploration_cost_std=exploration_cost_std,
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

        self.expected_cost = min_cost
        self.latest_ordering = frontier_ordering
        self.selected_subgoal = frontier_ordering[0]
        return self.selected_subgoal


class LSPCycleGAN(LSP):

    def __init__(self, goal, args, device=None, verbose=False):
        super(LSPCycleGAN, self).__init__(goal, args)

        self.cyclegan_generator = lsp_select.learning.models.cyclegan.ResnetGenerator.get_generator_fn(
            args.generator_network_file, self.device)

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

        if not self.args.disable_cyclegan:
            observation['image'] = self.cyclegan_generator(observation['image'])
            self.observation = observation

        # Update the subgoal inputs
        self._update_subgoal_inputs(observation['image'], robot_pose, self.goal)

        # Once the subgoal inputs are set, compute their properties
        self._update_subgoal_properties(robot_pose, self.goal)


class RuntimeMonitoringPlanner(Planner):

    def __init__(self, goal, planners, priors, threshold, args):
        super(RuntimeMonitoringPlanner, self).__init__(goal)
        self.args = args
        self.default_planner = DijkstraPlanner(goal=goal, args=args)
        self.planners = planners
        self.priors = np.array(priors)
        self.threshold = threshold
        self.chosen_planner_idx = None

        # avg cross entropy = total_xentropy / number of observations
        self.xentropy_avg = self.priors[:, 0] / self.priors[:, 1]
        lowest_idx = np.argmin(self.xentropy_avg)
        if self.xentropy_avg[lowest_idx] < self.threshold:
            self.chosen_planner_idx = lowest_idx
        self.subgoals_data = [[] for _ in self.planners]
        if self.chosen_planner_idx is not None:
            self.chosen_planner_name = type(self.planners[self.chosen_planner_idx]).__name__
        else:
            self.chosen_planner_name = type(self.default_planner).__name__

    def update(self, observation, observed_map, subgoals, robot_pose, visibility_mask):
        self.robot_pose = robot_pose
        self.observation = observation
        self.observed_map = observed_map
        self.default_planner.update(observation, observed_map, subgoals, robot_pose, visibility_mask)
        for i, planner in enumerate(self.planners):
            planner.update(observation, observed_map, subgoals, robot_pose, visibility_mask)
            is_goal_in_range = lsp.core.goal_in_range(planner.inflated_grid, robot_pose, self.goal, subgoals)
            if not is_goal_in_range:
                self.subgoals_data[i].append({'subgoals': planner.subgoals,
                                              'robot_pose': robot_pose})
        self.inflated_grid = self.planners[0].inflated_grid
        if self.chosen_planner_idx is not None:
            self.subgoals = self.planners[self.chosen_planner_idx].subgoals
        else:
            self.subgoals = self.default_planner.subgoals

    def compute_selected_subgoal(self):
        if self.chosen_planner_idx is not None:
            return self.planners[self.chosen_planner_idx].compute_selected_subgoal()
        return self.default_planner.compute_selected_subgoal()

    def compute_new_priors(self):
        inflated_grid = self.planners[0].inflated_grid
        subgoals = self.planners[0].subgoals
        planning_grid = lsp.core.mask_grid_with_frontiers(inflated_grid, subgoals)
        for i in range(len(self.planners)):
            if len(self.subgoals_data[i]) == 0:
                break
            subgoal_labels = lsp_select.utils.misc.get_subgoal_labels(self.subgoals_data[i], planning_grid, self.goal)
            xentropy_avg = log_loss(subgoal_labels[:, 2], subgoal_labels[:, 1], labels=[0, 1])
            no_of_obs = len(subgoal_labels)
            self.priors[i] += [xentropy_avg * no_of_obs, no_of_obs]
        return self.priors


class RTMDelta(Planner):

    def __init__(self, goal, planners, priors, args):
        super(RTMDelta, self).__init__(goal)
        self.args = args
        self.planners = [DijkstraPlanner(goal=goal, args=args)] + planners
        if priors is None:
            self.priors = np.zeros((len(self.planners), 2))
            self.avg_estimated_costs = np.zeros(len(self.planners))
            self.chosen_planner_idx = 0
        else:
            self.priors = np.array(priors)
            self.avg_estimated_costs = self.priors[:, 0] / self.priors[:, 1]
            self.chosen_planner_idx = np.argmin(self.avg_estimated_costs)

        self.navigation_data = {'policies': [[] for _ in self.planners],
                                'steps': [],
                                'spidx': self.chosen_planner_idx,
                                'start_pose': self.args.robot_pose,
                                'goal_pose': self.goal,
                                'net_motion': None,
                                'correct_subgoals': [],
                                'final_masked_grid': None,
                                'known_path': [],
                                'known_cost': None}
        self.chosen_planner_name = type(self.planners[self.chosen_planner_idx]).__name__

    def update(self, observation, observed_map, subgoals, robot_pose, visibility_mask):
        self.robot_pose = robot_pose
        self.observation = observation
        self.observed_map = observed_map
        for planner in self.planners:
            planner.update(observation, observed_map, subgoals, robot_pose, visibility_mask)
        self.inflated_grid = self.planners[self.chosen_planner_idx].inflated_grid
        self.subgoals = self.planners[self.chosen_planner_idx].subgoals

    def compute_selected_subgoal(self):
        chosen_subgoal = None
        for i, planner in enumerate(self.planners):
            is_goal_in_range = lsp.core.goal_in_range(planner.inflated_grid, self.robot_pose, self.goal, self.subgoals)
            if not is_goal_in_range:
                selected_subgoal = planner.compute_selected_subgoal()
                if i == self.chosen_planner_idx:
                    chosen_subgoal = selected_subgoal
                planner.latest_ordering = delta_cost.get_full_policy(planner.latest_ordering,
                                                                     planner.subgoals,
                                                                     planner.inflated_grid,
                                                                     planner.robot_pose,
                                                                     planner.goal)
                self.navigation_data['policies'][i].append(planner.latest_ordering)
        self.navigation_data['steps'].append([self.robot_pose,
                                              self.subgoals,
                                              self.inflated_grid,
                                              is_goal_in_range])
        return chosen_subgoal

    def compute_new_priors(self):
        step_data = self.navigation_data['steps']
        pose = self.navigation_data['start_pose']
        goal = self.navigation_data['goal_pose']
        final_subgoals = step_data[-1][1]
        final_masked_grid = lsp.core.mask_grid_with_frontiers(step_data[-1][2], final_subgoals)
        known_path = delta_cost.get_path_in_known_grid(final_masked_grid,
                                                       pose,
                                                       goal,
                                                       final_subgoals,
                                                       do_not_mask=None)
        known_cost = delta_cost.compute_cost_from_path(known_path)
        correct_subgoals = []
        for i, (robot_pose, subgoals, inflated_grid, is_goal_in_range) in enumerate(step_data):
            if is_goal_in_range:
                break
            for subgoal in subgoals:
                if lsp_select.utils.misc.is_feasible_subgoal(subgoal, final_masked_grid, subgoals, robot_pose, goal):
                    correct_subgoals.append(subgoal)
                    break
        self.navigation_data['correct_subgoals'] = correct_subgoals
        self.navigation_data['final_masked_grid'] = final_masked_grid
        self.navigation_data['net_motion'] = self.args.robot.net_motion
        self.navigation_data['known_path'] = known_path
        self.navigation_data['known_cost'] = known_cost

        delta_costs_array = np.zeros((len(self.planners),
                                      len(self.navigation_data['policies'][self.chosen_planner_idx])))
        estimated_lb_costs = np.zeros(len(self.planners))
        for i in range(len(self.planners)):
            if i == self.chosen_planner_idx:
                estimated_lb_costs[i] = self.navigation_data['net_motion']
                continue
            delta_costs_array[i, :] = delta_cost.eval_alternate_policies(self.navigation_data['steps'],
                                                                         self.navigation_data['correct_subgoals'],
                                                                         self.navigation_data['policies'][i],
                                                                         self.navigation_data['start_pose'],
                                                                         self.navigation_data['goal_pose'])
            estimated_lb_costs[i] = (delta_cost.aggregate_delta_costs(delta_costs_array[i, :])
                                     + self.navigation_data['known_cost'])
        priors = np.zeros_like(self.priors)
        priors[:, 0] = self.priors[:, 0] + estimated_lb_costs
        priors[:, 1] = self.priors[:, 1] + 1
        return priors


class RTMLowerBound(Planner):

    def __init__(self, goal, planners, priors, args):
        super(RTMLowerBound, self).__init__(goal)
        self.args = args
        self.planners = [DijkstraPlanner(goal=goal, args=args)] + planners
        self.poses = []
        self.images = []
        self.counter = 0

        if priors is None:
            self.priors = np.zeros((len(self.planners), 2))
            self.avg_estimated_costs = np.zeros(len(self.planners))
            self.chosen_planner_idx = 0
        else:
            self.priors = np.array(priors)
            self.avg_estimated_costs = self.priors[:, 0] / self.priors[:, 1]
            self.chosen_planner_idx = np.argmin(self.avg_estimated_costs)

        self.nearest_pose_data = np.array([[{'index': None, 'distance': np.inf} for _ in range(args.map_shape[1])]
                                           for _ in range(args.map_shape[0])])
        self.navigation_data = {'start': self.args.robot_pose,
                                'goal': self.goal,
                                'planner': type(self.planners[self.chosen_planner_idx]).__name__}
        self.chosen_planner_name = type(self.planners[self.chosen_planner_idx]).__name__

    def update(self, observation, observed_map, subgoals, robot_pose, visibility_mask):
        self.robot_pose = robot_pose
        self.observation = observation
        self.observed_map = observed_map
        self.planners[self.chosen_planner_idx].update(observation, observed_map, subgoals, robot_pose, visibility_mask)
        self.inflated_grid = self.planners[self.chosen_planner_idx].inflated_grid
        self.subgoals = self.planners[self.chosen_planner_idx].subgoals

        pose = [robot_pose.x, robot_pose.y, robot_pose.yaw]
        self.poses.append(pose)
        self.images.append(observation['image'])

        self.update_nearest_pose_data(visibility_mask, [pose[0], pose[1]])
        self.counter += 1

    def compute_selected_subgoal(self):
        return self.planners[self.chosen_planner_idx].compute_selected_subgoal()

    def compute_new_priors(self):
        self.navigation_data['poses'] = np.array(self.poses),
        self.navigation_data['images'] = self.images,
        self.navigation_data['nearest_pose_data'] = self.nearest_pose_data
        self.navigation_data['partial_map'] = self.observed_map
        self.navigation_data['final_subgoals'] = self.subgoals
        self.navigation_data['net_motion'] = self.args.robot.net_motion
        self.navigation_data['robot_path'] = self.args.robot.all_poses

        priors = np.zeros_like(self.priors)
        for i, planner in enumerate(self.planners):
            if i == self.chosen_planner_idx:
                cost = self.navigation_data['net_motion']
            else:
                cost = simulators.get_simulated_planner_cost(self.navigation_data, planner, self.args)
            priors[i, 0] = self.priors[i, 0] + cost
            priors[i, 1] = self.priors[i, 1] + 1
        return priors

    def update_nearest_pose_data(self, visibility_mask, current_pose):
        visible_cells = np.where(visibility_mask == 1)
        poses = np.column_stack(visible_cells)
        cost_grid, _ = gridmap.planning.compute_cost_grid_from_position(visibility_mask != 1,
                                                                        current_pose,
                                                                        use_soft_cost=True)
        for x, y in poses:
            distance = cost_grid[x, y]
            pose_data = self.nearest_pose_data[x, y]
            if pose_data['index'] is None:
                pose_data['index'] = self.counter
                pose_data['distance'] = distance
                continue
            if pose_data['distance'] > distance:
                pose_data['index'] = self.counter
                pose_data['distance'] = distance


class RTMSimulate(RTMLowerBound):
    def __init__(self, goal, planners, chosen_planner_idx, args):
        super(RTMLowerBound, self).__init__(goal)
        self.args = args
        self.planners = planners
        self.poses = []
        self.images = []
        self.counter = 0
        self.chosen_planner_idx = chosen_planner_idx
        self.nearest_pose_data = np.array([[{'index': None, 'distance': np.inf} for _ in range(args.map_shape[1])]
                                           for _ in range(args.map_shape[0])])
        self.navigation_data = {'start': self.args.robot_pose,
                                'goal': self.goal,
                                'planner': type(self.planners[self.chosen_planner_idx]).__name__}
        self.chosen_planner_name = type(self.planners[self.chosen_planner_idx]).__name__

    def get_costs(self):
        self.navigation_data['poses'] = np.array(self.poses),
        self.navigation_data['images'] = self.images,
        self.navigation_data['nearest_pose_data'] = self.nearest_pose_data
        self.navigation_data['partial_map'] = self.observed_map
        self.navigation_data['final_subgoals'] = self.subgoals
        self.navigation_data['net_motion'] = self.args.robot.net_motion
        self.navigation_data['robot_path'] = self.args.robot.all_poses

        planner_costs = np.zeros(len(self.planners))
        for i, planner in enumerate(self.planners):
            if i == self.chosen_planner_idx:
                cost = self.navigation_data['net_motion']
            else:
                cost = simulators.get_simulated_planner_cost(self.navigation_data, planner, self.args)
            planner_costs[i] = cost
        return planner_costs


class RTMSimulateLB(RTMSimulate):
    def get_costs(self):
        self.navigation_data['poses'] = np.array(self.poses),
        self.navigation_data['images'] = self.images,
        self.navigation_data['nearest_pose_data'] = self.nearest_pose_data
        self.navigation_data['partial_map'] = self.observed_map
        self.navigation_data['final_subgoals'] = self.subgoals
        self.navigation_data['net_motion'] = self.args.robot.net_motion
        self.navigation_data['robot_path'] = self.args.robot.all_poses

        alt_costs_data = {}
        planner_costs = np.zeros(len(self.planners))
        for i, planner in enumerate(self.planners):
            if i == self.chosen_planner_idx:
                cost = self.navigation_data['net_motion']
                alt_costs_data[i] = None
            else:
                all_alt_costs, cost = simulators.get_lowerbound_planner_cost(self.navigation_data, planner, self.args)
                alt_costs_data[i] = all_alt_costs
            planner_costs[i] = cost
        return planner_costs, alt_costs_data
