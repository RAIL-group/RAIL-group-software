import copy
from lsp.core import get_goal_distances, get_frontier_distances, get_robot_distances, get_top_n_frontiers
import lsp
import gridmap
import numpy as np
import logging
import time
import torch
from .explanation import Explanation
import lsp_xai
import lsp_xai.utils.plotting
from lsp_xai.learning.models import ExpNavVisLSP

NUM_MAX_FRONTIERS = 12


class SubgoalPlanner(lsp.planners.Planner):
    def __init__(self, goal, args, device=None):
        super(SubgoalPlanner, self).__init__(goal)

        self.subgoals = set()
        self.selected_subgoal = None
        self.observed_map = None
        self.args = args
        self.update_counter = 0

        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        if self.inflation_radius >= np.sqrt(5):
            self.downsample_factor = 2
        else:
            self.downsample_factor = 1

        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device

        self.subgoal_property_net, self.model = ExpNavVisLSP.get_net_eval_fn(
            args.network_file, device=self.device, do_return_model=True)

    def get_planner_state(self):
        return {
            'args': self.args,
            'goal': self.goal,
            'observed_map': self.observed_map,
            'inflated_grid': self.inflated_grid,
            'subgoals': self.subgoals,
            'robot_pose': self.robot_pose,
            'update_counter': self.update_counter
        }

    def update(self, observation, observed_map, subgoals, robot_pose,
               visibility_mask):
        """Updates the internal state with the new grid/pose/laser scan.

        This function also computes a few necessary items, like the new
        set of frontiers from the inflated grid and their properties from
        the trained network.
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
            max_dist=20.0 / self.args.base_resolution,
            chosen_frontier=self.selected_subgoal)

        # Also check that the goal is not inside the frontier
        lsp.core.update_frontiers_goal_in_frontier(self.subgoals, self.goal)

        self._update_frontier_props_oriented(robot_pose=robot_pose,
                                             goal_pose=self.goal,
                                             image=observation['image'],
                                             visibility_mask=visibility_mask)

    def _update_frontier_props_oriented(self,
                                        robot_pose,
                                        goal_pose,
                                        image,
                                        visibility_mask=None):
        if image is None:
            raise ValueError("argument 'image' must not be 'None'")

        image = image * 1.0 / 255.0

        # Loop through subgoals and set properties
        for subgoal in self.subgoals:
            if subgoal.props_set:
                continue

            # Compute the data that will be passed to the neural net
            input_data = lsp.utils.learning_vision.get_oriented_input_data(
                image, robot_pose, goal_pose, subgoal)

            # Store the input data alongside each subgoal
            subgoal.nn_input_data = input_data

            # Compute subgoal properties from neural network
            [prob_feasible, delta_success_cost, exploration_cost] = \
                self.subgoal_property_net(
                    image=input_data['image'],
                    goal_loc_x=input_data['goal_loc_x'],
                    goal_loc_y=input_data['goal_loc_y'],
                    subgoal_loc_x=input_data['subgoal_loc_x'],
                    subgoal_loc_y=input_data['subgoal_loc_y'])
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

    def _recompute_all_subgoal_properties(self):
        # Loop through subgoals and set properties
        for subgoal in self.subgoals:
            input_data = subgoal.nn_input_data

            # Compute subgoal properties from neural network
            [prob_feasible, delta_success_cost, exploration_cost] = \
                self.subgoal_property_net(
                    image=input_data['image'],
                    goal_loc_x=input_data['goal_loc_x'],
                    goal_loc_y=input_data['goal_loc_y'],
                    subgoal_loc_x=input_data['subgoal_loc_x'],
                    subgoal_loc_y=input_data['subgoal_loc_y'])
            subgoal.set_props(prob_feasible=prob_feasible,
                              delta_success_cost=delta_success_cost,
                              exploration_cost=exploration_cost,
                              last_observed_pose=subgoal.last_observed_pose)

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
            raise ValueError("Problem with planning")

        self.latest_ordering = frontier_ordering
        self.selected_subgoal = frontier_ordering[0]
        return self.selected_subgoal

    def compute_backup_subgoal(self, selected_subgoal):
        subgoals, distances = self.get_subgoals_and_distances()
        return lsp.core.get_lowest_cost_ordering_not_beginning_with(
            selected_subgoal, subgoals, distances)[1][0]

    def compute_subgoal_data(self,
                             chosen_subgoal,
                             num_frontiers_max=NUM_MAX_FRONTIERS,
                             do_return_ind_dict=False):
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose, self.goal,
                                                  self.subgoals)
        if is_goal_in_range:
            return None

        # Compute chosen frontier
        logger = logging.getLogger("SubgoalPlanner")
        stime = time.time()
        policy_data, subgoal_ind_dict = get_policy_data_for_frontiers(
            self.inflated_grid,
            self.robot_pose,
            self.goal,
            chosen_subgoal,
            self.subgoals,
            num_frontiers_max=num_frontiers_max,
            downsample_factor=self.downsample_factor)
        logger.debug(f"time to get policy data: {time.time() - stime}")

        if do_return_ind_dict:
            return policy_data, subgoal_ind_dict
        else:
            return policy_data

    def get_subgoals_and_distances(self, subgoals_of_interest=[]):
        """Helper function for getting data."""
        # Remove frontiers that are infeasible
        subgoals = [s for s in self.subgoals if s.prob_feasible > 0]
        subgoals = list(set(subgoals) | set(subgoals_of_interest))

        # Calculate the distance to the goal and to the robot.
        goal_distances = get_goal_distances(
            self.inflated_grid,
            self.goal,
            frontiers=subgoals,
            downsample_factor=self.downsample_factor)

        robot_distances = get_robot_distances(
            self.inflated_grid,
            self.robot_pose,
            frontiers=subgoals,
            downsample_factor=self.downsample_factor)

        # Get the most n probable frontiers to limit computational load
        if NUM_MAX_FRONTIERS > 0 and NUM_MAX_FRONTIERS < len(subgoals):
            subgoals = get_top_n_frontiers(subgoals, goal_distances,
                                           robot_distances, NUM_MAX_FRONTIERS)
            subgoals = list(set(subgoals) | set(subgoals_of_interest))

        # Calculate robot and frontier distances
        frontier_distances = get_frontier_distances(
            self.inflated_grid,
            frontiers=subgoals,
            downsample_factor=self.downsample_factor)

        distances = {
            'frontier': frontier_distances,
            'robot': robot_distances,
            'goal': goal_distances,
        }

        return subgoals, distances

    def generate_counterfactual_explanation(self,
                                            query_subgoal,
                                            limit_num=-1,
                                            do_freeze_selected=True,
                                            keep_changes=False,
                                            margin=0,
                                            learning_rate=1.0e-4):
        # Initialize the datum
        device = self.device
        chosen_subgoal = self.compute_selected_subgoal()
        datum, subgoal_ind_dict = self.compute_subgoal_data(
            chosen_subgoal, 24, do_return_ind_dict=True)
        datum = self.model.update_datum(datum, device)

        # Now we want to rearrange things a bit: the new 'target' subgoal we set to
        # our query_subgoal and we populate the 'backup'
        # subgoal with the 'chosen' subgoal (the subgoal the agent actually chose).
        datum['target_subgoal_ind'] = subgoal_ind_dict[query_subgoal]
        if do_freeze_selected:
            datum['backup_subgoal_ind'] = subgoal_ind_dict[chosen_subgoal]

        # We update the datum to reflect this change (and confirm it worked).
        datum = self.model.update_datum(datum, device)

        # Compute the 'delta subgoal data'. This is how we determine the
        # 'importance' of each of the subgoal properties. In practice, we will sever
        # the gradient for all but a handful of these with an optional parameter
        # (not set here).
        base_model_state = self.model.state_dict(keep_vars=False)
        base_model_state = copy.deepcopy(base_model_state)
        base_model_state = {k: v.cpu() for k, v in base_model_state.items()}

        updated_datum, delta_subgoal_data, base_subgoal_props, updated_subgoal_props = (
            self.model.update_model_counterfactual(datum, limit_num,
                                                   margin, learning_rate,
                                                   self.device))

        # Restore the model to its previous value
        if not keep_changes:
            print("Restoring Model")
            self.model.load_state_dict(base_model_state)
            self.model.eval()
            self.model = self.model.to(device)
        else:
            print("Keeping model")
            self._recompute_all_subgoal_properties()

        return Explanation(self.subgoals, subgoal_ind_dict, datum,
                           base_subgoal_props, updated_datum, updated_subgoal_props,
                           delta_subgoal_data, self.observed_map,
                           self.inflated_grid, self.goal, self.robot_pose,
                           limit_num)

    def plot_map_with_plan(self, ax=None, robot_poses=None, image=None,
                           query_subgoal=None, datum=None, subgoal_props=None, subgoal_ind_dict=None):
        import matplotlib.pyplot as plt
        ax_img = plt.subplot(121)
        ax_img.axes.xaxis.set_visible(False)
        ax_img.axes.yaxis.set_visible(False)

        # Initialize the datum
        device = self.device
        chosen_subgoal = self.compute_selected_subgoal()

        if chosen_subgoal is None:
            lsp_xai.utils.plotting.plot_map(
                ax_img, self, robot_poses=robot_poses)
            return

        if datum is None or subgoal_props is None:
            datum, subgoal_ind_dict = self.compute_subgoal_data(
                chosen_subgoal, 24, do_return_ind_dict=True)
            datum = self.model.update_datum(datum, device)
            delta_subgoal_data = self.model.get_subgoal_prop_impact(
                datum, device, delta_cost_limit=-1e10)

            # Compute the subgoal props
            nn_out, ind_mapping = self.model(datum, device)
            is_feasibles = torch.nn.Sigmoid()(nn_out[:, 0])
            delta_success_costs = nn_out[:, 1]
            exploration_costs = nn_out[:, 2]
            subgoal_props, _, _ = self.model.compute_subgoal_props(
                is_feasibles,
                delta_success_costs,
                exploration_costs,
                datum['subgoal_data'],
                ind_mapping,
                device,
                limit_subgoals_num=0,
                delta_subgoal_data=delta_subgoal_data)

        policy = datum['target_subgoal_policy']['policy']

        lsp_xai.utils.plotting.plot_map_with_plan(
            ax_img, self, subgoal_ind_dict, policy, subgoal_props,
            robot_poses=robot_poses)

        # Plot the onboard image
        if image is not None:
            ax = plt.subplot(3, 2, 2)
            ax.imshow(image)
            ax.set_title('Onboard Image')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        # Plt the chosen subgoal
        ax = plt.subplot(3, 2, 4)
        chosen_subgoal_ind = subgoal_ind_dict[chosen_subgoal]
        ax.imshow(datum['subgoal_data'][chosen_subgoal_ind]['image'])
        pf_chosen = subgoal_props[chosen_subgoal_ind].prob_feasible
        ax.set_title(f'Subgoal 0: $P_S$ = {pf_chosen*100:.1f}\\%')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.plot([0.5, 0.5], [1.0, 0.0],
                transform=ax.transAxes,
                color=[0, 0, 1],
                alpha=0.3)

        # Plot the query/backup subgoal
        ax = plt.subplot(3, 2, 6)
        if query_subgoal is None:
            query_subgoal = self.compute_backup_subgoal(chosen_subgoal)
        query_subgoal_ind = subgoal_ind_dict[query_subgoal]
        ax.imshow(datum['subgoal_data'][query_subgoal_ind]['image'])
        pf_query = subgoal_props[query_subgoal_ind].prob_feasible
        ax.set_title(f'Subgoal 1: $P_S$ = {pf_query*100:.1f}\\%')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.plot([0.5, 0.5], [1.0, 0.0],
                transform=ax.transAxes,
                color=[0, 0, 1],
                alpha=0.3)

    @classmethod
    def create_with_state(cls, planner_state_datum, network_file):
        # Initialize the planner
        args = planner_state_datum['args']
        goal = planner_state_datum['goal']
        args.network_file = network_file
        planner = cls(goal, args)

        planner.subgoals = planner_state_datum['subgoals']
        planner.observed_map = planner_state_datum['observed_map']
        planner.inflated_grid = planner_state_datum['inflated_grid']
        planner._recompute_all_subgoal_properties()

        return planner


# Alt versions of functions
def get_policy_data_for_frontiers(grid,
                                  robot_pose,
                                  goal_pose,
                                  chosen_frontier,
                                  all_frontiers,
                                  num_frontiers_max=0,
                                  downsample_factor=1):
    """Compute the optimal orderings for each frontier of interest and return a data
structure containing all the information that would be necessary to compute the
expected cost for each. Also returns the mapping from 'frontiers' to 'inds'."""

    # Remove frontiers that are infeasible
    frontiers = [f for f in all_frontiers if f.prob_feasible != 0]
    frontiers = list(set(frontiers) | set([chosen_frontier]))

    # Calculate the distance to the goal, if infeasible, remove frontier
    goal_distances = get_goal_distances(grid,
                                        goal_pose,
                                        frontiers=frontiers,
                                        downsample_factor=downsample_factor)

    robot_distances = get_robot_distances(grid,
                                          robot_pose,
                                          frontiers=frontiers,
                                          downsample_factor=downsample_factor)

    # Get the most n probable frontiers to limit computational load
    if num_frontiers_max > 0 and num_frontiers_max < len(frontiers):
        frontiers = lsp.core.get_top_n_frontiers_distance(
            frontiers, goal_distances, robot_distances, num_frontiers_max)
        frontiers = list(set(frontiers) | set([chosen_frontier]))

    # Calculate robot and frontier distances
    frontier_distances = get_frontier_distances(
        grid, frontiers=frontiers, downsample_factor=downsample_factor)

    frontier_ind_dict = {f: ind for ind, f in enumerate(frontiers)}
    robot_distances_ind = {
        frontier_ind_dict[f]: robot_distances[f]
        for f in frontiers
    }
    goal_distances_ind = {
        frontier_ind_dict[f]: goal_distances[f]
        for f in frontiers
    }
    frontier_distances_ind = {}
    for ind, f1 in enumerate(frontiers[:-1]):
        f1_ind = frontier_ind_dict[f1]
        for f2 in frontiers[ind + 1:]:
            f2_ind = frontier_ind_dict[f2]
            frontier_distances_ind[frozenset(
                [f1_ind, f2_ind])] = (frontier_distances[frozenset([f1, f2])])

    if frontier_distances is not None:
        assert len(frontier_distances.keys()) == len(
            frontier_distances_ind.keys())

    # Finally, store the data relevant for
    # estimating the frontier properties
    frontier_data = {
        ind: f.nn_input_data
        for f, ind in frontier_ind_dict.items()
    }

    return {
        'subgoal_data': frontier_data,
        'distances': {
            'frontier': frontier_distances_ind,
            'robot': robot_distances_ind,
            'goal': goal_distances_ind,
        },
        'target_subgoal_ind': frontier_ind_dict[chosen_frontier]
    }, frontier_ind_dict
