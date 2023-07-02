import copy
import gridmap
import lsp
import lsp_select
from lsp.planners.subgoal_planner import LearnedSubgoalPlanner


class LSPCycleGAN(LearnedSubgoalPlanner):
    """Class that extends LSP for transforming image observations using CycleGAN."""
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

        self.observation['image'] = self.cyclegan_generator(self.observation['image'])

        # Update the subgoal inputs
        self._update_subgoal_inputs(self.observation['image'], robot_pose, self.goal)

        # Once the subgoal inputs are set, compute their properties
        self._update_subgoal_properties(robot_pose, self.goal)
