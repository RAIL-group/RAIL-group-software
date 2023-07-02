import copy
import gridmap
import logging
import lsp
import time

NUM_MAX_FRONTIERS = 7


class KnownSubgoalPlanner(lsp.planners.KnownPlanner):
    def __init__(self, goal, known_map, args):
        super(KnownSubgoalPlanner, self).__init__(goal, known_map, args)
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

        # Update frontier properties using the known map
        self._update_frontier_props_known(robot_pose, self.goal,
                                          observation['image'],
                                          visibility_mask)

    def _update_frontier_props_known(self,
                                     robot_pose,
                                     goal_pose,
                                     image,
                                     visibility_mask=None,
                                     do_silence=False):
        """Compute frontier properties using the known grid and write data
        to a .tfrecords file."""
        logger = logging.getLogger("KnownSubgoalPlanner")

        f_gen = [f for f in self.subgoals if not f.props_set]
        updated_frontiers = []
        stime = time.time()
        lsp.core.update_frontiers_properties_known(self.inflated_known_grid,
                                                   self.inflated_grid,
                                                   self.subgoals, f_gen,
                                                   robot_pose, goal_pose,
                                                   self.downsample_factor)
        logger.debug(
            f"  time to update (all) frontier properties: {time.time() - stime}"
        )

        for frontier in f_gen:
            if frontier.is_obstructed:
                continue

            updated_frontiers.append(frontier)

            if not do_silence:
                lsp.utils.command_line.print_frontier_data(
                    frontier, num_leading_spaces=16)

        self.subgoal_data_list = lsp.utils.learning_vision.get_oriented_data_from_obs(
            updated_frontiers, robot_pose, self.goal, image)

        self.updated_subgoals = updated_frontiers
