import numpy as np
from .planner import Planner
import gridmap
from ..pose import compute_path_length


class KnownPlanner(Planner):
    def __init__(self, goal, known_map, args):
        super(KnownPlanner, self).__init__(goal)

        self.known_map = known_map
        self.subgoals = set()
        self.selected_subgoal = None
        self.observed_map = None
        self.args = args
        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        if self.inflation_radius >= np.sqrt(5):
            self.downsample_factor = 2
        else:
            self.downsample_factor = 1

        self.inflated_known_grid = gridmap.utils.inflate_grid(
            known_map, inflation_radius=self.inflation_radius)

        # Compute cost grid
        _, self.get_path = gridmap.planning.compute_cost_grid_from_position(
            self.inflated_known_grid, [goal.x, goal.y], use_soft_cost=True)

    def compute_path_to_goal(self):
        """Returns the path and distance to the goal."""
        did_plan, path = self.get_path([self.robot_pose.x, self.robot_pose.y],
                                       do_sparsify=True,
                                       do_flip=True,
                                       bound=None)

        distance = compute_path_length(path)
        return did_plan, path, distance

    def compute_selected_subgoal(self):
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
