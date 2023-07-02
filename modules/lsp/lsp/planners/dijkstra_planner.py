import numpy as np
from .planner import Planner
import gridmap


class DijkstraPlanner(Planner):
    def __init__(self, goal, args):
        super(DijkstraPlanner, self).__init__(goal)

        self.subgoals = None
        self.selected_subgoal = None
        self.observed_map = None
        self.args = args

        self.inflation_radius = args.inflation_radius_m / args.base_resolution

    def compute_selected_subgoal(self):
        if not self.subgoals:
            return None

        # Compute cost grid
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            self.inflated_grid, [self.goal.x, self.goal.y], use_soft_cost=True)

        # Compute the plan
        did_plan, path = get_path([self.robot_pose.x, self.robot_pose.y],
                                  do_sparsify=False,
                                  do_flip=True)
        if not did_plan:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(self.inflated_grid)
            plt.show()
            plt.pause(100)
        if np.argmax(self.observed_map[path[0, -1], path[1, -1]] >= 0):
            return None

        # Determine the chosen subgoal
        ind = np.argmax(self.observed_map[path[0, :], path[1, :]] < 0)
        return min(self.subgoals,
                   key=lambda s: s.get_distance_to_point((path.T)[ind]))
