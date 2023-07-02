import copy
import gridmap


class Planner(object):
    """Abstract class for planning with frontiers."""
    def __init__(self, goal):
        self.name = 'Planner'
        self.observed_map = None
        self.goal = goal

    def update(self, observation, observed_map, subgoals, robot_pose, *args,
               **kwargs):
        self.observation = observation
        self.observed_map = observed_map
        self.subgoals = [copy.copy(s) for s in subgoals]
        self.robot_pose = robot_pose
        self.inflated_grid = self._get_inflated_occupancy_grid()

    def compute_selected_subgoal(self):
        """Returns the selected subgoal (frontier)."""
        raise NotImplementedError()

    def _get_inflated_occupancy_grid(self):
        """Compute the inflated grid."""
        # Inflate the grid and generate a plan
        inflated_grid = gridmap.utils.inflate_grid(
            self.observed_map, inflation_radius=self.inflation_radius)

        inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
            inflated_grid, self.robot_pose)
        # Prevents robot from getting stuck occasionally: sometimes (very
        # rarely) the robot would reveal an obstacle and then find itself
        # within the inflation radius of that obstacle. This should have
        # no side-effects, since the robot is expected to be in free space.
        inflated_grid[int(self.robot_pose.x), int(self.robot_pose.y)] = 0

        return inflated_grid
