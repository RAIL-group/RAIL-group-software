import copy
import mrlsp
from mrlsp.utils.utility import get_multirobot_distances


class MRPlanner(object):
    """Abstract class for planning with frontiers."""

    def __init__(self, robots, goal):
        self.name = 'multirobot_planner'
        self.robot_grid = None
        self.inflated_grid = None
        self.robots = robots
        self.goal_pose = goal[0]

    def update(self, observation, robot_grid, inflated_grid, subgoals, robot, *args,
               **kwargs):
        self.observation = observation
        self.robot_grid = robot_grid
        self.inflated_grid = inflated_grid
        self.subgoals = [copy.copy(s) for s in subgoals]
        self.robots = robot

    def compute_selected_subgoal(self):
        """Returns the selected subgoal (frontier)."""
        raise NotImplementedError()


class BaseLSPPlanner(MRPlanner):
    '''LSP planner is used just to reuse the same subgoal properties as in LSP,
    In every planner where subgoal properties are used for multirobot, BaseLSPPlanner class is inherited.'''

    def __init__(self, robots, goal, args):
        super(BaseLSPPlanner, self).__init__(robots, goal)
        self.args = args
        self.subgoals = None
        self.planners = mrlsp.utils.utility.lsp_planner(args, len(robots), goal)

    def update(self, observation, robot_grid, inflated_grid, subgoals, robot, visibility_mask):
        # Update every lsp planner observation
        num_robots = len(self.robots)
        self.observation = observation
        self.robot_grid = robot_grid
        self.inflated_grid = inflated_grid
        frontiers = set([copy.copy(s) for s in subgoals])
        self.robots = robot
        for i in range(num_robots):
            self.planners[i].update(
                {'image': self.observation[i]},
                self.robot_grid,
                frontiers,
                self.robots[i].pose,
                visibility_mask[i])
        self.distances_mr = get_multirobot_distances(self.inflated_grid, self.robots, [self.goal_pose], frontiers)
        frontiers, loop_frontiers = mrlsp.core.update_frontier_properties_for_multirobot(
            self.planners, frontiers, self.distances_mr)
        self.subgoals, extra_subgoals = mrlsp.utils.utility.limit_total_subgoals(
            num_robots, frontiers, self.distances_mr, self.args.limit_frontiers
        )
        print("-----selected subgoals properties-----")
        for sg in self.subgoals:
            print(
                f"subgoal: {sg}:{sg.centroid}, P = {sg.prob_feasible},  \
                    Ts = {sg.delta_success_cost + self.distances_mr['goal'][sg]}, Te = {sg.exploration_cost}")
        print("------------------------------")
        extra_subgoals = extra_subgoals.union(loop_frontiers)
