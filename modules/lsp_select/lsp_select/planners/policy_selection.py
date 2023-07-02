import numpy as np
import gridmap
from lsp.planners import Planner
from lsp_select import offline_replay


class PolicySelectionPlanner(Planner):
    """Meta-planner class that handles selection among multiple planners/policies."""
    def __init__(self, goal, planners, chosen_planner_idx, args):
        super(PolicySelectionPlanner, self).__init__(goal)
        self.args = args
        self.planners = planners

        # Store history of robot poses and images
        self.poses = []
        self.images = []

        self.counter = 0
        self.chosen_planner_idx = chosen_planner_idx

        # Make a grid to store nearest pose data.
        # Each cell contrains a dictionary with index of nearest pose and the distance to it.
        self.nearest_pose_data = np.array([[{'index': None, 'distance': np.inf} for _ in range(args.map_shape[1])]
                                           for _ in range(args.map_shape[0])])
        self.navigation_data = {'start': self.args.robot_pose, 'goal': self.goal}

    def update(self, observation, observed_map, subgoals, robot_pose, visibility_mask):
        """Updates the information in currently chosen planner and records observed poses/images."""
        self.robot_pose = robot_pose
        self.observation = observation
        self.observed_map = observed_map
        self.planners[self.chosen_planner_idx].update(observation, observed_map, subgoals, robot_pose, visibility_mask)
        self.inflated_grid = self.planners[self.chosen_planner_idx].inflated_grid
        self.subgoals = self.planners[self.chosen_planner_idx].subgoals

        pose = [robot_pose.x, robot_pose.y, robot_pose.yaw]
        self.poses.append(pose)
        self.images.append(observation['image'])

        # Update nearest pose data
        self.update_nearest_pose_data(visibility_mask, [pose[0], pose[1]])
        self.counter += 1

    def compute_selected_subgoal(self):
        """Compute selected subgoal from the chosen planner."""
        return self.planners[self.chosen_planner_idx].compute_selected_subgoal()

    def update_nearest_pose_data(self, visibility_mask, current_pose):
        """As the robot navigates, update pose_data to reflect the nearest robot pose
        from all poses in the visibility region.
        """
        visible_cells = np.where(visibility_mask == 1)
        poses = np.column_stack(visible_cells)
        cost_grid, _ = gridmap.planning.compute_cost_grid_from_position(visibility_mask != 1,
                                                                        current_pose,
                                                                        use_soft_cost=True)
        # Update nearest pose data for all cells in the visibility region
        for x, y in poses:
            distance = cost_grid[x, y]
            pose_data = self.nearest_pose_data[x, y]
            if pose_data['index'] is None:
                pose_data['index'] = self.counter
                pose_data['distance'] = distance
            elif pose_data['distance'] > distance:
                pose_data['index'] = self.counter
                pose_data['distance'] = distance

    def get_costs(self):
        """ After navigation is complete, get replayed costs for all other planners."""
        self.navigation_data['poses'] = np.array(self.poses)
        self.navigation_data['images'] = self.images
        self.navigation_data['nearest_pose_data'] = self.nearest_pose_data
        self.navigation_data['partial_map'] = self.observed_map
        self.navigation_data['final_subgoals'] = self.subgoals
        self.navigation_data['net_motion'] = self.args.robot.net_motion
        self.navigation_data['robot_path'] = self.args.robot.all_poses

        lb_costs = np.full((len(self.planners), 2), np.nan)
        planner_costs = np.full(len(self.planners), np.nan)
        self.args.chosen_planner_idx = self.chosen_planner_idx
        for i, planner in enumerate(self.planners):
            self.args.replayed_planner_idx = i
            if i == self.chosen_planner_idx:
                # The cost of the chosen planner is the net distance traveled
                planner_costs[i] = self.navigation_data['net_motion']
            else:
                # For other planners, get lower bound costs via offline replay
                optimistic_lb, simply_connected_lb = offline_replay.get_lowerbound_planner_costs(self.navigation_data,
                                                                                                 planner,
                                                                                                 self.args)
                lb_costs[i] = [optimistic_lb, simply_connected_lb]

        return planner_costs, lb_costs
