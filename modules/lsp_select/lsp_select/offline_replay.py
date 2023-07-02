import numpy as np
import math
import matplotlib.pyplot as plt

import environments
from common import Pose
import gridmap
from gridmap import laser, utils
from gridmap.constants import UNOBSERVED_VAL, FREE_VAL, COLLISION_VAL
import lsp.core
import lsp_select


class OfflineReplay(lsp.simulators.Simulator):
    """Simulator class for offline replay of robot observations
    based on data collected during a trial.

    Requires the following to instantiate:
        partial_map: The final map obtained at the end of a navigation trial.
        poses: List of robot poses seen during a trial. Each pose as (x, y, yaw).
        images: List of images corresponding to poses.
        nearest_pose_data: A 2D-array with each cell as a dictionary consisting 'index' of
                           the pose nearest to that cell and 'distance' to the nearest pose.
        final_subgoals: List of final subgoals present in the partial_map.
        goal: Goal pose
        args: args with usual parameters as the parent Simulator class
        dist_mask_frontiers: Distance (grid cell units) between robot and frontiers below
                             which frontiers are masked during replay (so as to prevent
                             the robot from entering unseen space).
    """
    def __init__(self,
                 partial_map,
                 poses,
                 images,
                 nearest_pose_data,
                 final_subgoals,
                 goal,
                 args,
                 dist_mask_frontiers=10,
                 verbose=True):
        self.args = args
        self.goal = goal
        self.resolution = args.base_resolution
        self.inflation_radius = args.inflation_radius_m / self.resolution
        self.frontier_grouping_inflation_radius = 0

        self.laser_max_range_m = args.laser_max_range_m
        self.disable_known_grid_correction = args.disable_known_grid_correction

        # Store the known grid
        self.known_map = partial_map.copy()

        # Create the directions object
        self.laser_scanner_num_points = args.laser_scanner_num_points
        self.directions = laser.get_laser_scanner_directions(
            num_points=self.laser_scanner_num_points,
            field_of_view_rad=math.radians(args.field_of_view_deg))

        self.poses = poses
        self.images = images
        self.nearest_pose_data = nearest_pose_data
        self.final_subgoals = final_subgoals
        self.next_subgoals_to_mask = set()

        self.dist_mask_frontiers = dist_mask_frontiers
        self.verbose = verbose

    @property
    def pessimistic_map(self):
        """Represents map where all unknown space is considered as obstacles"""
        pessimistic_map = self.known_map.copy()
        pessimistic_map[pessimistic_map == UNOBSERVED_VAL] = COLLISION_VAL
        return pessimistic_map

    def get_laser_scan(self, robot):
        """Get a simulated laser scan."""
        # Get the laser scan
        ranges = laser.simulate_sensor_measurement(
            self.pessimistic_map,
            self.directions,
            max_range=self.laser_max_range_m / self.resolution + 2,
            sensor_pose=robot.pose)

        return ranges

    def get_image(self, robot, **kwargs):
        """Get the image from the stored pose nearest to current robot pose."""
        if self.verbose:
            print("Retrieving Image")

        nearest_pose = self.nearest_pose_data[int(robot.pose.x), int(robot.pose.y)]
        pose_idx = nearest_pose['index']
        distance = nearest_pose['distance']

        r_pose = self.poses[pose_idx]
        pano_image = self.images[pose_idx]
        aligned_pose = Pose(x=r_pose[0], y=r_pose[1], yaw=robot.pose.yaw - r_pose[2])

        pano_image = environments.utils.convert.image_aligned_to_robot(
            image=pano_image, r_pose=aligned_pose)

        if self.verbose:
            print(f"  image from nearest pose {(r_pose[0], r_pose[1])} at distance: {distance:.2f}")

        return (pano_image, r_pose), None

    def get_updated_frontier_set(self, inflated_grid, robot, saved_frontiers):
        """Compute the frontiers, store the new ones and compute properties."""
        new_frontiers = lsp.core.get_frontiers(
            inflated_grid,
            group_inflation_radius=self.frontier_grouping_inflation_radius)
        saved_frontiers = lsp.core.update_frontier_set(saved_frontiers,
                                                       new_frontiers)

        lsp.core.update_frontiers_goal_in_frontier(saved_frontiers, self.goal)

        # Mask and remove frontiers if they are in final_subgoals set and also near the robot.
        self.next_subgoals_to_mask = set()
        pose = np.array([robot.pose.x, robot.pose.y])
        for frontier in self.final_subgoals:
            dist_to_frontier = frontier.get_distance_to_point(pose)
            if dist_to_frontier <= self.dist_mask_frontiers:
                self.known_map = lsp.core.mask_grid_with_frontiers(self.known_map, [frontier])
                self.next_subgoals_to_mask.add(frontier)
                if frontier in saved_frontiers:
                    saved_frontiers.remove(frontier)

        return saved_frontiers


def get_lowerbound_planner_costs(navigation_data, planner, args):
    """Helper function to get optimistic and simply-connected lower bound cost through offline replay
    of a planner based on collected navigation data.
    """
    pose = navigation_data['start']
    goal = navigation_data['goal']
    partial_map = navigation_data['partial_map']
    final_subgoals = navigation_data['final_subgoals']

    robot = lsp.robot.Turtlebot_Robot(pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=None)
    simulator = OfflineReplay(partial_map,
                              poses=navigation_data['poses'],
                              images=navigation_data['images'],
                              nearest_pose_data=navigation_data['nearest_pose_data'],
                              final_subgoals=navigation_data['final_subgoals'],
                              goal=goal,
                              args=args)
    simulator.frontier_grouping_inflation_radius = simulator.inflation_radius
    planning_loop = lsp.planners.PlanningLoop(goal,
                                              partial_map,
                                              simulator,
                                              unity_bridge=None,
                                              robot=robot,
                                              args=args,
                                              verbose=True)

    masked_frontiers = set()
    all_alt_costs = []

    for counter, step_data in enumerate(planning_loop):
        planner.update(
            {'image': step_data['image'][0]},
            step_data['robot_grid'],
            step_data['subgoals'],
            step_data['robot_pose'],
            step_data['visibility_mask'])
        planning_loop.set_chosen_subgoal(planner.compute_selected_subgoal())

        inflated_grid = utils.inflate_grid(
            partial_map, inflation_radius=args.inflation_radius_m / args.base_resolution)

        frontier_grid = inflated_grid.copy()
        frontier_grid[inflated_grid == FREE_VAL] = COLLISION_VAL
        frontier_grid[inflated_grid == UNOBSERVED_VAL] = FREE_VAL
        goal_grid = inflated_grid.copy()
        goal_grid[inflated_grid == UNOBSERVED_VAL] = COLLISION_VAL

        # Open up all frontiers
        for f in final_subgoals:
            frontier_grid[f.points[0, :], f.points[1, :]] = FREE_VAL
            goal_grid[f.points[0, :], f.points[1, :]] = FREE_VAL

        # Block already masked frontiers in both grids
        masked_frontiers.update(simulator.next_subgoals_to_mask)
        for f in masked_frontiers:
            frontier_grid[f.points[0, :], f.points[1, :]] = COLLISION_VAL
            goal_grid[f.points[0, :], f.points[1, :]] = COLLISION_VAL

        # Block 'next mask' frontiers for goal grid
        for f in simulator.next_subgoals_to_mask:
            frontier_grid[f.points[0, :], f.points[1, :]] = FREE_VAL
            goal_grid[f.points[0, :], f.points[1, :]] = COLLISION_VAL

        cost_grid_goal, _ = gridmap.planning.compute_cost_grid_from_position(
            goal_grid, [goal.x, goal.y], use_soft_cost=True)

        alt_costs_to_goal = []
        for f in simulator.next_subgoals_to_mask:
            cost_grid_frontier, _ = gridmap.planning.compute_cost_grid_from_position(
                frontier_grid, f.get_frontier_point(), use_soft_cost=True)

            total_cost_grid = cost_grid_frontier + cost_grid_goal
            costs_temp = []
            for frontier in final_subgoals:
                if frontier in masked_frontiers:
                    continue
                f_x, f_y = frontier.get_frontier_point()
                costs_temp.append(total_cost_grid[f_x, f_y])
            if len(costs_temp) != 0:
                alt_costs_to_goal.append(costs_temp)

        if len(alt_costs_to_goal) != 0:
            all_alt_costs.append([robot.net_motion, alt_costs_to_goal])

        if args.do_plot:
            plt.ion()
            plt.figure(1, figsize=(12, 8))
            plt.clf()
            plt.subplot(211)
            plt.axis('off')
            plt.imshow(step_data['image'][0])
            ax = plt.subplot(224)
            plt.axis('off')
            lsp_select.utils.plotting.plot_pose(ax, robot.pose, color='blue')
            lsp_select.utils.plotting.plot_grid_with_frontiers(
                ax, planner.observed_map, simulator.known_map, planner.subgoals)
            lsp_select.utils.plotting.plot_pose(ax, goal, color='green', filled=False)
            lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses, 'r')
            alt_costs_min = []
            for net_motion, alt_costs in all_alt_costs:
                alt_costs_min.append(net_motion + min([min(c) if len(c) > 0 else float('inf')
                                                       for c in alt_costs]))

            lbopt = min(alt_costs_min) if len(alt_costs_min) != 0 else float('inf')
            plt.title(f'Offline Replay with {args.planner_names[args.replayed_planner_idx]}\n'
                      r'$C^{lb,opt}$= ' f'{lbopt:.2f}, ' r'$C^{lb,s.c.}$= ' f'{robot.net_motion:.2f}')
            nearest_pose = step_data['image'][1][:2]
            plt.scatter(nearest_pose[0], nearest_pose[1], color='darkorange', s=10)
            ax = plt.subplot(223)
            plt.axis('off')
            lsp_select.utils.plotting.plot_grid_with_frontiers(
                ax, simulator.known_map, None, frontiers=[])
            lsp_select.utils.plotting.plot_pose_path(ax, navigation_data['robot_path'])

            plt.title(r'Final partial map $m_{final}$ ' f'and path with {args.planner_names[args.chosen_planner_idx]}'
                      f'\nTotal cost: {navigation_data["net_motion"]:.2f}')
            plt.show()
            plt.pause(0.01)

        if robot.net_motion >= 3000:
            break

    # Compute optimistic lower bound
    alt_costs_min = []
    for net_motion, alt_costs in all_alt_costs:
        alt_costs_min.append(net_motion + min([min(c) if len(c) > 0 else float('inf')
                                               for c in alt_costs]))
    optimistic_lb = min(alt_costs_min) if len(alt_costs_min) != 0 else float('inf')
    simply_connected_lb = robot.net_motion

    return optimistic_lb, simply_connected_lb
