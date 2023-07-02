import gridmap
import lsp
import numpy as np
import time


class PlanningLoop():
    def __init__(self,
                 goal,
                 known_map,
                 simulator,
                 unity_bridge,
                 robot,
                 args,
                 verbose=False):
        self.goal = goal
        self.known_map = known_map
        self.simulator = simulator
        self.unity_bridge = unity_bridge
        self.robot = robot
        self.args = args
        self.did_succeed = True
        self.verbose = verbose
        self.chosen_subgoal = None

        self.goal_reached = False
        self.path_covered = None
        self.robot_grid = lsp.constants.UNOBSERVED_VAL * np.ones_like(
            self.known_map)

        # For checking
        self.subgoals = set()
        self.planning_grid = None
        self.inflated_grid = []

    def __iter__(self):
        counter = 0
        count_since_last_turnaround = 100
        fn_start_time = time.time()

        # Main planning loop
        while (not self.goal_reached):

            if self.verbose:
                print(f"Goal: {self.goal.x}, {self.goal.y}")
                print(
                    f"Robot: {self.robot.pose.x}, {self.robot.pose.y} [motion: {self.robot.net_motion}]"
                )
                print(f"Counter: {counter} | Count since last turnaround: "
                      f"{count_since_last_turnaround}")

            # Compute observations and update map
            pano_image = self.simulator.get_image(self.robot)
            _, self.robot_grid, visible_region = (
                self.simulator.get_laser_scan_and_update_map(
                    self.robot, self.robot_grid, True))

            # Compute intermediate map grids for planning
            visibility_mask = gridmap.utils.inflate_grid(
                visible_region, 1.8, -0.1, 1.0)
            inflated_grid = self.simulator.get_inflated_grid(
                self.robot_grid, self.robot)
            inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
                inflated_grid, self.robot.pose)
            self.inflated_grid = inflated_grid
            # Compute the subgoal
            subgoals = self.simulator.get_updated_frontier_set(
                self.inflated_grid, self.robot, set())

            yield {
                'subgoals': subgoals,
                'image': pano_image,
                'robot_grid': self.robot_grid,
                'robot_pose': self.robot.pose,
                'visibility_mask': visibility_mask,
            }
            if self.chosen_subgoal is None:
                if self.verbose:
                    print("Planning with naive/Dijkstra planner.")
                planning_grid = lsp.core.mask_grid_with_frontiers(
                    self.inflated_grid,
                    [],
                )
            else:
                if self.verbose:
                    print("Planning via subgoal masking.")

                planning_grid = lsp.core.mask_grid_with_frontiers(
                    self.inflated_grid,
                    self.subgoals,
                    do_not_mask=self.chosen_subgoal)

            # Check that the plan is feasible and compute path
            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                planning_grid, [self.goal.x, self.goal.y], use_soft_cost=True)
            did_plan, path = get_path([self.robot.pose.x, self.robot.pose.y],
                                      do_sparsify=True,
                                      do_flip=True)
            # Update the path in the path variable
            self.planning_grid = planning_grid
            self.path_covered = path
            # Move the robot
            motion_primitives = self.robot.get_motion_primitives()
            do_use_path = (count_since_last_turnaround > 10)
            costs, _ = lsp.primitive.get_motion_primitive_costs(
                planning_grid,
                cost_grid,
                self.robot.pose,
                path,
                motion_primitives,
                do_use_path=do_use_path)
            if abs(min(costs)) < 1e10:
                primitive_ind = np.argmin(costs)
                self.robot.move(motion_primitives, primitive_ind)
                print("robot move")
                if primitive_ind == len(motion_primitives) - 1:
                    count_since_last_turnaround = -1
            else:
                # Force the robot to return to known space
                cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                    planning_grid, [self.goal.x, self.goal.y],
                    use_soft_cost=True,
                    obstacle_cost=1e5)
                did_plan, path = get_path(
                    [self.robot.pose.x, self.robot.pose.y],
                    do_sparsify=True,
                    do_flip=True)
                costs, _ = lsp.primitive.get_motion_primitive_costs(
                    planning_grid,
                    cost_grid,
                    self.robot.pose,
                    path,
                    motion_primitives,
                    do_use_path=False)

                self.robot.move(motion_primitives, np.argmin(costs))

            # Check that the robot is not 'stuck'.
            if self.robot.max_travel_distance(
                    num_recent_poses=100) < 5 * self.args.step_size:
                print("Planner stuck")
                self.did_succeed = False
                break

            if self.robot.net_motion > 4000:
                print("Reached maximum distance.")
                self.did_succeed = False
                break
            counter += 1
            count_since_last_turnaround += 1
            if self.verbose:
                print("")
            # Check if goal is reached
            self.check_goal_reached()

        if self.verbose:
            print("TOTAL TIME:", time.time() - fn_start_time)

    def set_chosen_subgoal(self, new_chosen_subgoal):
        self.chosen_subgoal = new_chosen_subgoal

    def update_robot_grid(self, robot_grid):
        self.robot_grid = robot_grid

    def update_subgoals(self, subgoals, extra_subgoals=None):
        if extra_subgoals is None:
            self.subgoals = subgoals
        else:
            self.subgoals = subgoals.union(extra_subgoals)

    def update_inflated_grid(self, inflated_grid):
        self.inflated_grid = inflated_grid

    def check_goal_reached(self):
        if (not (np.abs(self.robot.pose.x - self.goal.x) >= 3 *
                 self.args.step_size or np.abs(self.robot.pose.y - self.goal.y)
                 >= 3 * self.args.step_size)):
            self.goal_reached = True
        else:
            self.goal_reached = False
