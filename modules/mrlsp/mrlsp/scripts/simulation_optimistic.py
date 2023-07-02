import environments
# add this to the imports
import numpy as np
import lsp
import mrlsp
from pathlib import Path


def navigate_unknown(args, termination=any, num_robots=2, do_plot=True):
    # Generate Maze
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    new_start_pose = mrlsp.utils.utility.get_path_middle_point(
        known_map, pose, goal, args)

    num_robots = args.num_robots

    robot_grid = lsp.constants.UNOBSERVED_VAL * np.ones_like(known_map)

    start_pose, goal_pose = mrlsp.utils.utility.generate_start_and_goal(
        num_robots=num_robots,
        known_map=known_map,
        same_start=True,
        same_goal=True,
        def_start=new_start_pose,
        def_goal=goal)

    # Instantiate the simulated environment
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements,
        min_breadcrumb_signed_distance=4.0 * args.base_resolution)

    builder = environments.simulated.WorldBuildingUnityBridge

    # Create robots
    robots = mrlsp.utils.utility.robot(num_robots=num_robots,
                                       start_pose=start_pose,
                                       primitive_length=args.step_size,
                                       num_primitives=args.num_primitives,
                                       map_data=map_data)
    print(f"Initially, robot = {robots}")
    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        # create a simulator
        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)
        # set the inflation radius
        simulator.frontier_grouping_inflation_radius = simulator.inflation_radius

        # add planner for the robots
        planner = mrlsp.planners.OptimisticPlanner(robots, goal_pose, args)

        # Now the planning loop
        planning_loops = [
            mrlsp.planners.PlanningLoop(goal_pose[i],
                                        known_map,
                                        simulator,
                                        unity_bridge=None,
                                        robot=robots[i],
                                        args=args,
                                        verbose=True)
            for i in range(num_robots)
        ]
        # path,image,subgoal,visibility mask to store the path taken by every robot: This is only used
        # for plotting and updating the planner
        paths = [0 for _ in range(num_robots)]
        individual_subgoal = [set() for _ in range(num_robots)]
        pano_image = [0 for _ in range(num_robots)]
        v_mask = [None for _ in range(num_robots)]
        pose = [None for _ in range(num_robots)]

        # creating planning loop object iterable so we can do next(iter_planning_loop) to get data
        iter_planning_loops = [
            iter(planning_loop) for planning_loop in planning_loops
        ]

        # keeps track of goal reached condition for individual robot
        goals_reached = np.array(
            [planning_loop.goal_reached for planning_loop in planning_loops])

        timestamp = 0
        # Get observation and update map and subgoals
        while (not termination(goals_reached)):
            for i, iter_planning_loop in enumerate(iter_planning_loops):

                # Get robot data. If the robot gets data, then do the update part
                try:
                    robot_data = next(iter_planning_loop)
                    # Since all the robot has individual observation, update the grid (observed map) all together.
                    robot_grid = mrlsp.utils.utility.update_grid(
                        robot_grid, robot_data['robot_grid'])
                    # Store the observations
                    pano_image[i] = robot_data['image']
                    v_mask[i] = robot_data['visibility_mask']
                    individual_subgoal[i] = robot_data['subgoals']
                    pose[i] = robot_data['robot_pose']

                # if next(iter_planning_loop) raises exception, (means goal is reached for that robot)
                # just pass it and plot using previous data of robot
                except StopIteration:
                    print("Exception encountered")
                    pass

                # check if goal is reached or not, if goals is reached; remove it from the iterable
                goals_reached[i] = planning_loops[i].goal_reached

            # calculate the subgoal from the observation of all robots
            subgoals, inflated_grid = mrlsp.utils.utility.find_subgoal(
                robot_grid, robots, goal_pose, simulator)

            # update LSP planner
            planner.update(
                {'image': pano_image[i]},  # observation image
                robot_grid,  # observed map
                inflated_grid,
                subgoals,
                robots,
                v_mask[i])

            # Update observation for every robot.
            for i in range(num_robots):
                extra_subgoals = None
                planning_loops[i].update_inflated_grid(inflated_grid)
                planning_loops[i].update_subgoals(subgoals, extra_subgoals)
                # for plotting
                paths[i] = planning_loops[i].path_covered

            joint_action = planner.compute_selected_subgoal()
            for i, action in enumerate(joint_action):
                planning_loops[i].set_chosen_subgoal(action)

            if do_plot:
                mrlsp.utils.plotting.plot_mrlsp(args,
                                                timestamp,
                                                num_robots,
                                                robots,
                                                goal_pose,
                                                subgoals,
                                                pano_image,
                                                robot_grid,
                                                known_map=known_map,
                                                paths=paths)

    index_robot = np.where(goals_reached == True)[0]  # noqa
    mrlsp.utils.plotting.plot_final_figure(args,
                                           num_robots,
                                           robots,
                                           goal_pose,
                                           subgoals,
                                           robot_grid,
                                           known_map=known_map,
                                           paths=paths,
                                           planner='optimistic')
    # First robot to reach goal
    cost_for_goal = robots[index_robot[0]].net_motion
    print(
        f"Total cost for goal = {cost_for_goal} reached by robot {index_robot[0] + 1}"
    )
    cost_file = Path(args.save_dir) / f'planner_optimistic_{args.map_type}_cost_{args.current_seed}_r{num_robots}.txt'
    with open(cost_file, 'w') as f:
        f.write(f'{cost_for_goal}')
    return cost_for_goal


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--num_robots', type=int)
    parser.add_argument('--limit_frontiers', type=int)
    parser.add_argument('--do_plot', type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()
    args.current_seed = args.seed[0]

    cost = navigate_unknown(args, termination=any, do_plot=args.do_plot)
