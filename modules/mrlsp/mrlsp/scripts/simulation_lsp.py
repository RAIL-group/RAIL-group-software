import environments
# add this to the imports
import gridmap
import lsp
import matplotlib.pyplot as plt
import mrlsp
from pathlib import Path


def navigate_unknown(args):
    # Generate maze
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)

    pose = mrlsp.utils.utility.get_path_middle_point(
        known_map, pose, goal, args)

    # Instantiate the simulated environment
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements,
        min_breadcrumb_signed_distance=4.0 * args.base_resolution)

    builder = environments.simulated.WorldBuildingUnityBridge

    # Create a robot
    robot = lsp.robot.Turtlebot_Robot(pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=map_data)

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

        # add the planner
        planner = lsp.planners.KnownSubgoalPlanner(goal, known_map, args)

        # Now the planning loop
        planning_loop = lsp.planners.PlanningLoop(goal,
                                                  known_map,
                                                  simulator,
                                                  unity_bridge=None,
                                                  robot=robot,
                                                  args=args,
                                                  verbose=True)

        # Now the loop calling the generator function in planning_loop
        for counter, step_data in enumerate(planning_loop):
            # update the planner objects
            planner.update(
                {'image': step_data['image']},  # observation
                step_data['robot_grid'],  # observed map
                step_data["subgoals"],
                step_data["robot_pose"],
                step_data["visibility_mask"])
            # compute subgoal, and update it to planning loop
            planning_loop.set_chosen_subgoal(
                planner.compute_selected_subgoal())

            # This step is required just for plotting

            # if subgoal is chosed, donot mask the subgoal, and generate water tight grid
            if planning_loop.chosen_subgoal is not None:
                planning_grid = lsp.core.mask_grid_with_frontiers(
                    planner.inflated_grid,
                    planner.subgoals,
                    do_not_mask=planning_loop.chosen_subgoal)
            # else
            else:
                planning_grid = lsp.core.mask_grid_with_frontiers(
                    planner.inflated_grid,
                    [],
                )

            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                planning_grid, [goal.x, goal.y], use_soft_cost=True)
            did_plan, path = get_path([robot.pose.x, robot.pose.y],
                                      do_sparsify=True,
                                      do_flip=True)
            # upto here

            # Plotting
            plt.ion()
            plt.figure(1)
            plt.clf()
            ax = plt.subplot(211)
            plt.imshow(step_data['image'])
            ax = plt.subplot(212)
            mrlsp.utils.plotting.plot_pose(ax, robot.pose, color='blue')
            mrlsp.utils.plotting.plot_grid_with_frontiers(
                ax, planner.observed_map, known_map, planner.subgoals)
            mrlsp.utils.plotting.plot_pose(ax,
                                           goal,
                                           color='green',
                                           filled=False)
            mrlsp.utils.plotting.plot_path(ax, path)
            mrlsp.utils.plotting.plot_pose_path(ax, robot.all_poses)
            plt.show()
            plt.pause(0.1)

        image_file = Path(args.save_dir) / f'images_{args.current_seed}.png'
        plt.savefig(image_file)

        cost_file = Path(args.save_dir) / f'cost_{args.current_seed}_r1.txt'
        with open(cost_file, 'w') as f:
            f.write(f'{robot.net_motion}')

        return robot.net_motion


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--network_file', type=str)
    args = parser.parse_args()
    args.current_seed = args.seed[0]

    cost = navigate_unknown(args)
