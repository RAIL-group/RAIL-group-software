import os
import matplotlib.pyplot as plt
import lsp
import environments


def data_gen_main(args, do_plan_with_naive=True):
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)

    # Open the connection to Unity (if desired)
    if args.unity_path is None:
        raise ValueError('Unity Environment Required')

    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements,
        min_breadcrumb_signed_distance=4.0 * args.base_resolution)
    builder = environments.simulated.WorldBuildingUnityBridge

    # Helper function for creating a new robot instance
    robot = lsp.robot.Turtlebot_Robot(pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=map_data)

    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)
        simulator.frontier_grouping_inflation_radius = (
            simulator.inflation_radius)

        known_planner = lsp.planners.KnownSubgoalPlanner(
            goal=goal, known_map=known_map, args=args,
            do_compute_weightings=True)

        planning_loop = lsp.planners.PlanningLoop(goal,
                                                  known_map,
                                                  simulator,
                                                  unity_bridge,
                                                  robot,
                                                  args,
                                                  verbose=True)

        for counter, step_data in enumerate(planning_loop):
            # Update the planner objects
            known_planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])

            # Get and write the data
            subgoal_training_data = known_planner.get_subgoal_training_data()
            lsp.utils.data.write_training_data_to_pickle(
                subgoal_training_data,
                step_counter=known_planner.update_counter,
                args=known_planner.args)

            if not do_plan_with_naive:
                planning_loop.set_chosen_subgoal(
                    known_planner.compute_selected_subgoal())

    # Write final plot to file
    image_file = os.path.join(
        args.save_dir, 'data_collect_plots',
        os.path.splitext(args.data_file_base_name)[0] + f'_{args.current_seed}.png')
    print(image_file)

    plt.figure(figsize=(8, 8))
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(known_planner.observed_map, known_map))
    path = robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    p = path[-1]
    plt.plot(ys, xs, 'r')
    plt.plot(p.y, p.x, 'go')
    plt.savefig(image_file, dpi=150)


if __name__ == "__main__":
    from . import shared
    parser = shared.get_command_line_parser()
    parser.add_argument('--data_file_base_name', type=str)
    args = parser.parse_args()

    data_gen_main(args)
