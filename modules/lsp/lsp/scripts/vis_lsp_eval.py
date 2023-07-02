import common
import os
import matplotlib.pyplot as plt
import lsp
import environments


def evaluate_main(args):
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
    def get_robot():
        return lsp.robot.Turtlebot_Robot(pose,
                                         primitive_length=args.step_size,
                                         num_primitives=args.num_primitives,
                                         map_data=map_data)

    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        # Write starting seed to the log file
        logfile = os.path.join(args.save_dir, args.logfile_name)
        with open(logfile, "a+") as f:
            f.write(f"LOG: {args.current_seed}\n")

        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)
        simulator.frontier_grouping_inflation_radius = (
            simulator.inflation_radius)

        learned_planner = lsp.planners.LearnedSubgoalPlanner(goal=goal,
                                                             args=args)
        learned_robot = get_robot()
        learned_planning_loop = lsp.planners.PlanningLoop(goal,
                                                          known_map,
                                                          simulator,
                                                          unity_bridge,
                                                          learned_robot,
                                                          args,
                                                          verbose=True)

        for counter, step_data in enumerate(learned_planning_loop):
            # Update the planner objects
            learned_planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])

            # Compute the subgoal and set
            learned_planning_loop.set_chosen_subgoal(
                learned_planner.compute_selected_subgoal())

        naive_robot = get_robot()
        naive_planning_loop = lsp.planners.PlanningLoop(goal,
                                                        known_map,
                                                        simulator,
                                                        unity_bridge,
                                                        naive_robot,
                                                        args,
                                                        verbose=True)

        for counter, step_data in enumerate(naive_planning_loop):
            non_learned_grid = step_data['robot_grid']

        learned_dist = common.compute_path_length(learned_robot.all_poses)
        naive_dist = common.compute_path_length(naive_robot.all_poses)
        did_succeed = learned_planning_loop.did_succeed and naive_planning_loop.did_succeed

        with open(logfile, "a+") as f:
            err_str = '' if did_succeed else '[ERR]'
            f.write(f"[Learn] {err_str} s: {args.current_seed:4d}"
                    f" | learned: {learned_dist:0.3f}"
                    f" | baseline: {naive_dist:0.3f}\n")

    # Write final plot to file
    image_file = os.path.join(args.save_dir, args.image_filename)

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(learned_planner.observed_map, known_map))
    path = learned_robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    plt.plot(ys, xs, 'r')
    plt.plot(path[-1].y, path[-1].x, 'go')
    plt.savefig(image_file, dpi=150)
    plt.title(f"Learned Cost: {common.compute_path_length(path):.2f}")

    plt.subplot(122)
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(non_learned_grid, known_map))
    path = naive_robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    plt.plot(ys, xs, 'r')
    plt.plot(path[-1].y, path[-1].x, 'go')
    plt.title(f"Naive Cost: {common.compute_path_length(path):.2f}")

    plt.savefig(image_file, dpi=150)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    from . import shared
    parser = shared.get_command_line_parser()
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--image_filename', type=str)
    args = parser.parse_args()

    evaluate_main(args)
