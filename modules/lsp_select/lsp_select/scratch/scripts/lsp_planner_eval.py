import environments
import lsp
import lsp_select
import matplotlib.pyplot as plt
from lsp.planners import LearnedSubgoalPlanner, DijkstraPlanner
from lsp_select.planners import LSPCycleGAN
from lsp_select.utils.distribution import corrupt_robot_pose
from pathlib import Path


def maze_eval(args):
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    goal = corrupt_robot_pose(known_map, args) if args.env == 'envC' else goal
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements,
        min_breadcrumb_signed_distance=4.0 * args.base_resolution)
    builder = environments.simulated.WorldBuildingUnityBridge
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
        simulator.frontier_grouping_inflation_radius = simulator.inflation_radius
        planning_loop = lsp.planners.PlanningLoop(goal,
                                                  known_map,
                                                  simulator,
                                                  unity_bridge=None,
                                                  robot=robot,
                                                  args=args,
                                                  verbose=True)

        planner_dict = {'lspA': LearnedSubgoalPlanner,
                        'lspB': LearnedSubgoalPlanner,
                        'lspC': LearnedSubgoalPlanner,
                        'lspcyclegan': LSPCycleGAN,
                        'nonlearned': DijkstraPlanner}
        planner = planner_dict[args.planner](goal=goal, args=args)

        for _, step_data in enumerate(planning_loop):
            planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])
            planning_loop.set_chosen_subgoal(planner.compute_selected_subgoal())

            if args.do_plot:
                plt.ion()
                plt.figure(1)
                plt.clf()
                plt.subplot(211)
                plt.imshow(step_data['image'])
                ax = plt.subplot(212)
                lsp_select.utils.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_select.utils.plotting.plot_grid_with_frontiers(
                    ax, planner.observed_map, known_map, planner.subgoals)
                lsp_select.utils.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_select.utils.plotting.plot_path(ax, planning_loop.current_path)
                lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)
                plt.title(f'{type(planner).__name__} ({args.env})')
                plt.xlabel(f'seed={args.current_seed}')
                plt.show()
                plt.pause(0.01)
    plt.close()
    plt.ioff()
    plt.figure(2)
    ax = plt.subplot(111)
    lsp_select.utils.plotting.plot_pose(ax, robot.pose, color='blue')
    lsp_select.utils.plotting.plot_grid_with_frontiers(
        ax, planner.observed_map, known_map, planner.subgoals)
    lsp_select.utils.plotting.plot_pose(ax, goal, color='green', filled=False)
    lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)
    plt.title(f'Cost: {robot.net_motion:.2f}')
    plt.xlabel(f'seed={args.current_seed}')
    plt.savefig(Path(args.save_dir) / f'plan_{args.planner}_{args.env}_{args.current_seed}.png')

    return robot.net_motion


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--generator_network_file', type=str)
    parser.add_argument('--disable_cyclegan', action='store_true')
    parser.add_argument('--planner', choices=['lspA', 'lspB', 'lspC', 'lspcyclegan', 'nonlearned'])
    parser.add_argument('--env', choices=['envA', 'envB', 'envC'])
    args = parser.parse_args()

    for seed in range(*args.seed):
        filepath = Path(args.save_dir) / f'cost_{args.planner}_{args.env}_{seed}.txt'
        if filepath.is_file():
            print(f'Data already exists for {args.planner}_{args.env}_{seed}.')
            continue

        print(f'Generating data for {args.planner}_{args.env}_{seed}.')
        args.current_seed = seed
        tot_distance = maze_eval(args)

        with open(filepath, 'w') as f:
            f.write(f'{tot_distance}')
