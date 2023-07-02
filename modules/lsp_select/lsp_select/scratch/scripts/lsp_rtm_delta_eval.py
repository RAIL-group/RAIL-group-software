import environments
import lsp
import lsp_select
import numpy as np
import matplotlib.pyplot as plt
from lsp.planners import LearnedSubgoalPlanner
from lsp_select.planners import LSPCycleGAN, RTMDelta
from lsp_select.utils.distribution import corrupt_robot_pose
from pathlib import Path


def maze_eval(args):
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    np.random.seed(args.current_seed)
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
        planners = [LearnedSubgoalPlanner(goal=goal, args=args),
                    LSPCycleGAN(goal=goal, args=args)]
        args.robot = robot
        args.robot_pose = pose
        planner = RTMDelta(goal, planners, args.priors, args)

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
                plt.title(f'RTMDelta - {planner.chosen_planner_name} ({args.env})')
                plt.xlabel(f'seed={args.current_seed}\n'
                           f'Avg. Estimated Cost: Dijkstra={planner.avg_estimated_costs[0]:.2f}, '
                           f'LSP={planner.avg_estimated_costs[1]:.2f}, '
                           f'LSPCycleGAN={planner.avg_estimated_costs[2]:.2f}')
                plt.show()
                plt.pause(0.01)

    return robot.net_motion, planner


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--generator_network_file', type=str)
    parser.add_argument('--disable_cyclegan', action='store_true')
    parser.add_argument('--env', choices=['envA', 'envB', 'envC'])

    args = parser.parse_args()

    args.planner = 'rtmdelta'
    args.priors = None

    for seed in range(*args.seed):
        cost_file = Path(args.save_dir) / f'cost_{args.planner}_{args.env}_{seed}.txt'
        priors_file = Path(args.save_dir) / f'priors_{args.planner}_{args.env}_{seed}.txt'
        if cost_file.is_file() and priors_file.is_file():
            print(f'Data already exists for {seed=}.')
            args.priors = np.genfromtxt(priors_file)[-1].reshape(-1, 2)
            continue

        print(f'Generating data for {seed=}.')
        args.current_seed = seed
        tot_distance, planner = maze_eval(args)

        priors = planner.compute_new_priors()
        chosen_planner = planner.chosen_planner_name
        with open(cost_file, 'w') as f:
            f.write(f'{tot_distance}')
        with open(priors_file, 'w') as f:
            np.savetxt(f, planner.priors.reshape(1, -1))
            np.savetxt(f, priors.reshape(1, -1))
        with open(Path(args.save_dir) / f'selected_{args.planner}_{args.env}_{seed}.txt', 'w') as f:
            f.write(f'{chosen_planner}')
        args.priors = priors
