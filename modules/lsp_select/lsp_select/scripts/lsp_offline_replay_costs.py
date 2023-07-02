import environments
import lsp
import lsp_select
import numpy as np
import matplotlib.pyplot as plt
from lsp.planners import LearnedSubgoalPlanner, DijkstraPlanner
from lsp_select.planners import PolicySelectionPlanner
from lsp_select.utils.misc import corrupt_robot_pose
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

        args.robot = robot
        args.robot_pose = pose
        args.map_shape = known_map.shape
        args.known_map = known_map

        planners = []
        for network in args.network_files:
            if network is None:
                planners.append(DijkstraPlanner(goal=goal, args=args))
            else:
                args.network_file = str(Path(args.network_path) / network)
                planners.append(LearnedSubgoalPlanner(goal=goal, args=args))
        chosen_planner_idx = args.planner_names.index(args.chosen_planner)

        planner = PolicySelectionPlanner(goal, planners, chosen_planner_idx, args)

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
                plt.title(f'Policy Selection - {args.chosen_planner} ({args.env})')
                plt.xlabel(f'seed={args.current_seed}')
                plt.show()
                plt.pause(0.01)

    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.imshow(step_data['image'])
    plt.subplot(212)
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(planner.observed_map.T, known_map.T))
    path = robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    plt.plot(xs, ys, 'r')
    plt.plot(pose.x, pose.y, 'bo')
    plt.plot(goal.x, goal.y, 'go')
    plt.title(f'{args.chosen_planner} ({args.env})')
    plt.xlabel(f'seed={args.current_seed}')
    plt.savefig(Path(args.save_dir) / f'img_{args.chosen_planner}_{args.env}_{args.current_seed}.png', dpi=200)

    return planner


if __name__ == "__main__":
    maze_params = {
        'envs': ['envA', 'envB', 'envC'],
        'planners': ['nonlearned', 'lspA', 'lspB', 'lspC'],
        'network_files': [None, 'mazeA/VisLSPOriented.pt', 'mazeB/VisLSPOriented.pt', 'mazeC/VisLSPOriented.pt']
    }
    office_params = {
        'envs': ['mazeA', 'office', 'officewall'],
        'planners': ['nonlearned', 'lspmaze', 'lspoffice', 'lspofficewallswap'],
        'network_files': [None, 'mazeA/VisLSPOriented.pt', 'office/VisLSPOriented.pt', 'officewall/VisLSPOriented.pt']
    }
    env_params = {
        'maze': maze_params,
        'office': office_params
    }

    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--experiment_type', choices=['maze', 'office'])
    args, _ = parser.parse_known_args()
    EXPERIMENT_TYPE = args.experiment_type
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_path', type=str)
    parser.add_argument('--env', choices=env_params[EXPERIMENT_TYPE]['envs'])
    parser.add_argument('--chosen_planner', choices=env_params[EXPERIMENT_TYPE]['planners'])
    args = parser.parse_args()

    args.planner_names = env_params[EXPERIMENT_TYPE]['planners']
    args.network_files = env_params[EXPERIMENT_TYPE]['network_files']

    all_planners = '_'.join(args.planner_names)

    args.current_seed = args.seed[0]
    path = Path(args.save_dir)
    cost_file = path / f'cost_{args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.txt'
    err_file = path / f'error_{args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.txt'
    lb_costs_file = path / f'lbc_{args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.txt'
    target_file = path / f'target_plcy_{args.chosen_planner}_envrnmnt_{args.env}_{args.current_seed}.txt'

    if cost_file.is_file():
        print(f'Data already exists for {args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.')
        exit()
    if err_file.is_file():
        print(f'Error file exists for {args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.')
        exit()

    print(f'Generating data for {args.chosen_planner}_all_{all_planners}_{args.env}_{args.current_seed}.')

    planner = maze_eval(args)
    try:
        costs, lb_costs = planner.get_costs()

        with open(cost_file, 'w') as f:
            np.savetxt(f, costs)
        with open(lb_costs_file, 'w') as f:
            np.savetxt(f, lb_costs)
        with open(target_file, 'w') as f:
            f.write('\n')
    except TypeError as e:
        with open(err_file, 'w') as f:
            f.write(f'{e}')
            print(f'{e}')
        with open(target_file, 'w') as f:
            f.write('\n')
