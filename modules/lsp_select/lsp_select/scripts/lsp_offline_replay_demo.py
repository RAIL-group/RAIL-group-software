import numpy as np
import matplotlib.pyplot as plt
import environments
import lsp
import lsp_select
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

        for counter, step_data in enumerate(planning_loop):
            planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])
            planning_loop.set_chosen_subgoal(planner.compute_selected_subgoal())

            if args.do_plot:
                plt.ion()
                plt.figure(1, figsize=(12, 8))
                plt.clf()
                plt.subplot(211)
                plt.imshow(step_data['image'])
                plt.axis('off')
                ax = plt.subplot(212)
                lsp_select.utils.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_select.utils.plotting.plot_grid_with_frontiers(
                    ax, planner.observed_map, known_map, planner.subgoals)
                lsp_select.utils.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)
                plt.title(f'Policy Selection - {args.chosen_planner} ({args.env}), Cost: {robot.net_motion:.2f}')
                plt.xlabel(f'seed={args.current_seed}')
                plt.axis('off')
                plt.savefig(Path(args.save_dir) / f'deploy_{args.chosen_planner}_{args.env}_'
                            f'{args.current_seed}_{counter}.png', dpi=300)
                plt.show()
                plt.pause(0.01)

    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.axis('off')
    plt.imshow(step_data['image'])
    plt.subplot(212)
    plt.axis('off')
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(planner.observed_map.T, known_map.T))
    path = robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    plt.plot(xs, ys, 'blue')
    plt.plot(pose.x, pose.y, 'bo')
    plt.plot(goal.x, goal.y, 'go')
    plt.title(f'{args.chosen_planner} ({args.env})')
    plt.xlabel(f'seed={args.current_seed}')
    plt.savefig(Path(args.save_dir) / f'img_{args.chosen_planner}_{args.env}_{args.current_seed}.png', dpi=300)

    return planner


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_path', type=str)
    parser.add_argument('--env', choices=['mazeA', 'office', 'officewall'])
    parser.add_argument('--chosen_planner', choices=['nonlearned', 'lspmaze', 'lspoffice', 'lspofficewallswap'])

    args = parser.parse_args()
    args.current_seed = args.seed[0]

    if args.env == 'mazeA' and args.map_type != 'maze':
        raise ValueError('map_type should be "maze" when env is "mazeA"')

    args.planner_names = ['nonlearned', 'lspmaze', 'lspoffice', 'lspofficewallswap']
    args.network_files = [None, 'mazeA/mazeA.pt', 'office/office_base.pt', 'office_wallswap/office_wallswap.pt']

    planner = maze_eval(args)
    costs, lb_costs = planner.get_costs()
    print(costs)
    print(lb_costs)
    cost_file = Path(args.save_dir) / f'cost_{args.chosen_planner}_{args.env}_{args.current_seed}.txt'
    lb_costs_file = Path(args.save_dir) / f'lbc_{args.chosen_planner}_{args.env}_{args.current_seed}.txt'
    with open(cost_file, 'w') as f:
        np.savetxt(f, costs)
    with open(lb_costs_file, 'w') as f:
        np.savetxt(f, lb_costs)
