import numpy as np
import matplotlib.pyplot as plt
import environments
import lsp
import lsp_select
from lsp.planners import KnownSubgoalPlanner, DijkstraPlanner
from lsp_select.planners import PolicySelectionPlanner


def test_offline_replay(unity_path, do_debug_plot):
    parser = lsp.utils.command_line.get_parser()
    args = parser.parse_args(['--save_dir', ''])
    args.current_seed = 400
    args.step_size = 1.8
    args.field_of_view_deg = 360
    args.map_type = 'maze'
    args.base_resolution = 0.3
    args.inflation_radius_m = 0.75
    args.laser_max_range_m = 18
    args.unity_path = unity_path
    args.do_plot = do_debug_plot
    args.planner_names = ['nonlearned', 'known']
    args.chosen_planner = 'known'
    args.env = 'Maze'

    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
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

        planners = [DijkstraPlanner(goal=goal, args=args),
                    KnownSubgoalPlanner(goal=goal, known_map=known_map, args=args)]
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
                plt.title(f'Policy Selection - {args.chosen_planner} ({args.env}), Cost: {robot.net_motion:.2f}')
                plt.xlabel(f'seed={args.current_seed}')
                plt.show()
                plt.pause(0.01)

        costs, lb_costs = planner.get_costs()

        assert np.isnan(costs[0])
        assert np.allclose(costs[1], 176.4)
        assert np.allclose(lb_costs[0], [440.41814055, 210.6])
        assert np.all(np.isnan(lb_costs[1]))
