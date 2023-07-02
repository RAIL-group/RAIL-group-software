import environments
import lsp
import lsp_select
import matplotlib.pyplot as plt

from lsp.planners import KnownSubgoalPlanner, LearnedSubgoalPlanner


def test_maze_navigation(unity_path, do_debug_plot, lsp_select_network_file):
    parser = lsp.utils.command_line.get_parser()
    args = parser.parse_args(['--save_dir', ''])
    args.current_seed = 100
    args.step_size = 1.8
    args.field_of_view_deg = 360
    args.map_type = 'maze'
    args.base_resolution = 0.3
    args.inflation_radius_m = 0.75
    args.laser_max_range_m = 18
    args.unity_path = unity_path
    args.network_file = lsp_select_network_file
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
        if args.network_file:
            planner = LearnedSubgoalPlanner(goal=goal, args=args)
        else:
            planner = KnownSubgoalPlanner(goal=goal, known_map=known_map, args=args)

        for _, step_data in enumerate(planning_loop):
            planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])
            planning_loop.set_chosen_subgoal(planner.compute_selected_subgoal())

            if do_debug_plot:
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
                plt.title(f'{type(planner).__name__}')
                plt.show()
                plt.pause(0.01)
