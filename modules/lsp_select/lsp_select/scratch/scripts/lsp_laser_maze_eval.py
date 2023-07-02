import environments
import lsp
import lsp_select
import matplotlib.pyplot as plt
import gridmap

from lsp_select.planners import LearnedSubgoalPlannerLaser


def maze_eval(args):

    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements)
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
        planner = LearnedSubgoalPlannerLaser(goal=goal, args=args)
        for counter, step_data in enumerate(planning_loop):
            # Update the planner objects
            planner.update(
                {'image': step_data['image'], 'ranges': step_data['ranges']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])
            planning_loop.set_chosen_subgoal(planner.compute_selected_subgoal())

            if args.do_plot:
                if planning_loop.chosen_subgoal is not None:
                    planning_grid = lsp.core.mask_grid_with_frontiers(
                        planner.inflated_grid,
                        planner.subgoals,
                        do_not_mask=planning_loop.chosen_subgoal)
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
                plt.ion()
                plt.figure(1)
                plt.clf()
                if step_data['image'] is not None:
                    ax = plt.subplot(211)
                    plt.imshow(step_data['image'])
                    ax = plt.subplot(212)
                else:
                    ax = plt.subplot(111)
                lsp_select.utils.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_select.utils.plotting.plot_grid_with_frontiers(
                    ax, planner.observed_map, known_map, planner.subgoals)
                lsp_select.utils.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_select.utils.plotting.plot_path(ax, path)
                lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)
                plt.show()
                plt.pause(0.01)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_file', type=str)
    args = parser.parse_args()
    args.current_seed = args.seed[0]
    maze_eval(args)
