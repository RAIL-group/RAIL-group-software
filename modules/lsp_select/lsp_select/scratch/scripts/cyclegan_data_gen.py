import environments
import lsp
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import lsp_select
from lsp.planners import DijkstraPlanner
from lsp_select.utils.distribution import corrupt_robot_pose


def maze_eval(args):
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    goal = corrupt_robot_pose(known_map, args) if args.envC else goal
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
        planner = DijkstraPlanner(goal=goal, args=args)

        for counter, step_data in enumerate(planning_loop):
            # Update the planner objects
            planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])
            planning_loop.set_chosen_subgoal(None)

            # Save image
            im = Image.fromarray(step_data['image'])
            im.save(Path(args.save_dir) / f'img_{args.current_seed}_{counter}.png')

            # if args.do_plot:
            if True:
                plt.ion()
                plt.figure(1)
                plt.clf()
                # plt.subplot(211)
                # plt.imshow(step_data['image'])
                ax = plt.subplot(111)
                lsp_select.utils.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_select.utils.plotting.plot_grid_with_frontiers(
                    ax, planner.observed_map, known_map, planner.subgoals)
                lsp_select.utils.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_select.utils.plotting.plot_path(ax, planning_loop.current_path)
                lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)
                plt.savefig(Path(args.save_dir) / f'grid_{args.current_seed}_{counter}.pdf', dpi=300)
                plt.show()
                plt.pause(0.01)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--envC', action='store_true')
    args = parser.parse_args()
    for seed in range(args.seed[0], args.seed[1]):
        print(f'{seed=}')
        args.current_seed = seed
        maze_eval(args)
