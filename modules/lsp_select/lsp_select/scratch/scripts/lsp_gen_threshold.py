import environments
import lsp
import lsp_select
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lsp.planners import LearnedSubgoalPlanner
from sklearn.metrics import log_loss


def maze_eval(args):
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
        planner = LearnedSubgoalPlanner(goal=goal, args=args)

        subgoals_data = []
        for _, step_data in enumerate(planning_loop):
            planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])
            planning_loop.set_chosen_subgoal(planner.compute_selected_subgoal())

            # If goal is not visible (in which case chosen_subgoal is not None), collect data.
            if planning_loop.chosen_subgoal is not None:
                subgoals_data.append({
                    'subgoals': planner.subgoals,
                    'robot_pose': planner.robot_pose})

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
                plt.title(f'{type(planner).__name__} (seed={args.current_seed})')
                plt.show()
                plt.pause(0.01)

    final_grid = lsp.core.mask_grid_with_frontiers(planner.inflated_grid, planner.subgoals)
    return subgoals_data, final_grid, goal


def save_threshold(path):
    path = Path(path)
    files = path.glob('subgoal_probs_*.txt')
    subgoal_data = []
    for f in sorted(files):
        data = np.loadtxt(f)
        subgoal_data.extend(data)
    subgoal_data = np.array(subgoal_data)
    cross_entropy = log_loss(subgoal_data[:, 2], subgoal_data[:, 1])
    with open(path / 'threshold.txt', 'w') as f:
        f.write(f'{cross_entropy}')


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--generator_network_file', type=str)
    parser.add_argument('--disable_cyclegan', action='store_true')
    args = parser.parse_args()

    for seed in range(*args.seed):
        filepath = Path(args.save_dir) / f'subgoal_probs_{seed}.txt'
        if filepath.is_file():
            print(f'Data already exists for {seed=}.')
            continue

        print(f'Generating data for {seed=}.')
        args.current_seed = seed
        subgoals_data, final_grid, goal_pose = maze_eval(args)

        with open(filepath, 'w') as f:
            if len(subgoals_data) != 0:
                subgoal_labels = lsp_select.utils.misc.get_subgoal_labels(subgoals_data, final_grid, goal_pose)
                np.savetxt(f, subgoal_labels, fmt=['%d', '%.6f', '%d'])
            else:
                f.write('')
                print(f'No data to save for {seed=}.')

    save_threshold(args.save_dir)
