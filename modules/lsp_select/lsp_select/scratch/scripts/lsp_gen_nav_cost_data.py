import environments
import lsp
import lsp_select
import numpy as np
import matplotlib.pyplot as plt
import gridmap
from pathlib import Path
import scipy.stats as stats
from lsp_select.planners import LearnedSubgoalPlanner
from lsp_select.utils.distribution import get_cost_distribution, corrupt_robot_pose
import sys


def maze_eval(args):
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    np.random.seed(args.current_seed)
    pose = corrupt_robot_pose(known_map, args) if args.do_corrupt_pose else pose
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
        planner = LearnedSubgoalPlanner(goal=goal, args=args)
        probs_costs_motion = {'probs': [], 'costs': [], 'motion': []}
        for counter, step_data in enumerate(planning_loop):
            # Update the planner objects
            planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])
            if args.chosen_model == 'learned':
                chosen_frontier = planner.compute_selected_subgoal()
                planning_loop.set_chosen_subgoal(chosen_frontier)

            if planning_loop.chosen_subgoal is not None:
                distances = lsp.core.compute_distances(planner.latest_ordering,
                                                       step_data['robot_grid'],
                                                       step_data['robot_pose'],
                                                       goal)
                probs, costs = get_cost_distribution(planner.latest_ordering, distances)
                # expected_cost = planner.expected_cost
                probs_costs_motion['probs'].append(probs)
                probs_costs_motion['costs'].append(costs)
                probs_costs_motion['motion'].append(robot.net_motion)

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
                ax = plt.subplot(211)
                plt.imshow(step_data['image'])
                ax = plt.subplot(234)
                plt.title(f'Using {args.chosen_model}')
                lsp_select.utils.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_select.utils.plotting.plot_grid_with_frontiers(
                    ax, planner.observed_map, known_map, planner.subgoals)
                lsp_select.utils.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_select.utils.plotting.plot_path(ax, path)
                lsp_select.utils.plotting.plot_pose_path(ax, robot.all_poses)

                if planning_loop.chosen_subgoal is not None:
                    plt.subplot(235)
                    plt.title('Individual Gaussians')
                    pdfs = []
                    min_mu_idx, max_mu_idx = costs[:, 0].argmin(), costs[:, 0].argmax()
                    x = np.linspace(costs[min_mu_idx, 0] - 3 * costs[min_mu_idx, 1],
                                    costs[max_mu_idx, 0] + 3 * costs[max_mu_idx, 1], 1000)
                    for p, (mu, sigma) in zip(probs, costs):
                        y = stats.norm.pdf(x, mu, sigma)
                        plt.plot(x, y)
                        pdfs.append(p * y)
                    plt.subplot(236)
                    plt.plot(x, np.sum(np.array(pdfs), axis=0), color='tab:blue')
                    plt.title('Gaussian Mixture')

                plt.show()
                plt.pause(0.1)
        distance = []
        likelihood = []
        tot_distance = robot.net_motion
        for probs, costs, motion in zip(*probs_costs_motion.values()):
            # print(probs, costs, motion)
            mu, sigma = costs.T
            pdf = stats.norm.pdf(tot_distance - motion, mu, sigma)
            distance.append(tot_distance - motion)
            likelihood.append(np.sum(probs * pdf))
        # plt.savefig(Path(args.save_dir)/f'end_map.png', dpi=150)
        return distance, likelihood, tot_distance


def save_data_2(nll, tot_distance, model, do_corrupt_pose, args):
    filename = model + '_corrupted' if do_corrupt_pose else model + '_uncorrupted'
    with open(f'{Path(args.save_dir)/filename}.csv', 'a') as file:
        file.write(f'{args.current_seed},{tot_distance},{nll}\n')


def save_data(distance, likelihood, tot_distance, model, do_corrupt_pose, args):
    filename = model + '_corrupted' if do_corrupt_pose else model + '_uncorrupted'
    if model == 'non_learned':
        data, fmt = [tot_distance], '%.2f'
    else:
        data, fmt = np.vstack([distance, likelihood]).T, ['%.2f', '%.20e']
    np.savetxt(Path(args.save_dir) / 'data' / f'{filename}_{args.current_seed}.gz', data, fmt=fmt)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--no_of_maps', type=int)
    args = parser.parse_args()

    setup = [('learned', False),
             ('learned', True),
             ('non_learned', False),
             ('non_learned', True)]

    args.current_seed = args.seed[0]
    no_of_explored_maps = 0

    while no_of_explored_maps < args.no_of_maps:
        cost_data = {i: None for i in setup}
        for model, do_corrupt_pose in setup:
            args.chosen_model = model
            args.do_corrupt_pose = do_corrupt_pose
            stdout = sys.stdout
            sys.stdout = open('/dev/null', 'w')
            distance, likelihood, tot_distance = maze_eval(args)
            sys.stdout = stdout
            print(f'{args.current_seed} {model} {do_corrupt_pose}\n{tot_distance}, {-np.log(np.mean(likelihood))}')
            if args.chosen_model == 'learned' and len(likelihood) == 0:
                break
            cost_data[(model, do_corrupt_pose)] = distance, likelihood, tot_distance

        if None in cost_data.values():
            print(f'Skipping seed {args.current_seed}')
            args.current_seed += 1
            continue

        print(f'Map {no_of_explored_maps+1}/{args.no_of_maps} successfully saved with seed {args.current_seed}.')
        for (model, do_corrupt_pose), (distance, likelihood, tot_distance) in cost_data.items():
            save_data(distance, likelihood, tot_distance, model, do_corrupt_pose, args)

        args.current_seed += 1
        no_of_explored_maps += 1
