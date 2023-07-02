import os
import gridmap
import lsp
import lsp_gnn
import environments
from lsp_gnn.planners import ConditionalKnownSubgoalPlanner

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

viridis = cm.get_cmap('viridis')
possible_choice = [True, False]


def navigate(args, do_plot=False, make_video=False):
    if args.map_type.lower() == 'ploader':
        known_map, map_data, pose, goal = \
            environments.generate.map_and_poses(args)
        map_data['wall_class'] = {
            'hallway': map_data['semantic_labels']['hallway'],
            'blue': 100,
            'red': map_data['semantic_labels']['room']}
    else:
        known_map, map_data, pose, goal = \
            lsp_gnn.environments.generate.map_and_poses(args)
    if make_video:
        fig = plt.figure()
        writer = animation.FFMpegWriter(8)
        writer.setup(fig, os.path.join(args.save_dir,
                     f'{args.data_file_base_name}_{args.current_seed}.mp4'),
                     500)

    use_known = random.choice(possible_choice)
    if use_known:
        print('Generating data using known planner')
        current_planner_str = 'Known'
    else:
        print('Generating data using Dijkstra planner')
        current_planner_str = 'Dijkstra'

    # # Instantiate the simulation environment
    # world = environments.simulated.OccupancyGridWorld(
    #     known_map,
    #     map_data,
    #     num_breadcrumb_elements=0,
    #     min_interlight_distance=3.0,
    #     min_light_to_wall_distance=1)
    # # builder = environments.simulated.WorldBuildingUnityBridge

    robot = lsp.robot.Turtlebot_Robot(pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=map_data)

    # Intialize and update the planner
    planner = ConditionalKnownSubgoalPlanner(
        goal, args, known_map,
        semantic_grid=map_data['semantic_grid'],
        wall_class=map_data['wall_class'])

    # with builder(args.unity_path) as unity_bridge:
    if True:  # This is here to avoid having to change indentation
        # unity_bridge.make_world(world)

        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             # unity_bridge=unity_bridge,
                                             world=None)
        simulator.frontier_grouping_inflation_radius = (
            simulator.inflation_radius)

        planning_loop = lsp.planners.PlanningLoop(
            goal, known_map, simulator, None, robot,
            args, verbose=False)

        for counter, step_data in enumerate(planning_loop):
            # Update the planner objects
            planner.update(
                {'image': step_data['image'],
                 'seg_image': step_data['seg_image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'])

            training_data = planner.compute_training_data()
            planner.save_training_data(training_data)
            chosen_subgoal = planner.compute_selected_subgoal()
            if use_known:
                planning_loop.set_chosen_subgoal(chosen_subgoal)
            if do_plot:
                # Mask grid with chosen subgoal (if not None)
                # and compute the cost grid for motion planning.
                if chosen_subgoal is not None:
                    planning_grid = lsp.core.mask_grid_with_frontiers(
                        planner.inflated_grid, planner.subgoals,
                        do_not_mask=chosen_subgoal)
                else:
                    planning_grid = lsp.core.mask_grid_with_frontiers(
                        planner.inflated_grid,
                        [],
                    )
                # Check that the plan is feasible and compute path
                cost_grid, get_path = gridmap.planning. \
                    compute_cost_grid_from_position(
                        planning_grid, [goal.x, goal.y], use_soft_cost=True)
                did_plan, path = get_path([robot.pose.x, robot.pose.y],
                                          do_sparsify=True,
                                          do_flip=True)

                # Plotting
                plt.ion()
                fig = plt.figure(1)
                fig_title = 'Seed: [' + str(args.current_seed) + \
                    '] <> Planner: [' + current_planner_str + ']'
                plt.clf()
                fig.suptitle(fig_title, fontsize='x-large')
                # Plot the classic lsp version
                ax = plt.subplot(121)
                ax.set_title('LSP')
                lsp_gnn.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_gnn.plotting.plot_grid(
                    ax, step_data['robot_grid'], known_map, planner.original_subgoal)
                lsp_gnn.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_gnn.plotting.plot_path(ax, path)
                lsp_gnn.plotting.plot_pose_path(ax, robot.all_poses)
                # Plot the skeletonized version
                ax = plt.subplot(122)
                ax.set_title('GCN-LSP')

                lsp_gnn.plotting.plot_pose(ax, robot.pose, color='blue')
                lsp_gnn.plotting.plot_grid(
                    ax, step_data['robot_grid'], known_map, None
                )
                lsp_gnn.plotting.plot_pose(
                    ax, goal, color='green', filled=False)

                is_subgoal = training_data['is_subgoal']
                prob_feasible = training_data['is_feasible']

                vertex_points = planner.vertex_points
                for vp_idx, ps in enumerate(vertex_points):
                    if not is_subgoal[vp_idx]:
                        color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
                        plt.plot(ps[0], ps[1], '+', color=color, markersize=3, markeredgecolor='r')
                for vp_idx, ps in enumerate(vertex_points):
                    if is_subgoal[vp_idx]:
                        color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
                        plt.plot(ps[0], ps[1], '.', color=color, markersize=4)
                for (start, end) in planner.edge_data:
                    p1 = vertex_points[start]
                    p2 = vertex_points[end]
                    x_values = [p1[0], p2[0]]
                    y_values = [p1[1], p2[1]]
                    plt.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)

                image_file = args.save_dir + 'data_gen_image' + '.png'
                plt.savefig(image_file, dpi=200)
                if make_video:
                    writer.grab_frame()
        open(os.path.join(
            args.save_dir,
            'data_completion_logs',
            f'{args.data_file_base_name}_{args.current_seed}.txt'), "x")
        if make_video:
            writer.finish()


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    args = lsp_gnn.utils.parse_args()
    if args.map_type == 'ploader':
        from os import listdir
        from os.path import isfile, join
        dir_str = '/data/lsp_gnn/university_building_floorplans/train'
        file_count = len(listdir(dir_str))
        onlyfiles = [
            join((dir_str + '/'), f)
            for f in listdir(dir_str)
            if isfile(join((dir_str + '/'), f))]
        args.map_file = onlyfiles
        args.cirriculum_fraction = None
    # Always freeze your random seeds
    torch.manual_seed(args.current_seed)
    np.random.seed(args.current_seed)
    random.seed(args.current_seed)
    # Generate Training Data
    navigate(args, make_video=False, do_plot=False)
