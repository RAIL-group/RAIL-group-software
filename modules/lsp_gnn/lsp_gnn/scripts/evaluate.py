import os
import common
import numpy as np
import time as time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

import lsp
import gridmap
import lsp_gnn
import environments
from lsp_gnn.planners import ConditionalUnknownSubgoalPlanner, GCNLSP, LSP

viridis = cm.get_cmap('viridis')


def evaluate_main(args, make_video=False):
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
        writer = animation.FFMpegWriter(12)
        writer.setup(fig, os.path.join(args.save_dir,
                     f'Eval_{args.current_seed}.mp4'), 500)

    # # Initialize the world and builder objects
    # world = environments.simulated.OccupancyGridWorld(
    #     known_map,
    #     map_data,
    #     num_breadcrumb_elements=0,
    #     min_interlight_distance=3.0,
    #     min_light_to_wall_distance=1)
    # # builder = environments.simulated.WorldBuildingUnityBridge

    # Helper function for creating a new robot instance
    def get_robot():
        return lsp.robot.Turtlebot_Robot(pose,
                                         primitive_length=args.step_size,
                                         num_primitives=args.num_primitives,
                                         map_data=map_data)

    # with builder(args.unity_path) as unity_bridge:
    if True:
        unity_bridge = None
        # unity_bridge.make_world(world)

        # Write starting seed to the log file
        logfile = os.path.join(args.save_dir, args.logfile_name)
        with open(logfile, "a+") as f:
            f.write(f"LOG: {args.current_seed}\n")

        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=None)
        simulator.frontier_grouping_inflation_radius = (
            simulator.inflation_radius)

        if args.logfile_name == 'clsp_logfile.txt':
            planner = ConditionalUnknownSubgoalPlanner(goal, args)
            cost_str = 'cond_lsp'
        elif args.logfile_name == 'mlsp_logfile.txt':
            planner = GCNLSP(
                goal, args,
                semantic_grid=map_data['semantic_grid'],
                wall_class=map_data['wall_class'])
            cost_str = 'gcn_lsp'
        elif args.logfile_name == 'lsp_logfile.txt':
            planner = LSP(
                goal, args,
                semantic_grid=map_data['semantic_grid'],
                wall_class=map_data['wall_class'])
            cost_str = 'lsp'
        robot = get_robot()
        planning_loop = lsp.planners.PlanningLoop(
            goal, known_map, simulator, unity_bridge, robot,
            args, verbose=True)

        for counter, step_data in enumerate(planning_loop):
            # Update the planner objects
            s_time = time.time()
            planner.update(
                {'image': step_data['image'],
                 'seg_image': step_data['seg_image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'])
            print(f"Time taken to update: {time.time() - s_time}")
            # Compute the subgoal and set
            s_time = time.time()
            chosen_subgoal = planner.compute_selected_subgoal()
            print(f"Time taken to choose subgoal: {time.time() - s_time}")
            planning_loop.set_chosen_subgoal(chosen_subgoal)

            if make_video and args.logfile_name != 'lsp_logfile.txt':
                # Mask grid with chosen subgoal (if not None)
                # and compute the cost grid for motion planning.
                if chosen_subgoal is not None:
                    planning_grid = lsp.core.mask_grid_with_frontiers(
                        planner.inflated_grid, planner.subgoals, do_not_mask=chosen_subgoal)
                else:
                    planning_grid = lsp.core.mask_grid_with_frontiers(
                        planner.inflated_grid,
                        [],
                    )
                # Check that the plan is feasible and compute path
                cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                    planning_grid, [goal.x, goal.y], use_soft_cost=True)
                did_plan, path = get_path([robot.pose.x, robot.pose.y],
                                          do_sparsify=True,
                                          do_flip=True)

                # Plotting
                plt.ion()
                fig = plt.figure(1)
                plt.clf()
                fig_title = 'Seed: [' + str(args.current_seed) + \
                            '] <> Planner: [' + cost_str + ']'
                fig.suptitle(fig_title, fontsize='x-large')
                ax = plt.subplot(121)
                lsp_gnn.plotting.plot_semantic_grid_with_frontiers(
                    ax, step_data['robot_grid'], None, planner.subgoals,
                    map_data['semantic_grid'], map_data['wall_class'])
                lsp_gnn.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_gnn.plotting.plot_path(ax, path)
                lsp_gnn.plotting.plot_pose_path(ax, robot.all_poses)
                lsp_gnn.plotting.plot_pose(ax, robot.pose, color='magenta')
                is_subgoal = planner.is_subgoal
                prob_feasible = planner.out

                ax = plt.subplot(122)
                lsp_gnn.plotting.plot_grid(
                    ax, step_data['robot_grid'], known_map, None
                )
                lsp_gnn.plotting.plot_pose(ax, goal, color='green', filled=False)
                lsp_gnn.plotting.plot_path(ax, path)
                lsp_gnn.plotting.plot_pose_path(ax, robot.all_poses)
                lsp_gnn.plotting.plot_pose(ax, robot.pose, color='magenta')
                for vp_idx, ps in enumerate(planner.vertex_points):
                    if not is_subgoal[vp_idx]:
                        color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
                        plt.plot(ps[0], ps[1], '+', color=color, markersize=3, markeredgecolor='r')
                for vp_idx, ps in enumerate(planner.vertex_points):
                    if is_subgoal[vp_idx]:
                        color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
                        plt.plot(ps[0], ps[1], '.', color=color, markersize=4)
                for (start, end) in planner.edge_data:
                    p1 = planner.vertex_points[start]
                    p2 = planner.vertex_points[end]
                    x_values = [p1[0], p2[0]]
                    y_values = [p1[1], p2[1]]
                    plt.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)

                writer.grab_frame()
        if make_video:
            writer.finish()
        dist = common.compute_path_length(robot.all_poses)
        did_succeed = planning_loop.did_succeed

        with open(logfile, "a+") as f:
            err_str = '' if did_succeed else '[ERR]'
            f.write(f"[Learn] {err_str} s: {args.current_seed:4d}"
                    f" | {cost_str}: {dist:0.3f}\n")

    if planner.observed_map is None:
        planner.observed_map = -1 * np.ones_like(known_map)

    # Write final plot to file
    image_file = os.path.join(args.save_dir, args.image_filename)
    plt.figure(figsize=(4, 4))
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(planner.observed_map, known_map))
    path = robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    plt.plot(ys, xs, 'r')
    plt.plot(path[-1].y, path[-1].x, 'go')
    plt.title(f"Cost: {common.compute_path_length(path):.2f}")
    plt.savefig(image_file, dpi=150)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    args = lsp_gnn.utils.parse_args()
    if args.map_type == 'ploader':
        from os import listdir
        from os.path import isfile, join
        dir_str = '/data/lsp_gnn/university_building_floorplans/test'
        file_count = len(listdir(dir_str))
        onlyfiles = [
            join((dir_str + '/'), f)
            for f in listdir(dir_str)
            if isfile(join((dir_str + '/'), f))]
        args.map_file = onlyfiles
        args.cirriculum_fraction = None
    print(f'Evaluation seed: [{args.current_seed}]')
    evaluate_main(args, make_video=True)
