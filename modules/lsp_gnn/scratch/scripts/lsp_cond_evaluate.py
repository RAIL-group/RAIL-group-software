import common
import os
import matplotlib.pyplot as plt
import numpy as np
import lsp
import lsp_cond
import environments
import matplotlib.animation as animation
from matplotlib import cm
from lsp_cond.planners import ConditionalUnknownSubgoalPlanner, \
    LSP, GCNLSP
from lsp.planners import KnownPlanner
import time as time

viridis = cm.get_cmap('viridis')


def evaluate_main(args, make_video=False):
    if args.map_type == 'jshaped':
        known_map, map_data, pose, goal = \
            lsp_cond.environments.generate.lsp_cond_map_and_poses(args)
    elif args.map_type == 'new_office':
        known_map, map_data, pose, goal = \
            lsp_cond.environments.generate.generate_map(args)
    if make_video:
        fig = plt.figure()
        writer = animation.FFMpegWriter(15)
        writer.setup(fig, os.path.join(args.save_dir, 
                     f'Eval_{args.current_seed}.mp4'), 500)
    # Randomize/Corrupt the initial robot pose
    if args.do_randomize_start_pose:
        if not args.map_type == 'maze':
            raise ValueError('Pose randomization allowed for Maze.')
        else:
            pose = lsp_cond.utils.get_path_middle_point(
                known_map, pose, goal, args)

    # Open the connection to Unity (if desired)
    if args.unity_path is None:
        raise ValueError('Unity Environment Required')

    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=0,
        min_interlight_distance=3.0,
        min_light_to_wall_distance=1)
    builder = environments.simulated.WorldBuildingUnityBridge

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
                                             world=world)
        simulator.frontier_grouping_inflation_radius = (
            simulator.inflation_radius)

        # ################
        # #  Conditional #
        # ################
        # learned_planner = ConditionalUnknownSubgoalPlanner(goal, args)
        # learned_robot = get_robot()
        # learned_planning_loop = lsp.planners.PlanningLoop(
        #     goal, known_map, simulator, unity_bridge, learned_robot,
        #     args, verbose=True)

        # for counter, step_data in enumerate(learned_planning_loop):
        #     # Update the planner objects
        #     s_time = time.time()
        #     learned_planner.update(
        #         {'image': step_data['image']},
        #         step_data['robot_grid'],
        #         step_data['subgoals'],
        #         step_data['robot_pose'])
        #     print(f"Time taken to update: {time.time() - s_time}")
        #     # Compute the subgoal and set
        #     s_time = time.time()
        #     chosen_subgoal = learned_planner.compute_selected_subgoal()
        #     print(f"Time taken to choose subgoal: {time.time() - s_time}")      
        #     learned_planning_loop.set_chosen_subgoal(chosen_subgoal)

        #     if make_video:
        #         # Mask grid with chosen subgoal (if not None)
        #         # and compute the cost grid for motion planning.
        #         if chosen_subgoal is not None:
        #             planning_grid = lsp.core.mask_grid_with_frontiers(
        #                 learned_planner.inflated_grid, learned_planner.subgoals, do_not_mask=chosen_subgoal)
        #         else:
        #             planning_grid = lsp.core.mask_grid_with_frontiers(
        #                 learned_planner.inflated_grid,
        #                 [],
        #             )
        #         # Check that the plan is feasible and compute path
        #         cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        #             planning_grid, [goal.x, goal.y], use_soft_cost=True)
        #         did_plan, path = get_path([learned_robot.pose.x, learned_robot.pose.y],
        #                                   do_sparsify=True,
        #                                   do_flip=True)

        #         # Plotting
        #         plt.ion()
        #         plt.figure(1)
        #         plt.clf()
        #         ax = plt.subplot(121)
        #         plt.imshow(step_data['image'])
        #         ax = plt.subplot(122)
        #         lsp_cond.plotting.plot_pose(ax, learned_robot.pose, color='blue')
        #         lsp_cond.plotting.plot_grid_with_frontiers(
        #             ax, step_data['robot_grid'], known_map, step_data['subgoals'])
        #         lsp_cond.plotting.plot_pose(ax, goal, color='green', filled=False)
        #         lsp_cond.plotting.plot_path(ax, path)
        #         lsp_cond.plotting.plot_pose_path(ax, learned_robot.all_poses)
        #         for (s, e) in learned_planner.edge_data:
        #             ps = learned_planner.current_graph[s][e]['pts']
        #             plt.plot(ps[:, 0], ps[:, 1], 'lightgray')
                
        #         is_subgoal = learned_planner.is_subgoal
        #         prob_feasible = learned_planner.out
        #         for vp_idx, ps in enumerate(learned_planner.vertex_points):
        #             if not is_subgoal[vp_idx]:
        #                 color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
        #                 plt.plot(ps[0], ps[1], '.', color=color, markersize=3, markeredgecolor='r')
        #         for vp_idx, ps in enumerate(learned_planner.vertex_points):
        #             if is_subgoal[vp_idx]:
        #                 color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
        #                 plt.plot(ps[0], ps[1], '.', color=color, markersize=4)
        #         # plt.show()  # TODO - add plot pause to render during planning
        #         writer.grab_frame()
        # if make_video:
        #     writer.finish()

        ################
        # ~~~ Known ~~ #
        ################
        base_planner = KnownPlanner(goal, known_map, args)
        base_robot = get_robot()
        base_planning_loop = lsp.planners.PlanningLoop(
            goal, known_map, simulator, unity_bridge, base_robot, 
            args, verbose=True)

        for counter, step_data in enumerate(base_planning_loop):
            # Update the planner objects
            base_planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'])
            chosen_subgoal = base_planner.compute_selected_subgoal()
            base_planning_loop.set_chosen_subgoal(chosen_subgoal)

        ################
        # ~~~ Naive ~~ #
        ################
        naive_robot = get_robot()
        naive_planning_loop = lsp.planners.PlanningLoop(
            goal, known_map, simulator, unity_bridge, naive_robot,
            args, verbose=True)

        for counter, step_data in enumerate(naive_planning_loop):
            naive_observed_map = step_data['robot_grid']

        ################
        # ~~ CNN LSP ~ #
        ################
        lsp_planner = LSP(
            goal, args,
            semantic_grid=map_data['semantic_grid'],
            wall_class=map_data['wall_class'])
        lsp_robot = get_robot()
        lsp_planning_loop = lsp.planners.PlanningLoop(
            goal, known_map, simulator, unity_bridge, lsp_robot, 
            args, verbose=True)

        for counter, step_data in enumerate(lsp_planning_loop):
            # Update the planner objects
            lsp_planner.update(
                {'image': step_data['image'],
                 'seg_image': step_data['seg_image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'], None)
            # Compute the subgoal and set
            chosen_subgoal = lsp_planner.compute_selected_subgoal()
            lsp_planning_loop.set_chosen_subgoal(chosen_subgoal)

        ################
        # ~~ GCN LSP ~ #
        ################
        gcn_lsp_planner = GCNLSP(
            goal, args,
            semantic_grid=map_data['semantic_grid'],
            wall_class=map_data['wall_class'])
        gcn_lsp_robot = get_robot()
        gcn_lsp_planning_loop = lsp.planners.PlanningLoop(
            goal, known_map, simulator, unity_bridge, gcn_lsp_robot,
            args, verbose=True)

        for counter, step_data in enumerate(gcn_lsp_planning_loop):
            # Update the planner objects
            gcn_lsp_planner.update(
                {'image': step_data['image'],
                 'seg_image': step_data['seg_image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'])
            # Compute the subgoal and set
            chosen_subgoal = gcn_lsp_planner.compute_selected_subgoal()
            gcn_lsp_planning_loop.set_chosen_subgoal(chosen_subgoal)

        # learned_dist = common.compute_path_length(learned_robot.all_poses)
        base_dist = common.compute_path_length(base_robot.all_poses)
        naive_dist = common.compute_path_length(naive_robot.all_poses)
        lsp_dist = common.compute_path_length(lsp_robot.all_poses)
        gcn_lsp_dist = common.compute_path_length(gcn_lsp_robot.all_poses)
        did_succeed = gcn_lsp_planning_loop.did_succeed  # \
        # and naive_planning_loop.did_succeed

        with open(logfile, "a+") as f:
            err_str = '' if did_succeed else '[ERR]'
            f.write(f"[Learn] {err_str} s: {args.current_seed:4d}"
                    # f" | cond_lsp: {learned_dist:0.3f}"
                    f" | baseline: {base_dist:0.3f}"
                    f" | naive: {naive_dist:0.3f}"
                    f" | lsp: {lsp_dist:0.3f}"
                    f" | gcn_lsp: {gcn_lsp_dist:0.3f}\n")

    if gcn_lsp_planner.observed_map is None:
        # learned_planner.observed_map = -1 * np.ones_like(known_map)
        # base_planner.observed_map = -1 * np.ones_like(known_map)
        # naive_observed_map = -1 * np.ones_like(known_map)
        lsp_planner.observed_map = -1 * np.ones_like(known_map)
        gcn_lsp_planner.observed_map = -1 * np.ones_like(known_map)

    # Write final plot to file
    image_file = os.path.join(args.save_dir, args.image_filename)
    plt.figure(figsize=(16, 16))
    # plt.subplot(231)
    # plt.imshow(
    #    lsp.utils.plotting.make_plotting_grid(
    #        learned_planner.observed_map, known_map))
    # path = learned_robot.all_poses
    # xs = [p.x for p in path]
    # ys = [p.y for p in path]
    # plt.plot(ys, xs, 'r')
    # plt.plot(path[-1].y, path[-1].x, 'go')
    # plt.title(f"Cond LSP Cost: {common.compute_path_length(path):.2f}")
    # plt.savefig(image_file, dpi=150)

    plt.subplot(221)
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(
            base_planner.observed_map, known_map))
    path = base_robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    plt.plot(ys, xs, 'r')
    plt.plot(path[-1].y, path[-1].x, 'go')
    plt.title(f"Known Cost: {common.compute_path_length(path):.2f}")
    plt.savefig(image_file, dpi=150)

    plt.subplot(222)
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(naive_observed_map, known_map))
    path = naive_robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    plt.plot(ys, xs, 'r')
    plt.plot(path[-1].y, path[-1].x, 'go')
    plt.title(f"Naive Cost: {common.compute_path_length(path):.2f}")
    plt.savefig(image_file, dpi=150)

    plt.subplot(223)
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(
            lsp_planner.observed_map, known_map))
    path = lsp_robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    plt.plot(ys, xs, 'r')
    plt.plot(path[-1].y, path[-1].x, 'go')
    plt.title(f"CNN LSP Cost: {common.compute_path_length(path):.2f}")
    plt.savefig(image_file, dpi=150)

    plt.subplot(224)
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(
            gcn_lsp_planner.observed_map, known_map))
    path = gcn_lsp_robot.all_poses
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    plt.plot(ys, xs, 'r')
    plt.plot(path[-1].y, path[-1].x, 'go')
    plt.title(f"GNN LSP Cost*: {common.compute_path_length(path):.2f}")
    plt.savefig(image_file, dpi=150)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    args = lsp_cond.utils.parse_args()
    print(f'Evaluation seed: [{args.current_seed}]')
    evaluate_main(args)
