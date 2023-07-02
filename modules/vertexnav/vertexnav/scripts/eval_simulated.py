"""Generate figures for the simulated hallway environments"""
import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import environments
import vertexnav_accel
import vertexnav
import math
import numpy as np
import os
import random
import time as time
import torch
import shapely


def set_up_video_gen(video_path, figure_only=False):
    """Helper function. Returns axes and writer."""
    fig = plt.figure(figsize=(4, 8))
    fig.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=0.9,
                        wspace=None,
                        hspace=None)

    # For video
    ax1 = plt.subplot2grid((5, 1), (0, 0), colspan=1, rowspan=2)
    ax2 = plt.subplot2grid((5, 1), (2, 0), colspan=1, rowspan=3)

    ax1.autoscale(enable=True, axis='x', tight=True)
    ax2.set_aspect('equal')
    ax2.autoscale(enable=True, axis='x', tight=True)

    if figure_only:
        return ax1, ax2
    else:
        writer = animation.FFMpegWriter(15)
        writer.setup(fig, os.path.join(video_path), 500)
        return (ax1, ax2), writer


def get_environment(args):
    if args.environment == 'dungeon':
        inflation_rad = 6.0
        world = vertexnav.environments.dungeon.DungeonWorld(
            hall_width=20, inflate_ratio=0.30, random_seed=args.current_seed,

        )
        builder = environments.simulated.WorldBuildingUnityBridge
        path = [world.get_random_pose(min_signed_dist=inflation_rad)]
    elif args.environment == 'outdoor':
        # inflation_rad = 6.0
        # world = vertexnav.environments.simulated.OutdoorWorld(
        #     num_buildings=5,
        #     num_clutter_elements=15,
        #     min_clutter_signed_distance=20.0)
        # builder = vertexnav.environments.simulated.OutdoorBuildingUnityBridge
        # path = [world.get_random_pose(min_signed_dist=inflation_rad,
        #                               max_signed_dist=2*inflation_rad)]
        raise NotImplementedError()
    elif args.environment == 'guidedmaze':
        # from lsp.utils.mapgen import generate_map_and_poses
        # args.base_resolution = args.grid_resolution
        # args.inflation_radius_m = 2.5 * args.grid_resolution
        # args.map_type = 'maze'
        # args.current_seed = args.seed
        # args.map_maze_path_width = 10
        # args.map_maze_cell_dims = [7, 7]
        # args.map_maze_wide_path_width = 14
        # args.map_maze_all_wide = True
        # grid, map_data, pose, goal = generate_map_and_poses(args)
        # inflation_rad = args.base_resolution * 4.0
        # world = vertexnav.environments.simulated.OccupancyGridWorld(grid, map_data)
        # builder = vertexnav.environments.simulated.DungeonBuildingUnityBridge
        # path = [vertexnav.Pose(pose.x, pose.y)]
        raise NotImplementedError()
    else:
        raise ValueError("Environment '{}' unrecognized.".format(
            args.environment))

    return world, builder, inflation_rad, path


def get_data_iterator(unity_bridge,
                      world,
                      path,
                      segment_steps=50,
                      smooth_factor=1.0 / 3,
                      robots=None,
                      do_return_time=False):
    path_iterator = vertexnav.utils.data.follow_path_iterator(
        path, segment_steps, smooth_factor)

    prev_poses = None

    while True:
        iter_start_time = time.time()
        if robots is not None:
            poses = [robot.pose for robot in robots]
        else:
            poses = [next(path_iterator)]

        if prev_poses is None:
            odoms = [None for robot in robots]
        else:
            odoms = [
                vertexnav.Pose.get_odom(p_new=pn, p_old=po)
                for pn, po in zip(poses, prev_poses)
            ]

        prev_poses = [
            vertexnav.Pose(p.x, p.y, p.yaw, p.robot_id) for p in poses
        ]

        # Get the images
        def get_image(p):
            unity_bridge.move_object_to_pose("robot", p)
            pano_image = unity_bridge.get_image("robot/pano_camera")
            pano_image = vertexnav.utils.convert.image_aligned_to_robot(
                image=pano_image, r_pose=p)
            return pano_image[64:-64] * 1.0 / 255

        pano_images = [get_image(p) for p in poses]

        if do_return_time:
            yield poses, odoms, pano_images, time.time() - iter_start_time
        else:
            yield poses, pano_images


def write_video_frame(axs,
                      writer,
                      counter,
                      pose,
                      pano_images,
                      net_data,
                      obs,
                      nvg,
                      perfect_nvg,
                      plot_data,
                      args,
                      figure_only=False):
    ax1, ax3 = axs

    do_plot_frontier_paths = True
    do_plot_vis_polys = False

    s = pano_images[0].shape
    buf = np.ones([8, s[1], 3])

    composite_image = pano_images[0]

    for im in pano_images[1:]:
        composite_image = np.concatenate((composite_image, buf, im), axis=0)

    ax1.clear()
    ax1.axis('off')
    ax1.imshow(composite_image)
    ax1.set_xlim(0, composite_image.shape[1])
    ax1.set_ylim(composite_image.shape[0], 0)

    ax3.clear()
    ax3.set_facecolor([0.8, 0.8, 0.8])
    ax3.get_xaxis().set_ticks([])
    ax3.get_yaxis().set_ticks([])

    proposed_world = nvg.get_proposed_world()
    known_space_poly = plot_data['known_space_poly']

    vertexnav.plotting.plot_polygon(ax3,
                                    known_space_poly,
                                    color=[1.0, 1.0, 1.0],
                                    alpha=1.0)

    # Plot the clusters
    vertex_remapping = vertexnav_accel.get_vertex_remapping(
        nvg.vertices, set(tuple(c) for c in nvg.topology))
    vertex_id_dict = {v.id: v for v in nvg.vertices}
    clusters = [
        vertexnav.prob_vertex_graph.Cluster(c, vertex_id_dict)
        for c in nvg.topology
    ]
    for cluster in clusters:
        if not cluster.is_active and cluster.num_dets > 2:
            vert = vertex_remapping[cluster.vids[0]]
            ax3.plot(vert.position[0],
                     vert.position[1],
                     'r.',
                     alpha=0.4,
                     markersize=vertexnav.plotting.SCATTERSIZE * 2)

    vertexnav.plotting.plot_world(ax3, plot_data['world'], alpha=0.2)

    if do_plot_frontier_paths:
        plot_data['frontier_paths'] = sorted(plot_data['frontier_paths'],
                                             key=lambda d: id(d[2]))

        for ii in range(args.num_robots):
            perfect_points = np.array([
                (p.x, p.y, p.yaw, p.index)
                for p in perfect_nvg.r_poses[ii::args.num_robots]
            ])
            ax3.plot(perfect_points[:, 0],
                     perfect_points[:, 1],
                     '--k',
                     alpha=0.5)

        for ii, (f, path, f_robot) in enumerate(plot_data['frontier_paths']):
            points = vertexnav.utils.calc.smooth_path(path)
            line, = ax3.plot(points[:, 0],
                             points[:, 1],
                             alpha=0.5,
                             linewidth=1.5)
            if f is not None:
                vertexnav.plotting.plot_linestring(ax3,
                                                   f.linestring,
                                                   color=line.get_color(),
                                                   alpha=1.0,
                                                   linewidth=1.0,
                                                   linestyle='--')
            r_points = np.array([
                (p.x, p.y, p.yaw, p.index)
                for p in nvg.r_poses[ii::args.num_robots]
            ])
            ax3.plot(r_points[:, 0],
                     r_points[:, 1],
                     color=line.get_color(),
                     alpha=0.75)

            ax3.plot(f_robot.pose.x,
                     f_robot.pose.y,
                     '.',
                     color=line.get_color())
            ax3.plot(f_robot.pose.x,
                     f_robot.pose.y,
                     'x',
                     color=line.get_color())

    vertexnav.plotting.plot_proposed_world(ax3,
                                           proposed_world,
                                           do_show_points=True,
                                           do_plot_visibility=False,
                                           robot_pose=pose)

    if do_plot_vis_polys:
        poly_points = vertexnav.noisy.compute_conservative_space_from_obs(
            nvg.r_poses[-1], nvg.observations[-1], radius=args.max_range)
        obs_space = shapely.geometry.Polygon(poly_points)
        vertexnav.plotting.plot_polygon(ax3,
                                        obs_space,
                                        color=[1.0, 0.0, 0.0],
                                        alpha=0.4)

        h_obs = vertexnav.noisy.compute_hypothetical_observation(
            proposed_world,
            nvg.r_poses[-1],
            nvg.observations[-1],
            radius=args.max_range)
        poly_points = vertexnav.noisy.compute_conservative_space_from_obs(
            nvg.r_poses[-1], h_obs, radius=args.max_range)
        obs_space = shapely.geometry.Polygon(poly_points)
        vertexnav.plotting.plot_polygon(ax3,
                                        obs_space,
                                        color=[0.0, 0.0, 1.0],
                                        alpha=0.3)

    xbounds, ybounds = plot_data['world'].map_bounds
    ax3.set_xlim(xbounds[0] - 15, xbounds[1] + 15)
    ax3.set_ylim(ybounds[1] + 15, ybounds[0] - 15)

    if not figure_only:
        writer.grab_frame()


def travel_and_make_plot(args):
    """Integration test running the robot through a simple environment,
    adding new vertext detections, and performing inference to determine
    the most likely map"""

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Neural Net Compute Device: {device}")
    # eval_net = vertexnav.learning.load_pytorch_net_vertex(
    #     args.network_file, device)
    eval_net = vertexnav.models.VertexNavGrid.get_net_eval_fn(
        network_file=args.network_file, device=device)

    save_dir = os.path.dirname(args.figure_path)
    log_text_file_name = os.path.join(save_dir, "data_log.txt")
    logs = {
        "r_poses": [],
        "net_time": [],
        "perfect_coverage": [],
        "noisy_coverage": [],
        "final_map": None
    }

    def get_range(position, obs_pose):
        return math.sqrt((position[1] - obs_pose.y)**2 +
                         (position[0] - obs_pose.x)**2)

    if args.video_path is not None:
        axs, writer = set_up_video_gen(args.video_path)

    world, builder, inflation_rad, path = get_environment(args)
    plogs = []

    start_pose = path[0]
    start_time = time.time()
    with builder(args.unity_exe_path, sim_scale=0.15) as unity_bridge:
        if args.do_use_robot:
            if args.environment == 'outdoor':
                robots = [
                    vertexnav.robot.Turtlebot_Robot(start_pose,
                                                    primitive_length=2.0 * 1.8,
                                                    unity_bridge=unity_bridge)
                    for _ in range(args.num_robots)
                ]
            else:
                robots = [
                    vertexnav.robot.Turtlebot_Robot(start_pose,
                                                    primitive_length=1.8,
                                                    unity_bridge=unity_bridge)
                    for _ in range(args.num_robots)
                ]
            goals = [
                world.get_random_pose(min_signed_dist=inflation_rad)
                for _ in range(100)
            ]
            goal = max(goals,
                       key=lambda g: (g.x - start_pose.x)**2 +
                       (g.y - start_pose.y)**2)
        else:
            robots = None

        if args.do_explore:
            goal = vertexnav.Pose(x=10000, y=10000)

        # Set up environment and data loop
        unity_bridge.make_world(world)
        data_iterator = get_data_iterator(unity_bridge,
                                          world,
                                          path,
                                          robots=robots,
                                          do_return_time=True)

        # nvg = vertexnav.vertex_graph.NoisyVertexGraph()
        nvg = vertexnav.prob_vertex_graph.ProbVertexGraph()
        nvg.DO_SLAM = True
        nvg.DO_SAMPLE_TOPOLOGY = args.do_merge_vertices
        nvg.DO_MULTI_VERTEX_MERGE = args.do_multi_vertex_merge
        perfect_nvg = vertexnav.vertex_graph.PerfectVertexGraph()

        counter = 0
        t_counter = 0
        net_time = 0

        for poses, odoms, pano_images, get_data_time in data_iterator:
            t_counter += 1
            counter += len(poses)

            print(("=" * 80))
            print(("Pose: {} | Counter: {}".format(poses[0], t_counter)))
            print(("dtime(image data): {}".format(get_data_time)))

            for pose in poses:
                perfect_obs_time = time.time()
                perfect_obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
                    world.get_vertices_for_pose(pose,
                                                max_range=args.max_range),
                    pose,
                    do_add_noise=False,
                    cov_rt=[[2.0**2, 0], [0, 0.15**2]])
                perfect_nvg.add_observation(perfect_obs, pose)
                print(("dtime(perfect obs): {}".format(time.time() -
                                                       perfect_obs_time)))

            if not args.do_use_perfect_obs:
                neural_net_time = time.time()
                net_data = [eval_net(pano_image) for pano_image in pano_images]

                obss = [
                    vertexnav.noisy.convert_net_grid_data_to_noisy_detection(
                        nd,
                        pose,
                        max_range=args.max_range,
                        num_range=args.num_range,
                        num_bearing=args.num_bearing,
                        sig_r=args.sig_r,
                        sig_th=args.sig_th,
                        nn_peak_thresh=args.nn_peak_thresh)
                    for pose, nd in zip(poses, net_data)
                ]

                print(("dtime(neural net time): {}".format(time.time() -
                                                           neural_net_time)))

                add_obs_time = time.time()
                for robot_id, (obs, odom,
                               pose) in enumerate(zip(obss, odoms, poses)):
                    if odom is None:
                        pose.robot_id = robot_id
                        nvg.add_observation(obs, r_pose=pose)
                    else:
                        odom.robot_id = robot_id
                        odom = vertexnav.Pose(x=odom.x,
                                              y=odom.y,
                                              yaw=odom.yaw + 0.1 *
                                              (random.random() - 0.5),
                                              robot_id=odom.robot_id)
                        # yaw=odom.yaw + 0.01)
                        print(("odom", odom))
                        nvg.add_observation(obs, odom=odom)

                print(("dtime(add observation): {}".format(time.time() -
                                                           add_obs_time)))

                if t_counter % 5 == 0:
                    stime = time.time()
                    nvg.sample_states(
                        vertex_association_time_window=15 * args.num_robots,
                        vertex_sampling_time_window=15 * args.num_robots,
                        num_topology_samples=20,
                        num_vertex_samples=50,
                        vertex_association_dist_threshold=20,
                        do_update_state=True)
                    dtime = time.time() - stime
                    net_time += dtime
                    print(("Sampling: ", dtime, net_time))
            else:
                net_data = None
                obs = perfect_obs
                nvg = perfect_nvg

            # Planning
            proposed_world_time = time.time()
            # FIXME(gjstein): get_proposed_world should take robot.all_poses
            proposed_world = nvg.get_proposed_world()
            print(("dtime(proposed world): {}".format(time.time() -
                                                      proposed_world_time)))

            vis_graph_time = time.time()
            visibility_graph = vertexnav.planning.VisibilityGraph(
                proposed_world, inflation_rad=0.8 * inflation_rad)
            print(("dtime(compute vis graph): {}".format(time.time() -
                                                         vis_graph_time)))

            known_poly_time = time.time()
            known_space_poly = nvg.get_known_poly(proposed_world,
                                                  args.max_range)
            print(("dtime(compute known poly): {}".format(time.time() -
                                                          known_poly_time)))

            perfect_known_space_poly = perfect_nvg.get_known_poly(
                perfect_nvg.get_proposed_world(), args.max_range)

            coverage_perfect_iou = world.compute_iou(perfect_known_space_poly)
            print(("Coverage (Perfect): {}".format(coverage_perfect_iou)))
            coverage_noisy_iou = world.compute_iou(known_space_poly)
            print(("Coverage (Noisy): {}".format(coverage_noisy_iou)))

            compute_path_time = time.time()

            uninflated_ksp = known_space_poly
            if args.environment == 'outdoor':
                known_space_poly = known_space_poly.buffer(inflation_rad / 8)
            if args.do_use_frontiers:
                frontiers = vertexnav.planning.compute_frontiers_from_poly(
                    uninflated_ksp, proposed_world, inflation_rad)
                frontier_paths = vertexnav.planning.multiagent_select_frontiers_greedy(
                    robots,
                    frontiers,
                    visibility_graph,
                    do_explore=args.do_explore,
                    known_space_poly=known_space_poly,
                    goal=goal,
                    nearby_clutter_fn=lambda rob: world.get_nearby_clutter(
                        rob.pose, 3 * inflation_rad),
                    cl_inflation_rad=1.4 * inflation_rad)
            else:
                nearby_clutter = world.get_nearby_clutter(
                    robots[0].pose, 3 * inflation_rad)
                if args.do_explore:
                    path, cost = visibility_graph.get_shortest_path(
                        start_point=(robots[0].pose.x, robots[0].pose.y),
                        known_space_poly=known_space_poly,
                        do_return_cost=True,
                        nearby_clutter=nearby_clutter,
                        cl_inflation_rad=1.4 * inflation_rad)
                else:
                    path, cost = visibility_graph.get_shortest_path(
                        start_point=(robots[0].pose.x, robots[0].pose.y),
                        end_point=(goal.x, goal.y),
                        do_return_cost=True,
                        nearby_clutter=nearby_clutter,
                        cl_inflation_rad=inflation_rad)

                frontier_paths = [(None, path, robots[0])]

            known_space_poly = uninflated_ksp

            print(
                ("dtime(compute compute path): {}".format(time.time() -
                                                          compute_path_time)))

            # Plotting
            plotting_time = time.time()
            plot_data = {
                'known_space_poly': known_space_poly,
                'world': world,
                'frontier_paths': frontier_paths,
            }
            plogs.append({
                'world_boundary':
                world.boundary,
                'proposed_world_walls':
                proposed_world.walls,
                'proposed_world_verts':
                proposed_world.vertices,
                'inactive_verts': [
                    v.position for v in nvg.vertices
                    if not v.is_active and v.num_detections > 2
                ],
                'known_poly':
                known_space_poly,
                # 'clutter_data': world.clutter_element_data,
                'poses': [(p.x, p.y) for p in robots[0].all_poses],
                'perfect_poses': [(p.x, p.y) for p in perfect_nvg.r_poses]
            })

            if counter > args.num_robots * args.max_steps:
                break

            if args.video_path is not None:
                try:
                    data = sorted(zip(pano_images, robots),
                                  key=lambda d: id(d[1]))
                    s_pano_images = [environments.utils.convert.image_aligned_from_robot_to_global(im, ro.pose)
                                     for im, ro in data]

                    write_video_frame(axs,
                                      writer,
                                      t_counter,
                                      poses[0],
                                      s_pano_images,
                                      net_data,
                                      obs=obs,
                                      nvg=nvg,
                                      perfect_nvg=perfect_nvg,
                                      plot_data=plot_data,
                                      args=args)
                except:  # noqa
                    raise ValueError

            print(("dtime(compute plotting): {}".format(time.time() -
                                                        plotting_time)))

            # Store logs
            rp = robots[0].pose
            logs["r_poses"].append(
                vertexnav.Pose(rp.x, rp.y, rp.yaw, rp.robot_id))
            logs["perfect_coverage"].append(coverage_perfect_iou)
            logs["noisy_coverage"].append(coverage_noisy_iou)

            # Robot motion
            if args.do_use_robot:
                robot_motion_time = time.time()

                try:
                    for f, path, robot in frontier_paths:
                        robot.move(goal, path, inflation_rad=inflation_rad)

                except ValueError:
                    pass

                print(("dtime(robot motion): {}".format(time.time() -
                                                        robot_motion_time)))

                goal_dist = min([
                    math.sqrt((robot.pose.x - goal.x)**2 +
                              (robot.pose.y - goal.y)**2) for robot in robots
                ])
                all_still = all([r.still_count > 10 for r in robots])
                if all_still or goal_dist < 2:
                    logs["net_time"].append(time.time() - start_time)
                    print(("Total time: {}".format(time.time() - start_time)))
                    print("Finishing up.")
                    break

            logs["net_time"].append(time.time() - start_time)
            print(("Total time: {}".format(time.time() - start_time)))

        # Compute and print the final map likelihood
        nvg.perform_slam_update()
        pose_obs_pairs = list(zip(nvg.r_poses, nvg.observations))
        final_proposed_world = nvg.get_proposed_world_fast(
            topology=nvg.topology)
        final_proposed_world_log_prob = nvg.compute_world_log_prob(
            final_proposed_world, pose_obs_pairs)
        print(
            ("Final Log Prob: {:8.3f}".format(final_proposed_world_log_prob)))

        # Finish up the video
        writer.finish()

        # Generate a plot
        axs = set_up_video_gen(video_path=None, figure_only=True)
        write_video_frame(axs,
                          None,
                          t_counter,
                          poses[0],
                          s_pano_images,
                          net_data,
                          obs=obs,
                          nvg=nvg,
                          perfect_nvg=perfect_nvg,
                          plot_data=plot_data,
                          args=args,
                          figure_only=True)
        plt.savefig(args.figure_path, dpi=300)

        with open(log_text_file_name, "a+") as log_file:
            if args.do_use_perfect_obs:
                s_planner = "perfect"
            else:
                s_planner = "  noisy"

            if args.do_explore:
                s_planner += " exp"
            else:
                s_planner += " nav"

            if args.do_merge_vertices:
                if args.do_multi_vertex_merge:
                    s_planner += " Mvmp"
                else:
                    s_planner += " Svmp"
            else:
                s_planner += " NOmp"

            log_file.write(
                "{:10d}  {}  {:8d}  {:12.6f}  {:8.6f}  {:8.6f} {} {}\n".format(
                    args.seed,
                    s_planner,
                    len(logs["net_time"]),
                    max(logs["net_time"]),
                    max(logs["perfect_coverage"][-10:]),
                    max(logs["noisy_coverage"][-10:]),
                    args.environment,
                    args.num_robots,
                ))


def parse_args():
    """Define the command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate video showing neural net results ' +
        'in simulated hallway environment.')
    parser.add_argument('--environment', type=str)
    parser.add_argument('--unity_exe_path', type=str)
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--figure_path', type=str, default=None)
    parser.add_argument('--save_frames', type=int, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--do_use_perfect_obs', action='store_true')
    parser.add_argument('--do_explore', action='store_true')
    parser.add_argument('--do_use_frontiers', action='store_true')
    parser.add_argument('--num_robots',
                        type=int,
                        default=1,
                        help='Number of robots.')
    parser.add_argument('--grid_resolution', type=float, default=1.0)
    parser.add_argument('--merge_type', type=str, required=True)

    # Grid variables
    parser.add_argument('--max_range',
                        type=int,
                        default=120,
                        help='Max range for range in output grid.')
    parser.add_argument('--num_range',
                        type=int,
                        default=32,
                        help='Number of range cells in output grid.')
    parser.add_argument('--num_bearing',
                        type=int,
                        default=128,
                        help='Number of bearing cells in output grid.')

    parser.add_argument('--sig_r', type=float, default=10.0)
    parser.add_argument('--sig_th', type=float, default=0.25)
    parser.add_argument('--nn_peak_thresh', type=float, required=True)

    parser.add_argument('--xbounds',
                        nargs='+',
                        type=int,
                        default=[-50, 220],
                        help='Boundary in "x".')
    parser.add_argument('--ybounds',
                        nargs='+',
                        type=int,
                        default=[-100, 250],
                        help='Boundary in "y".')
    parser.add_argument('--max_steps', type=int, default=400)

    # Robot variables
    parser.add_argument('--do_use_robot', action='store_true')

    args = parser.parse_args()

    if not args.do_use_perfect_obs and args.network_file is None:
        raise ValueError(
            "Network file required when perfect detections not used.")

    if 'none' not in args.merge_type:
        args.do_merge_vertices = True
    else:
        args.do_merge_vertices = False
    if 'multi' in args.merge_type:
        args.do_multi_vertex_merge = True
    else:
        args.do_multi_vertex_merge = False

    if args.do_use_robot:
        if args.num_robots < 1:
            raise ValueError("There must be at least one robot.")
        if not args.num_robots == 1 and not args.do_use_frontiers:
            raise ValueError("Frontiers must be used for multiple robots.")

    return args


if __name__ == "__main__":
    args = parse_args()
    args.current_seed = args.seed
    travel_and_make_plot(args)
