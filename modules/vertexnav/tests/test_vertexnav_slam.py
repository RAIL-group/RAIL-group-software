import numpy as np
import environments
import vertexnav
import math
import pytest
import os.path
import random
from shapely import geometry
import time

# Obvously, replace this with your own Unity project.
UNITY_EXE = '/Users/gjstein/rrg_data/unity/unity_tcp_test.app/Contents/MacOS/unity_tcp_test'


def get_map_and_path_hall_snake():
    maze_poly = geometry.Polygon([(10, -10), (10, 20), (50, 20), (50, 50),
                                  (70, 50), (70, 00), (130, 00), (130, 20),
                                  (90, 20), (90, 70), (30, 70), (30, 40),
                                  (-10, 40), (-10, -10)])
    path = [(40, 30), (40, 60), (80, 60), (80, 10), (120, 10)]
    return vertexnav.world.World(obstacles=[maze_poly]), path


def follow_path_data_iterator(unity_bridge, world, path, steps=50, pause=None):
    """Loop through data along a path."""
    stime = time.time()
    unity_bridge.make_world(world)
    print(f"Time to Make World: {time.time() - stime}")
    pose_generator = (vertexnav.Pose(
        ii * 1.0 * seg[1][0] / steps + (1 - ii * 1.0 / steps) * seg[0][0],
        ii * 1.0 * seg[1][1] / steps + (1 - ii * 1.0 / steps) * seg[0][1])
                      for seg in zip(path[:-1], path[1:])  # noqa: E126
                      for ii in range(steps))

    for pose in pose_generator:
        # Get the images
        if pause is not None:
            unity_bridge.move_object_to_pose("robot", pose, pause)
            pano_image = unity_bridge.get_image("robot/pano_camera", pause)
            pano_depth_image = unity_bridge.get_image(
                "robot/pano_depth_camera", pause)
        else:
            pano_image = unity_bridge.get_image("robot/pano_camera")
            pano_depth_image = unity_bridge.get_image(
                "robot/pano_depth_camera")
            unity_bridge.move_object_to_pose("robot", pose)

        pano_image = pano_image[64:-64] * 1.0 / 255
        ranges = vertexnav.utils.convert.ranges_from_depth_image(
            pano_depth_image)

        yield pose, pano_image, ranges


def _add_new_observation(pvg,
                         world,
                         pose,
                         odom,
                         err=None,
                         association_window=-1):
    pose = vertexnav.Pose(pose.x, pose.y, pose.yaw, pose.robot_id)
    obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
        world.get_vertices_for_pose(pose),
        pose,
        do_add_noise=False,
        cov_rt=[[0.1**2, 0], [0, 0.1**2]])
    assert len(obs) < 4

    if odom is None:
        pvg.add_observation(obs,
                            r_pose=pose,
                            association_window=association_window)
    else:
        odom = vertexnav.Pose(odom.x, odom.y, odom.yaw, odom.robot_id)
        if err is not None:
            odom.x += err[0]
            odom.y += err[1]
            odom.yaw += err[2]
        # We need to update the position of the detections to make this a
        # fair test.
        nposes = len(pvg.r_poses)
        num_robot_ids = len(set(p.robot_id for p in pvg.r_poses))
        updated_pose = odom * pvg.r_poses[nposes - num_robot_ids]
        for det in obs:
            det.update_props(updated_pose)

        pvg.add_observation(obs,
                            odom=odom,
                            association_window=association_window)


def _add_new_observation_learned(pvg,
                                 image,
                                 eval_net,
                                 args,
                                 pose,
                                 odom,
                                 err=None):
    pose = vertexnav.Pose(pose.x, pose.y, pose.yaw, pose.robot_id)
    # Get the observation
    obs = vertexnav.noisy.convert_net_grid_data_to_noisy_detection(
        eval_net(image),
        pose,
        max_range=args.max_range,
        num_range=args.num_range,
        num_bearing=args.num_bearing,
        sig_r=args.sig_r,
        sig_th=args.sig_th,
        nn_peak_thresh=args.nn_peak_thresh)

    assert len(obs) > 0

    if odom is None:
        pvg.add_observation(obs, r_pose=pose)
    else:
        odom = vertexnav.Pose(odom.x, odom.y, odom.yaw, odom.robot_id)
        if err is not None:
            odom.x += err[0]
            odom.y += err[1]
            odom.yaw += err[2]
        # We need to update the position of the detections to make this a
        # fair test.
        nposes = len(pvg.r_poses)
        updated_pose = odom * pvg.r_poses[nposes - 1]
        for det in obs:
            det.update_props(updated_pose)

        pvg.add_observation(obs, odom=odom)


def _plot_pvg_data(ax, pvg, world, title):
    ax.clear()
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    pose = pvg.r_poses[len(pvg.r_poses) - 1]
    proposed_world = pvg.get_proposed_world()

    # Plot the clusters
    vertex_remapping = vertexnav.prob_vertex_graph._get_vertex_remapping(
        vertices=pvg.vertices, topology=pvg.topology)
    vertex_id_dict = {v.id: v for v in pvg.vertices}
    clusters = [
        vertexnav.prob_vertex_graph.Cluster(c, vertex_id_dict)
        for c in pvg.topology
    ]
    for cluster in clusters:
        if not cluster.is_active and cluster.num_dets > 2:
            vert = vertex_remapping[cluster.vids[0]]
            ax.plot(vert.position[0],
                    vert.position[1],
                    'r.',
                    alpha=0.4,
                    markersize=vertexnav.plotting.SCATTERSIZE * 2)

    vertexnav.plotting.plot_world(ax, world, alpha=0.2)

    robot_ids = set(p.robot_id for p in pvg.r_poses)
    for rid in robot_ids:
        r_points = np.array([(p.x, p.y) for p in pvg.r_poses
                             if p.robot_id == rid])
        line, = ax.plot(r_points[:, 0], r_points[:, 1], alpha=0.75)
        ax.plot(r_points[-1, 0], r_points[-1, 1], '.', color=line.get_color())
        ax.plot(r_points[-1, 0], r_points[-1, 1], 'x', color=line.get_color())

    vertexnav.plotting.plot_proposed_world(ax,
                                           proposed_world,
                                           do_show_points=True,
                                           do_plot_visibility=False,
                                           robot_pose=pose)


def test_slam_orbiting_square(do_debug_plot):
    random.seed(304)

    # A square world
    square_poly = geometry.Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    world = vertexnav.world.World(obstacles=[square_poly])

    pvg_perfect = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_perfect.DO_SLAM = False
    pvg_error = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_error.DO_SLAM = False
    pvg_slam = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_slam.DO_SLAM = True

    # Set the noise models
    pvg_slam.PRIOR_NOISE = np.array([0.3, 0.3, 0.1])
    pvg_slam.ODOMETRY_NOISE = np.array([0.02, 0.02, 0.5])
    pvg_slam.CLUSTERING_NOISE = np.array([0.005, 0.005])

    # Add a number of observations
    n_poses_per_loop = 80
    n_loops = 4
    for ii in range(n_loops * n_poses_per_loop + 1):
        th = 2 * math.pi * ii / n_poses_per_loop + 0.0001
        pose = vertexnav.Pose(x=2.5 * math.cos(th),
                              y=2.5 * math.sin(th),
                              yaw=th)

        if ii == 0:
            odom = None
        else:
            nposes = len(pvg_perfect.r_poses)
            odom = vertexnav.Pose.get_odom(p_new=pose,
                                           p_old=pvg_perfect.r_poses[nposes -
                                                                     1])

        # A very simple error model
        err = [
            0.01 * (random.random() - 0.5), 0.01 * (random.random() - 0.5),
            0.005 * random.random()
        ]

        _add_new_observation(pvg=pvg_perfect,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=None)

        _add_new_observation(pvg=pvg_error,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err)

        print("PVG_SLAM")
        _add_new_observation(pvg=pvg_slam,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err)

        p = pvg_error.r_poses[len(pvg_error.r_poses) - 1]
        print((p.x, p.y, p.yaw, p.index))
        p = pvg_slam.r_poses[len(pvg_slam.r_poses) - 1]
        print((p.x, p.y, p.yaw, p.index))

    if do_debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()

        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)

        _plot_pvg_data(ax1, pvg_perfect, world, "Perfect Odometry")
        _plot_pvg_data(ax2, pvg_error, world, "Noisy Odom")
        _plot_pvg_data(ax3, pvg_slam, world, "Noisy Odom + SLAM")

        plt.show()

    assert not (pvg_perfect.r_poses[len(pvg_perfect.r_poses) - 1]
                == pvg_error.r_poses[len(pvg_error.r_poses) - 1])

    assert len(pvg_perfect.vertices) == 4
    assert len(pvg_error.vertices) > 4
    assert len(pvg_slam.vertices) == 4

    _, state_perfect = pvg_perfect.sample_vertices(do_update_state=False)
    _, state_error = pvg_error.sample_vertices(do_update_state=False)
    _, state_slam = pvg_slam.sample_vertices(do_update_state=False)

    assert state_perfect.log_prob > state_error.log_prob
    assert state_slam.log_prob > state_error.log_prob

    # Confirm that the 'perfect' world is as expected
    proposed_world = pvg_perfect.get_proposed_world()
    assert len(proposed_world.vertices) == 4
    for v in proposed_world.vertices:
        assert abs(v[0]) == pytest.approx(1.0, abs=1e-2)
        assert abs(v[1]) == pytest.approx(1.0, abs=1e-2)
    assert len(proposed_world.walls) == 4

    # Confirm the 'SLAM' world should look quite similar.
    proposed_world = pvg_slam.get_proposed_world()
    assert len(proposed_world.vertices) == 4
    for v in proposed_world.vertices:
        assert abs(v[0]) == pytest.approx(1.0, abs=5e-2)
        assert abs(v[1]) == pytest.approx(1.0, abs=5e-2)
    assert len(proposed_world.walls) == 4


def test_slam_pose_robot_id():
    pose = vertexnav.Pose(0, 0, 0, 1)
    assert (pose.robot_id == 1)

    pose = vertexnav.Pose(0, 0, 0, 2)
    assert (pose.robot_id == 2)


def test_slam_orbiting_square_multi_agent(do_debug_plot):
    random.seed(304)

    # A square world
    square_poly = geometry.Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    world = vertexnav.world.World(obstacles=[square_poly])

    pvg_perfect = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_perfect.DO_SLAM = False
    pvg_error = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_error.DO_SLAM = False
    pvg_slam = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_slam.DO_SLAM = True

    # Set the noise models
    pvg_slam.PRIOR_NOISE = np.array([0.3, 0.3, 0.1])
    pvg_slam.ODOMETRY_NOISE = np.array([0.02, 0.02, 0.5])
    pvg_slam.CLUSTERING_NOISE = np.array([0.005, 0.005])

    # Add a number of observations
    n_poses_per_loop = 80
    n_loops = 4
    for ii in range(n_loops * n_poses_per_loop + 1):

        # Robot 1
        th = 2 * math.pi * ii / n_poses_per_loop + 0.0001
        pose = vertexnav.Pose(x=2.5 * math.cos(th),
                              y=2.5 * math.sin(th),
                              yaw=th,
                              robot_id=1)
        pose1 = pose

        if ii == 0:
            odom = None
        else:
            nposes = len(pvg_perfect.r_poses)
            odom = vertexnav.Pose.get_odom(p_new=pose1,
                                           p_old=pvg_perfect.r_poses[nposes -
                                                                     2])

        # A very simple error model
        if ii == 0:
            err = [0, 0, 0]
        else:
            err = [
                0.01 * (random.random() - 0.5), 0.01 * (random.random() - 0.5),
                0.005 * random.random()
            ]

        _add_new_observation(pvg=pvg_perfect,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=None)

        _add_new_observation(pvg=pvg_error,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err)

        _add_new_observation(pvg=pvg_slam,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err)

        # Robot 2
        th = 2 * math.pi * ii / n_poses_per_loop + 0.0001 + 2 * np.pi / 3
        pose = vertexnav.Pose(x=3.5 * math.cos(th),
                              y=3.5 * math.sin(th),
                              yaw=th,
                              robot_id=2)
        pose2 = pose

        if ii == 0:
            odom = None
        else:
            nposes = len(pvg_perfect.r_poses)
            odom = vertexnav.Pose.get_odom(p_new=pose2,
                                           p_old=pvg_perfect.r_poses[nposes -
                                                                     2])

        # A very simple error model
        if ii == 0:
            err = [0, 0, 0]
        else:
            err = [
                0.01 * (random.random() - 0.5), 0.01 * (random.random() - 0.5),
                0.005 * random.random()
            ]

        _add_new_observation(pvg=pvg_perfect,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=None)

        _add_new_observation(pvg=pvg_error,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err)

        _add_new_observation(pvg=pvg_slam,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err)

    if do_debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()

        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)

        _plot_pvg_data(ax1, pvg_perfect, world, "Perfect Odometry")
        _plot_pvg_data(ax2, pvg_error, world, "Noisy Odom")
        _plot_pvg_data(ax3, pvg_slam, world, "Noisy Odom + SLAM")

        plt.show()

    assert abs(pvg_perfect.r_poses[len(pvg_perfect.r_poses) - 1].x -
               pvg_perfect.r_poses[len(pvg_perfect.r_poses) - 2].x) > 2.0
    assert abs(pvg_perfect.r_poses[len(pvg_perfect.r_poses) - 1].y -
               pvg_perfect.r_poses[len(pvg_perfect.r_poses) - 2].y) > 2.0

    assert not (pvg_perfect.r_poses[len(pvg_perfect.r_poses) - 1]
                == pvg_error.r_poses[len(pvg_error.r_poses) - 1])

    assert len(pvg_perfect.vertices) == 4
    assert len(pvg_error.vertices) > 4
    assert len(pvg_slam.vertices) == 4

    _, state_perfect = pvg_perfect.sample_vertices(do_update_state=False)
    _, state_error = pvg_error.sample_vertices(do_update_state=False)
    _, state_slam = pvg_slam.sample_vertices(do_update_state=False)

    assert state_perfect.log_prob > state_error.log_prob
    assert state_slam.log_prob > state_error.log_prob

    # Confirm that the 'perfect' world is as expected
    proposed_world = pvg_perfect.get_proposed_world()
    assert len(proposed_world.vertices) == 4
    for v in proposed_world.vertices:
        assert abs(v[0]) == pytest.approx(1.0, abs=1e-2)
        assert abs(v[1]) == pytest.approx(1.0, abs=1e-2)
    assert len(proposed_world.walls) == 4

    # Confirm the 'SLAM' world should look quite similar.
    proposed_world = pvg_slam.get_proposed_world()
    assert len(proposed_world.vertices) == 4
    for v in proposed_world.vertices:
        assert abs(v[0]) == pytest.approx(1.0, abs=5e-2)
        assert abs(v[1]) == pytest.approx(1.0, abs=5e-2)
    assert len(proposed_world.walls) == 4


def test_slam_vertex_disabling(do_debug_plot):
    random.seed(304)

    # A square world
    square_poly = geometry.Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    world = vertexnav.world.World(obstacles=[square_poly])

    pvg_perfect = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_error = vertexnav.prob_vertex_graph.ProbVertexGraph()

    # Add a number of observations
    n_poses_per_loop = 40
    n_loops = 6
    for ii in range(n_loops * n_poses_per_loop + 1):
        th = 2 * math.pi * ii / n_poses_per_loop + 0.0001
        pose = vertexnav.Pose(x=2.5 * math.cos(th),
                              y=2.5 * math.sin(th),
                              yaw=th)

        if ii == 0:
            odom = None
        else:
            nposes = len(pvg_perfect.r_poses)
            odom = vertexnav.Pose.get_odom(p_new=pose,
                                           p_old=pvg_perfect.r_poses[nposes -
                                                                     1])

        _add_new_observation(pvg=pvg_perfect,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=None)

        _add_new_observation(pvg=pvg_error,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=None)

    pvg_perfect.perform_slam_update()
    pvg_error.perform_slam_update()

    # Confirm that the two roughly match
    assert all([
        abs(pp.x - pe.x) < 1e-3 and abs(pp.y - pe.y) < 1e-3
        for pp, pe in zip(pvg_perfect.r_poses, pvg_error.r_poses)
    ])

    # Corrupt observations of one of the vertices
    for obs, r_pose in zip(pvg_error.observations, pvg_error.r_poses):
        for det in obs:
            if det.position[0] > 0.5 and det.position[1] > 0.5:
                det.range += 0.1
                det.update_props(r_pose)

    # Run SLAM on the erroneous system
    pvg_error.perform_slam_update()

    # Confirm that some of the poses have shifted away
    assert not all(
        abs(pp.x - pe.x) < 1e-3 and abs(pp.y - pe.y) < 1e-3
        for pp, pe in zip(pvg_perfect.r_poses, pvg_error.r_poses))
    dist_err = sum(
        math.sqrt((pp.x - pe.x)**2 + (pp.y - pe.y)**2)
        for pp, pe in zip(pvg_perfect.r_poses, pvg_error.r_poses))

    # Disabling the selected vertex should restore the system to its original
    # state. I disable this vertex, rerun SLAM, and test.
    for v in pvg_error.vertices:
        if v.position[0] > 0.5 and v.position[1] > 0.5:
            v.is_active = False

    pvg_error.perform_slam_update()

    dist_disabled = sum(
        math.sqrt((pp.x - pe.x)**2 + (pp.y - pe.y)**2)
        for pp, pe in zip(pvg_perfect.r_poses, pvg_error.r_poses))

    assert dist_disabled < dist_err


def test_slam_vertex_clustering_manual(do_debug_plot):
    random.seed(304)

    # A square world
    square_poly = geometry.Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    world = vertexnav.world.World(obstacles=[square_poly])

    pvg_perfect = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_no_top = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_no_top.DO_SLAM = False
    pvg_with_top = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_with_top.DO_SLAM = False

    # Add a number of observations
    n_poses_per_loop = 40
    n_loops = 0.75
    for ii in range(int(n_loops * n_poses_per_loop) + 1):
        th = 2 * math.pi * ii / n_poses_per_loop + 0.0001
        pose = vertexnav.Pose(x=2.5 * math.cos(th),
                              y=2.5 * math.sin(th),
                              yaw=th)
        err = [0.0, 0.0, 0.005]

        if ii == 0:
            odom = None
        else:
            nposes = len(pvg_perfect.r_poses)
            odom = vertexnav.Pose.get_odom(p_new=pose,
                                           p_old=pvg_perfect.r_poses[nposes -
                                                                     1])

        _add_new_observation(pvg=pvg_perfect,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=None)

        _add_new_observation(pvg=pvg_no_top,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err)

        _add_new_observation(pvg=pvg_with_top,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err)

    # Check some properties
    assert len(pvg_perfect.vertices) == 4
    assert len(pvg_no_top.vertices) == 5
    assert len(pvg_with_top.vertices) == 5

    pvg_no_top.DO_SLAM = True
    pvg_with_top.DO_SLAM = True

    # We should be sure the system is
    # "settled" before changing it.
    pvg_no_top.perform_slam_update()

    # Merge a couple of the vertices
    clustered_verts = []
    topology = []
    for v in pvg_with_top.vertices:
        if v.position[0] > 0.5 and v.position[1] < -0.5:
            clustered_verts += [v.id]
        else:
            topology += [(v.id, )]

    topology += [tuple(clustered_verts)]
    pvg_with_top.topology = topology

    # Rerun the SLAM system and observe the change
    pvg_with_top.perform_slam_update()

    # The poses with the merge should be closer to perfect
    dist_sq_no_top = 0
    dist_sq_with_top = 0
    for pp, pn, pm in zip(pvg_perfect.r_poses, pvg_no_top.r_poses,
                          pvg_with_top.r_poses):
        dist_sq_no_top += (pp.x - pn.x)**2 + (pp.y - pn.y)**2
        dist_sq_with_top += (pp.x - pm.x)**2 + (pp.y - pm.y)**2

    assert dist_sq_with_top < dist_sq_no_top

    if do_debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()

        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)

        _plot_pvg_data(ax1, pvg_perfect, world, "Perfect Odometry")
        _plot_pvg_data(ax2, pvg_no_top, world, "SLAM (no merging)")
        _plot_pvg_data(ax3, pvg_with_top, world, "SLAM (with merging)")

        plt.show()

    # Compare the likelihoods of each map
    _, state_perfect = pvg_perfect.sample_vertices(do_update_state=False)
    _, state_no_top = pvg_no_top.sample_vertices(do_update_state=False)
    _, state_with_top = pvg_with_top.sample_vertices(do_update_state=False)

    print((pvg_perfect.topology))
    print((pvg_with_top.topology))
    print((pvg_no_top.topology))
    assert state_perfect.log_prob > state_no_top.log_prob
    assert state_with_top.log_prob > state_no_top.log_prob

    # Now merge all the vertices into a single cluster
    # (to see if anything breaks)
    topology = [tuple([v.id for v in pvg_with_top.vertices])]
    pvg_with_top.topology = topology
    pvg_with_top.perform_slam_update()


@pytest.mark.skip("Test too succeptable to numerical noise.")
def test_slam_vertex_clustering_full(do_debug_plot):
    random.seed(304)

    # A square world
    square_poly = geometry.Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    world = vertexnav.world.World(obstacles=[square_poly])

    pvg_perfect = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_no_top = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_with_top = vertexnav.prob_vertex_graph.ProbVertexGraph()

    # Add a number of observations
    n_poses_per_loop = 40
    n_loops = 2
    for ii in range(int(n_loops * n_poses_per_loop) + 1):
        th = 2 * math.pi * ii / n_poses_per_loop + 0.0001
        pose = vertexnav.Pose(x=2.5 * math.cos(th),
                              y=2.5 * math.sin(th),
                              yaw=th)

        if ii == 0:
            odom = None
        else:
            nposes = len(pvg_perfect.r_poses)
            odom = vertexnav.Pose.get_odom(p_new=pose,
                                           p_old=pvg_perfect.r_poses[nposes -
                                                                     1])

        _add_new_observation(pvg=pvg_perfect,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=None)

        _add_new_observation(pvg=pvg_no_top,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=[0.0, 0.0, 0.01],
                             association_window=5)

        _add_new_observation(pvg=pvg_with_top,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=[0.0, 0.0, 0.01],
                             association_window=5)

        if ii % 5 == 0:
            # Sample only vertices for the no-top one
            pvg_no_top.sample_vertices(p_window=15,
                                       num_samples=50,
                                       do_update_state=True)

            # Sample topologies for the other
            pvg_with_top.sample_states(num_topology_samples=20,
                                       num_vertex_samples=50,
                                       vertex_association_dist_threshold=2,
                                       vertex_association_time_window=15,
                                       vertex_sampling_time_window=15,
                                       do_update_state=True)

    # Add one final sample for each
    pvg_no_top.sample_vertices(p_window=15,
                               num_samples=50,
                               do_update_state=True)
    pvg_with_top.sample_states(num_topology_samples=40,
                               num_vertex_samples=50,
                               vertex_association_dist_threshold=2,
                               vertex_association_time_window=15,
                               vertex_sampling_time_window=15,
                               do_update_state=True)

    # Test some basic properties
    print((pvg_perfect.vertices))
    print(pvg_with_top.vertices)
    print(pvg_no_top.vertices)
    print(pvg_with_top.topology)
    print(pvg_no_top.topology)
    assert len(pvg_perfect.vertices) == 4
    assert len(pvg_with_top.vertices) > 4
    assert len(pvg_no_top.vertices) > 4

    print((pvg_with_top.vertices))
    print((pvg_with_top.topology))
    assert len(pvg_with_top.topology) < 6
    assert len(pvg_no_top.topology) > 4

    if do_debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()

        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)

        _plot_pvg_data(ax1, pvg_perfect, world, "Perfect Odometry")
        _plot_pvg_data(ax2, pvg_no_top, world, "SLAM (no merging)")
        _plot_pvg_data(ax3, pvg_with_top, world, "SLAM (with merging)")

        plt.show()

    # Compare the likelihoods of each map
    _, state_perfect = pvg_perfect.sample_vertices(do_update_state=False)
    _, state_no_top = pvg_no_top.sample_vertices(do_update_state=False)
    _, state_with_top = pvg_with_top.sample_vertices(do_update_state=False)

    assert state_perfect.log_prob > state_no_top.log_prob
    assert state_with_top.log_prob > state_no_top.log_prob


@pytest.mark.skip("Test too succeptable to numerical noise.")
def test_slam_orbiting_square_multi_agent_windowed(do_debug_plot):
    random.seed(304)

    # Parameters
    bg_association = 10000
    sm_association = 5

    # A square world
    square_poly = geometry.Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    world = vertexnav.world.World(obstacles=[square_poly])

    pvg_perfect = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_perfect.DO_SLAM = False
    pvg_bg_window = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_bg_window.DO_SLAM = True
    pvg_bg_window_top = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_bg_window_top.DO_SLAM = True
    pvg_sm_window = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_sm_window.DO_SLAM = True
    pvg_sm_window_top = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_sm_window_top.DO_SLAM = True

    # Set the noise models
    pvg_bg_window.PRIOR_NOISE = np.array([0.3, 0.3, 0.1])
    pvg_bg_window.ODOMETRY_NOISE = np.array([0.02, 0.02, 0.5])
    pvg_bg_window.CLUSTERING_NOISE = np.array([0.005, 0.005])
    pvg_bg_window_top.PRIOR_NOISE = np.array([0.3, 0.3, 0.1])
    pvg_bg_window_top.ODOMETRY_NOISE = np.array([0.02, 0.02, 0.5])
    pvg_bg_window_top.CLUSTERING_NOISE = np.array([0.005, 0.005])
    pvg_sm_window.PRIOR_NOISE = np.array([0.3, 0.3, 0.1])
    pvg_sm_window.ODOMETRY_NOISE = np.array([0.02, 0.02, 0.5])
    pvg_sm_window.CLUSTERING_NOISE = np.array([0.005, 0.005])
    pvg_sm_window_top.PRIOR_NOISE = np.array([0.3, 0.3, 0.1])
    pvg_sm_window_top.ODOMETRY_NOISE = np.array([0.02, 0.02, 0.5])
    pvg_sm_window_top.CLUSTERING_NOISE = np.array([0.005, 0.005])

    # Add a number of observations
    n_poses_per_loop = 80
    n_loops = 2
    for ii in range(n_loops * n_poses_per_loop + 1):

        # Robot 1
        th = 2 * math.pi * ii / n_poses_per_loop + 0.0001
        pose = vertexnav.Pose(x=2.5 * math.cos(th),
                              y=2.5 * math.sin(th),
                              yaw=th,
                              robot_id=1)
        pose1 = pose

        if ii == 0:
            odom = None
        else:
            nposes = len(pvg_perfect.r_poses)
            odom = vertexnav.Pose.get_odom(p_new=pose1,
                                           p_old=pvg_perfect.r_poses[nposes -
                                                                     2])

        # A very simple error model
        if ii == 0:
            err = [0, 0, 0]
        else:
            err = [
                0.01 * (random.random() - 0.5), 0.01 * (random.random() - 0.5),
                0.005 * random.random()
            ]

        _add_new_observation(pvg=pvg_perfect,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=None)
        _add_new_observation(pvg=pvg_bg_window,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err,
                             association_window=bg_association)
        _add_new_observation(pvg=pvg_bg_window_top,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err,
                             association_window=bg_association)
        _add_new_observation(pvg=pvg_sm_window,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err,
                             association_window=sm_association)
        _add_new_observation(pvg=pvg_sm_window_top,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err,
                             association_window=sm_association)

        # Robot 2
        th = math.pi * ii / n_poses_per_loop + 0.0001
        pose = vertexnav.Pose(x=3.5 * math.cos(th),
                              y=3.5 * math.sin(th),
                              yaw=th,
                              robot_id=2)
        pose2 = pose

        if ii == 0:
            odom = None
        else:
            nposes = len(pvg_perfect.r_poses)
            odom = vertexnav.Pose.get_odom(p_new=pose2,
                                           p_old=pvg_perfect.r_poses[nposes -
                                                                     2])

        _add_new_observation(pvg=pvg_perfect,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=None)
        _add_new_observation(pvg=pvg_bg_window,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err,
                             association_window=bg_association)
        _add_new_observation(pvg=pvg_bg_window_top,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err,
                             association_window=bg_association)
        _add_new_observation(pvg=pvg_sm_window,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err,
                             association_window=sm_association)
        _add_new_observation(pvg=pvg_sm_window_top,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err,
                             association_window=sm_association)

        # A very simple error model
        if ii == 0:
            err = [0, 0, 0]
        else:
            err = [
                0.01 * (random.random() - 0.5), 0.01 * (random.random() - 0.5),
                0.005 * random.random()
            ]

        if ii % 5 == 0:
            # Sample topologies
            pvg_bg_window_top.sample_states(
                num_topology_samples=20,
                num_vertex_samples=50,
                vertex_association_dist_threshold=2,
                vertex_association_time_window=15,
                vertex_sampling_time_window=15,
                do_update_state=True)
            pvg_sm_window_top.sample_states(
                num_topology_samples=20,
                num_vertex_samples=50,
                vertex_association_dist_threshold=2,
                vertex_association_time_window=15,
                vertex_sampling_time_window=15,
                do_update_state=True)

    if do_debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()

        ax1 = plt.subplot(3, 2, 1)
        ax2 = plt.subplot(3, 2, 3)
        ax3 = plt.subplot(3, 2, 4)
        ax4 = plt.subplot(3, 2, 5)
        ax5 = plt.subplot(3, 2, 6)

        _plot_pvg_data(ax1, pvg_perfect, world, "Perfect Odometry")
        _plot_pvg_data(ax2, pvg_bg_window, world, "Bg")
        _plot_pvg_data(ax3, pvg_bg_window_top, world, "Bg + Top")
        _plot_pvg_data(ax4, pvg_sm_window, world, "Sm")
        _plot_pvg_data(ax5, pvg_sm_window_top, world, "Sm + Top")

        plt.show()

    # Sample one more time
    pvg_bg_window_top.sample_states(num_topology_samples=100,
                                    num_vertex_samples=50,
                                    vertex_association_dist_threshold=2,
                                    vertex_association_time_window=15,
                                    vertex_sampling_time_window=15,
                                    do_update_state=True)
    pvg_sm_window_top.sample_states(num_topology_samples=100,
                                    num_vertex_samples=50,
                                    vertex_association_dist_threshold=2,
                                    vertex_association_time_window=15,
                                    vertex_sampling_time_window=15,
                                    do_update_state=True)

    assert len(pvg_perfect.vertices) == 4
    assert len(pvg_bg_window.topology) == 8
    assert len(pvg_bg_window_top.topology) < 8  # ==4, but sensitive to noise
    assert len(pvg_sm_window.topology) > 10
    assert len(pvg_sm_window_top.topology) < 8  # ==4, but sensitive to noise


@pytest.mark.skip(reason="requires CNN that does not exist yet.")
def test_slam_dungeon_rails_nav_unity(do_debug_plot, unity_path,
                                      sim_dungeon_network_path):
    random.seed(304)

    if unity_path is None:
        pytest.xfail("Missing Unity dungeon environment path. "
                     "Set via '--unity-dungeon-path'.")

    if sim_dungeon_network_path is None:
        pytest.xfail("Missing network .proto for sim hallway. "
                     "Set via '--sim-dungeon-network-path'.")

    # Set the parameters for learning + processing
    args = lambda: None  # noqa: E731
    args.max_range = 100
    args.num_range = 32
    args.num_bearing = 128
    args.sig_r = 5.0
    args.sig_th = 0.25
    args.nn_peak_thresh = 0.5

    world, path = get_map_and_path_hall_snake()
    world.breadcrumb_element_poses = []
    pvg_perfect = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_error = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_error.DO_SLAM = False
    pvg_slam = vertexnav.prob_vertex_graph.ProbVertexGraph()

    world_building_unity_bridge = \
        environments.simulated.WorldBuildingUnityBridge
    with world_building_unity_bridge(unity_path) as unity_bridge:

        data_iterator = follow_path_data_iterator(unity_bridge,
                                                  world,
                                                  path,
                                                  steps=20)
        unity_bridge.make_world(world)

        if not os.path.exists(sim_dungeon_network_path):
            pytest.xfail(
                f"Network file {sim_dungeon_network_path} does not exist.")

        eval_net = vertexnav.learning.load_pytorch_net_vertex(
            sim_dungeon_network_path, 'cuda')

        for pose, image, ranges in data_iterator:

            if len(pvg_perfect.r_poses) == 0:
                odom = None
            else:
                nposes = len(pvg_perfect.r_poses)
                odom = vertexnav.Pose.get_odom(
                    p_new=pose, p_old=pvg_perfect.r_poses[nposes - 1])

            # A very simple error model
            err = [
                0.05 * (random.random() - 0.5), 0.05 * (random.random() - 0.5),
                0.125 * (random.random() - 0.5)
            ]

            _add_new_observation_learned(pvg=pvg_perfect,
                                         image=image,
                                         eval_net=eval_net,
                                         args=args,
                                         pose=pose,
                                         odom=odom,
                                         err=None)

            _add_new_observation_learned(pvg=pvg_error,
                                         image=image,
                                         eval_net=eval_net,
                                         args=args,
                                         pose=pose,
                                         odom=odom,
                                         err=err)

            _add_new_observation_learned(pvg=pvg_slam,
                                         image=image,
                                         eval_net=eval_net,
                                         args=args,
                                         pose=pose,
                                         odom=odom,
                                         err=err)
            pvg_slam.perform_slam_update()

    if do_debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()

        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)

        _plot_pvg_data(ax1, pvg_perfect, world, "Perfect Odometry")
        _plot_pvg_data(ax2, pvg_error, world, "Noisy Odom")
        _plot_pvg_data(ax3, pvg_slam, world, "Noisy Odom + SLAM")

        plt.show()

    # The poses with the merge should be closer to perfect
    for pp, pn, pm in zip(pvg_perfect.r_poses, pvg_error.r_poses,
                          pvg_slam.r_poses):
        dist_sq_no_top = (pp.x - pn.x)**2 + (pp.y - pn.y)**2
        dist_sq_with_top = (pp.x - pm.x)**2 + (pp.y - pm.y)**2
        assert dist_sq_with_top - dist_sq_no_top <= 1e-2
