import math
import numpy as np
import pytest
import random
from shapely import geometry
import vertexnav


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


@pytest.mark.skip("Test unreliable; fails more often than not.")
def test_mvmp_slam_orbiting_square(do_debug_plot):
    # Overwrite this for now
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
    pvg_slam.DO_SAMPLE_TOPOLOGY = True

    pvg_mvmp = vertexnav.prob_vertex_graph.ProbVertexGraph()
    pvg_mvmp.DO_SLAM = True
    pvg_mvmp.DO_SAMPLE_TOPOLOGY = True

    # Set the noise models
    pvg_slam.PRIOR_NOISE = np.array([0.3, 0.3, 0.1])
    pvg_slam.ODOMETRY_NOISE = np.array([0.02, 0.02, 0.5])
    pvg_slam.CLUSTERING_NOISE = np.array([0.005, 0.005])

    # Set the noise models
    pvg_mvmp.PRIOR_NOISE = np.array([0.3, 0.3, 0.1])
    pvg_mvmp.ODOMETRY_NOISE = np.array([0.02, 0.02, 0.5])
    pvg_mvmp.CLUSTERING_NOISE = np.array([0.005, 0.005])

    association_window = 1

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
                             err=err,
                             association_window=association_window)

        _add_new_observation(pvg=pvg_mvmp,
                             world=world,
                             pose=pose,
                             odom=odom,
                             err=err,
                             association_window=association_window)

        p = pvg_error.r_poses[len(pvg_error.r_poses) - 1]
        print((p.x, p.y, p.yaw, p.index))
        p = pvg_slam.r_poses[len(pvg_slam.r_poses) - 1]
        print((p.x, p.y, p.yaw, p.index))

    if do_debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()

        ax1 = plt.subplot(1, 4, 1)
        ax2 = plt.subplot(1, 4, 2)
        ax3 = plt.subplot(1, 4, 3)
        ax4 = plt.subplot(1, 4, 4)

        _plot_pvg_data(ax1, pvg_perfect, world, "Perfect Odometry")
        _plot_pvg_data(ax2, pvg_error, world, "Noisy Odom")
        _plot_pvg_data(ax3, pvg_slam, world, "Noisy Odom + SLAM")
        _plot_pvg_data(ax4, pvg_mvmp, world, "Noisy Odom + SLAM + MVMP")

        plt.show()

    assert not (pvg_perfect.r_poses[len(pvg_perfect.r_poses) - 1]
                == pvg_error.r_poses[len(pvg_error.r_poses) - 1])

    assert len(pvg_perfect.vertices) == 4
    assert len(pvg_error.vertices) > 4
    assert len(pvg_slam.vertices) > 4
    assert len(pvg_mvmp.vertices) > 4

    _, state_perfect = pvg_perfect.sample_vertices(do_update_state=False)
    _, state_error = pvg_error.sample_vertices(do_update_state=False)

    # Sample topologies without and *with* MVMP to compare results.
    num_topology_samples = 40
    mvmp_merge_dist_threshold = 0.1
    _, state_slam = pvg_slam.sample_states(do_update_state=True,
                                           num_topology_samples=num_topology_samples, do_multi_vertex_merge=False)
    _, state_mvmp = pvg_mvmp.sample_states(do_update_state=True,
                                           num_topology_samples=num_topology_samples, do_multi_vertex_merge=True,
                                           mvmp_merge_dist_threshold=mvmp_merge_dist_threshold)

    if do_debug_plot:
        import matplotlib.pyplot as plt

        plt.figure()

        ax1 = plt.subplot(1, 4, 1)
        ax2 = plt.subplot(1, 4, 2)
        ax3 = plt.subplot(1, 4, 3)
        ax4 = plt.subplot(1, 4, 4)

        _plot_pvg_data(ax1, pvg_perfect, world, "Perfect Odometry")
        _plot_pvg_data(ax2, pvg_error, world, "Noisy Odom")
        _plot_pvg_data(ax3, pvg_slam, world, "Noisy Odom + SLAM")
        _plot_pvg_data(ax4, pvg_mvmp, world, "Noisy Odom + SLAM + MVMP")

        plt.show()

    proposed_world = pvg_mvmp.get_proposed_world()
    print("== MVMP ==")
    print(f"Number of vertices: {len(proposed_world.vertices)}")
    print(f"Number of walls: {len(proposed_world.walls)}")
    print(f"Length of topology: {len(pvg_mvmp.topology)}")

    print("== SLAM ==")
    proposed_world = pvg_slam.get_proposed_world()
    print(f"Number of vertices: {len(proposed_world.vertices)}")
    print(f"Number of walls: {len(proposed_world.walls)}")
    print(f"Length of topology: {len(pvg_slam.topology)}")

    assert state_perfect.log_prob > state_error.log_prob
    assert state_slam.log_prob > state_error.log_prob

    # Confirm that the 'perfect' world is as expected
    proposed_world = pvg_perfect.get_proposed_world()
    assert len(proposed_world.vertices) == 4
    for v in proposed_world.vertices:
        assert abs(v[0]) == pytest.approx(1.0, abs=1e-2)
        assert abs(v[1]) == pytest.approx(1.0, abs=1e-2)
    assert len(proposed_world.walls) == 4

    # Confirm the 'MVMP' world should look quite similar.
    proposed_world = pvg_mvmp.get_proposed_world()
    assert len(proposed_world.vertices) == 4
    for v in proposed_world.vertices:
        assert abs(v[0]) == pytest.approx(1.0, abs=5e-2)
        assert abs(v[1]) == pytest.approx(1.0, abs=5e-2)
    assert len(proposed_world.walls) == 4
