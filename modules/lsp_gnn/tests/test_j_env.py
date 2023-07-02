import environments.generate
import lsp_gnn
import matplotlib.pyplot as plt
from common import Pose
import numpy as np


def test_jshaped_env_occupancy_gen(do_debug_plot, unity_path):
    args = lambda: None  # noqa: E731
    args.current_seed = 0
    args.map_type = 'jshaped'
    args.base_resolution = 0.4
    args.inflation_radius_m = 0.75
    known_map, map_data, start_pose, goal_pose = \
        lsp_gnn.environments.generate.map_and_poses(args)

    if do_debug_plot:
        import shapely.geometry

        def plot_shapely_linestring(ax, ls, color=[0.25, 0.25, 1.0], alpha=1.0):
            if isinstance(ls, shapely.geometry.MultiLineString):
                [plot_shapely_linestring(ax, line, color, alpha) for line in ls]
                return

            x, y = ls.xy
            ax.plot(x, y, color=color, alpha=alpha, linewidth=0.2)

        plt.subplot(121)
        plt.imshow(known_map)
        plt.subplot(122)
        plt.imshow(map_data['semantic_grid'])
        plt.show()

    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        known_map, map_data,
        num_breadcrumb_elements=0,  # args.num_breadcrumb_elements,
        min_interlight_distance=3.0,
        min_light_to_wall_distance=1.5)
    builder = environments.simulated.WorldBuildingUnityBridge

    with builder(unity_path) as unity_bridge:
        unity_bridge.make_world(world)
        unity_bridge.move_object_to_pose("robot", args.base_resolution * start_pose)
        pano_image = unity_bridge.get_image("robot/pano_camera")
        if do_debug_plot:
            plt.imshow(pano_image)
            plt.show()


def test_jshaped_env_wall(do_debug_plot, unity_path):
    args = lambda: None  # noqa: E731
    args.current_seed = 0
    args.map_type = 'jshaped'
    args.base_resolution = 0.4
    args.inflation_radius_m = 0.75
    known_map, map_data, start_pose, goal_pose = \
        lsp_gnn.environments.generate.map_and_poses(args)

    if do_debug_plot:
        import shapely.geometry

        def plot_shapely_linestring(ax, ls, color=[0.25, 0.25, 1.0], alpha=1.0):
            if isinstance(ls, shapely.geometry.MultiLineString):
                [plot_shapely_linestring(ax, line, color, alpha) for line in ls]
                return

            x, y = ls.xy
            ax.plot(x, y, color=color, alpha=alpha, linewidth=0.2)

        plt.subplot(121)
        plt.imshow(known_map)
        plt.subplot(122)
        plt.imshow(map_data['semantic_grid'])
        plt.show()

    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        known_map, map_data,
        num_breadcrumb_elements=0,
        min_interlight_distance=3.0,
        min_light_to_wall_distance=1.5)
    builder = environments.simulated.WorldBuildingUnityBridge

    with builder(unity_path) as unity_bridge:
        unity_bridge.make_world(world)
        start_pose = Pose(x=55, y=16, yaw=0)
        unity_bridge.move_object_to_pose("robot", args.base_resolution * start_pose)
        pano_image_0 = unity_bridge.get_image("robot/pano_camera")
        pano_seg_image_0 = unity_bridge.get_image("robot/pano_segmentation_camera")
        inside_j = Pose(x=15, y=54, yaw=0)
        unity_bridge.move_object_to_pose("robot", args.base_resolution * inside_j)
        pano_image_1 = unity_bridge.get_image("robot/pano_camera")
        pano_seg_image_1 = unity_bridge.get_image("robot/pano_segmentation_camera")
        if do_debug_plot:
            plt.figure(figsize=(12, 8))
            plt.subplot(221)
            plt.imshow(pano_image_0)
            plt.title(f"start x: {start_pose.x}, y: {start_pose.y}")
            plt.subplot(223)
            plt.imshow(pano_seg_image_0)
            plt.subplot(222)
            plt.imshow(pano_image_1)
            plt.subplot(224)
            plt.imshow(pano_seg_image_1)
            plt.show()

        assert np.std(pano_seg_image_0) > 1  # 1 is just a buffer. It mostly just needs to be nonzero
        assert np.std(pano_seg_image_1) > 1  # 1 is just a buffer. It mostly just needs to be nonzero
