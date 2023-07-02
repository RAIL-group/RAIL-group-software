"""This test is mostly for *visual* inspection of the office environment."""
import environments.generate
import matplotlib.pyplot as plt


def test_office_env_occupancy_gen(do_debug_plot, unity_path):
    args = lambda: None  # noqa: E731
    args.current_seed = 8616
    args.map_type = 'office'
    known_map, map_data, start_pose, goal_pose = environments.generate.map_and_poses(args)

    if do_debug_plot:
        import shapely.geometry

        def plot_shapely_linestring(ax, ls, color=[0.25, 0.25, 1.0], alpha=1.0):
            if isinstance(ls, shapely.geometry.MultiLineString):
                [plot_shapely_linestring(ax, line, color, alpha) for line in ls]
                return

            x, y = ls.xy
            ax.plot(x, y, color=color, alpha=alpha, linewidth=0.2)

        plt.subplot(131)
        plt.imshow(known_map)
        plt.subplot(132)
        plt.imshow(map_data['semantic_grid'])
        plt.subplot(133)
        for wall in map_data['walls']['room']:
            plot_shapely_linestring(plt.gca(), wall, color=[1.0, 0, 0])
        for wall in map_data['walls']['hallway']:
            plot_shapely_linestring(plt.gca(), wall, color=[0.0, 0, 1.0])
        for wall in map_data['walls']['base']:
            plot_shapely_linestring(plt.gca(), wall, color=[0.5, 0.5, 0.5])
        plt.show()

    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        known_map, map_data, num_breadcrumb_elements=0,
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
