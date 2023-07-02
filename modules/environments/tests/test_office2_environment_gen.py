from environments import office2
import environments
import matplotlib.pyplot as plt
import scipy
import numpy as np


def test_office2_mapgen_intermediate(do_debug_plot, unity_path):

    def make_plotting_grid(known_map):
        kernel = np.ones((3, 3)) / 9
        grid = scipy.ndimage.convolve(known_map, kernel)
        walls = (known_map == 1) & (grid < 1)
        grid_map = known_map.copy()
        grid_map += 1
        grid_map[known_map == 0] = 0
        grid_map[known_map == 1] = 1
        grid_map[walls] = 2

        grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
        grid[:, :, 0][grid_map == 0] = 1
        grid[:, :, 0][grid_map == 2] = 0
        grid[:, :, 1][grid_map == 0] = 1
        grid[:, :, 1][grid_map == 2] = 0
        grid[:, :, 2][grid_map == 0] = 1
        grid[:, :, 2][grid_map == 2] = 0

        grid[:, :, 0][grid_map == 1] = 0.65
        grid[:, :, 1][grid_map == 1] = 0.65
        grid[:, :, 2][grid_map == 1] = 0.75

        return grid

    seed = 2005
    grid_with_lines, line_segments = office2.generate_random_lines(seed=seed)
    grid_with_hallway = office2.inflate_lines_to_create_hallways(grid_with_lines)
    features = office2.determine_intersections(grid_with_hallway == office2.semantic_labels['hallway'])
    grid_with_special_rooms, special_rooms_coords = office2.add_special_rooms(grid_with_hallway,
                                                                              intersections=features['intersections'])
    grid_with_rooms, rooms_coords = office2.add_rooms(grid_with_special_rooms, line_segments)
    rooms_coords += special_rooms_coords
    grid_with_tables, _ = office2.add_tables(grid_with_rooms, rooms_coords)

    known_map = (grid_with_tables <= office2.L_CLUTTER).astype(float)

    if do_debug_plot:
        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.imshow(grid_with_lines.T, cmap='viridis')
        plt.title('Randomly generated lines')
        plt.subplot(232)
        plt.imshow(grid_with_hallway.T, cmap='viridis')
        plt.title('Lines inflated to create hallways')
        plt.subplot(233)
        plt.imshow(grid_with_special_rooms.T, cmap='viridis')
        plt.title('Grid with special rooms')
        plt.subplot(234)
        plt.imshow(grid_with_rooms.T, cmap='viridis')
        plt.title('Grid with rooms')
        plt.subplot(235)
        plt.imshow(grid_with_tables.T, cmap='viridis')
        plt.title('Grid with tables')
        plt.subplot(236)
        plt.imshow(make_plotting_grid(known_map.T))
        plt.title('Final occupancy grid')
        plt.show()

    assert np.std(grid_with_lines) > 0
    assert np.std(grid_with_lines - grid_with_hallway) > 0
    assert np.std(grid_with_hallway - grid_with_rooms) > 0
    assert np.std(grid_with_special_rooms - grid_with_rooms) > 0
    assert np.std(grid_with_rooms - grid_with_tables) > 0


def test_office2_mapgen_final(do_debug_plot, unity_path):
    args = lambda: None  # noqa: E731
    args.current_seed = 2005
    args.map_type = 'office2'
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)

    if do_debug_plot:
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.imshow(known_map.T, cmap='binary')
        plt.scatter(pose.x, pose.y, color='blue')
        plt.scatter(goal.x, goal.y, color='green')
        plt.title('Occupancy grid')
        plt.subplot(212)
        plt.imshow(map_data['semantic_grid'].T, cmap='viridis')
        plt.title('Semantic grid')
        plt.show()

    assert np.std(known_map) > 0
    assert known_map.size > 0
    assert np.std(map_data['semantic_grid']) > 0
    assert map_data['semantic_grid'].size > 0


def test_office2_simulator(do_debug_plot, unity_path):
    args = lambda: None  # noqa: E731
    args.current_seed = 2005
    args.map_type = 'office2'
    args.base_resolution = office2.RESOLUTION
    args.inflation_radius_m = office2.INFLATION_RADIUS_M

    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)

    world = environments.simulated.OccupancyGridWorld(
        known_map, map_data, num_breadcrumb_elements=0)
    builder = environments.simulated.WorldBuildingUnityBridge

    with builder(unity_path) as unity_bridge:
        unity_bridge.move_object_to_pose('robot', args.base_resolution * pose)
        pano_image_init = unity_bridge.get_image('robot/pano_camera')
        unity_bridge.make_world(world)
        pano_image_start = unity_bridge.get_image('robot/pano_camera')
        unity_bridge.move_object_to_pose('robot', args.base_resolution * goal)
        pano_image_goal = unity_bridge.get_image('robot/pano_camera')

        if do_debug_plot:
            plt.figure(figsize=(8, 6))
            plt.subplot(211)
            plt.imshow(pano_image_start)
            plt.subplot(212)
            plt.imshow(pano_image_goal)
            plt.show()

        assert np.std(pano_image_start.mean(2)) > 0
        assert np.std(pano_image_goal.mean(2)) > 0
        assert np.std(pano_image_init - pano_image_start) > 0
        assert np.std(pano_image_start - pano_image_goal) > 0
        assert np.mean(pano_image_start) > 60
        assert np.mean(pano_image_goal) > 60
