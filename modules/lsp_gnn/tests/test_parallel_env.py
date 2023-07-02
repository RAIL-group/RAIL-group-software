import numpy as np
import matplotlib.pyplot as plt

import lsp_gnn
import environments


def get_args():
    args = lambda: None  # noqa
    args.current_seed = 0
    args.map_type = 'new_office'
    args.base_resolution = .4
    args.inflation_radius_m = .75
    return args


def test_parallel_hallway_sim(do_debug_plot, unity_path):
    ''' Tests if parallel office is correctly generating the map.
    Also checks if the simulator is generating the images/seg-images
    from different poses.
    '''
    args = get_args()
    known_map, map_data, pose, goal = \
        lsp_gnn.environments.generate.map_and_poses(args)

    occupancy_grid = map_data['occ_grid'].copy()
    occupancy_grid[int(pose.x)][int(pose.y)] = 3
    occupancy_grid[int(goal.x)][int(goal.y)] = 4
    if do_debug_plot:
        plt.subplot(121)
        plt.imshow(occupancy_grid)
        plt.subplot(122)
        plt.imshow(map_data['semantic_grid'])
        plt.show()

    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        map_data['occ_grid'], map_data,
        num_breadcrumb_elements=0,
        min_interlight_distance=3.0,
        min_light_to_wall_distance=1.5)
    builder = environments.simulated.WorldBuildingUnityBridge

    with builder(unity_path) as unity_bridge:
        unity_bridge.make_world(world)
        unity_bridge.move_object_to_pose("robot", args.base_resolution * pose)
        pano_image = unity_bridge.get_image("robot/pano_camera")
        unity_bridge.move_object_to_pose("robot", args.base_resolution * goal)
        pano_image_1 = unity_bridge.get_image("robot/pano_camera")
        if do_debug_plot:
            plt.figure(figsize=(16, 8))
            plt.subplot(211)
            plt.imshow(pano_image_1)
            plt.subplot(212)
            plt.imshow(pano_image)
            plt.show()
        assert np.std(pano_image) > 1  # 1 is just a buffer. It mostly just needs to be nonzero
        assert np.std(pano_image_1) > 1  # 1 is just a buffer. It mostly just needs to be nonzero


def test_parallel_hallway_semantic(do_debug_plot, unity_path):
    '''Tests if parallel office is correctly generating the map.'''
    args = get_args()
    known_map, map_data, pose, goal = \
        lsp_gnn.environments.generate.map_and_poses(args)

    if do_debug_plot:
        occupancy_grid = map_data['occ_grid'].copy()
        occupancy_grid[int(pose.x)][int(pose.y)] = 3
        occupancy_grid[int(goal.x)][int(goal.y)] = 4
        plt.subplot(121)
        plt.imshow(occupancy_grid)
        plt.subplot(122)
        plt.imshow(map_data['semantic_grid'])
        plt.show()


def test_parallel_hallway_count(do_debug_plot, unity_path):
    '''Tests if the map generator is creating required number of hallways'''
    args = get_args()
    for seed in range(100, 101):
        args.current_seed = seed
        known_map, map_data, pose, goal = \
            lsp_gnn.environments.generate.map_and_poses(args)
        assert map_data['hallways_count'] == lsp_gnn.environments.parallel_hallway.NUM_OF_HALLWAYS
