import environments
import vertexnav
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely import geometry


def get_world_square():
    # A square
    square_poly = geometry.Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    return vertexnav.world.World(obstacles=[square_poly])


def test_vertexnav_data_formatted_correctly(do_debug_plot, unity_path):
    world = get_world_square()
    world.breadcrumb_element_poses = []
    world.breadcrumb_type = None
    world_building_unity_bridge = \
        environments.simulated.WorldBuildingUnityBridge
    with world_building_unity_bridge(unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        for counter in range(10):
            datum = vertexnav.learning.get_vertex_datum_for_pose(
                vertexnav.Pose(0, 0, 2 * np.pi * random.random()),
                world,
                unity_bridge,
                max_range=10,
                num_range=32,
                num_bearing=128,
                min_range=0.2)

            print(counter)

            if do_debug_plot:
                plt.subplot(311)
                plt.imshow(datum['image'])
                plt.subplot(312)
                plt.imshow(datum['depth'])
                plt.subplot(313)
                plt.imshow(datum['is_vertex'])
                plt.show()

            assert datum['image'].max() <= 1.0
            assert datum['image'].min() >= 0.0

            # Pick a few rotations and confirm that the minimum depth corresponds to
            # a vertex location.
            min_depth_loc = np.argmin(datum['depth'][64, :])
            depth_cols = datum['depth'].shape[1]
            corner_vertex_loc = np.argmax(np.max(datum['is_corner'], axis=0))
            vertex_cols = datum['is_corner'].shape[1]
            diff = min_depth_loc / depth_cols - corner_vertex_loc / vertex_cols
            assert abs(diff) < 0.05 or abs(abs(diff) - 1) < 0.05
