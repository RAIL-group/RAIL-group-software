import environments
import lsp
import numpy as np
import matplotlib.pyplot as plt
import time


def test_simulator_speed(unity_path, do_debug_plot):
    parser = lsp.utils.command_line.get_parser()
    args = parser.parse_args(['--save_dir', ''])
    args.current_seed = 100
    args.step_size = 1.8
    args.field_of_view_deg = 360
    args.map_type = 'maze'
    args.base_resolution = 0.3
    args.inflation_radius_m = 0.75
    args.laser_max_range_m = 18
    args.unity_path = unity_path
    num_frames = 100

    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements,
        min_breadcrumb_signed_distance=4.0 * args.base_resolution)
    builder = environments.simulated.WorldBuildingUnityBridge

    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)
        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)
        time_taken_list = []
        for i in range(num_frames):
            pose = world.get_random_pose()
            t = time.time()
            unity_bridge.move_object_to_pose("robot", simulator.pose_grid_to_world(pose))
            pano_image = unity_bridge.get_image("robot/pano_camera")
            frame_time = time.time() - t
            time_taken_list.append(frame_time)
            print(f'Frame {i + 1} Time: {frame_time}')

            if do_debug_plot:
                plt.ion()
                plt.figure(1)
                plt.clf()
                plt.subplot(111)
                plt.imshow(pano_image)
                plt.title(f'Frame {i + 1} Time: {frame_time}')
                plt.show()
                plt.pause(0.01)

        print('With ceiling and floor')
        print(f'Average time per frame (with {num_frames} frames): {np.mean(time_taken_list)}')
        print(f'Standard deviation: {np.std(time_taken_list)}')

        # Remove all ceiling/floor poses
        world.ceiling_poses = []

    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)
        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)
        time_taken_list = []
        for i in range(num_frames):
            pose = world.get_random_pose()
            t = time.time()
            unity_bridge.move_object_to_pose("robot", simulator.pose_grid_to_world(pose))
            pano_image = unity_bridge.get_image("robot/pano_camera")
            frame_time = time.time() - t
            time_taken_list.append(frame_time)
            print(f'Frame {i + 1} Time: {frame_time}')

            if do_debug_plot:
                plt.ion()
                plt.figure(1)
                plt.clf()
                plt.subplot(111)
                plt.imshow(pano_image)
                plt.title(f'Frame {i + 1} Time: {frame_time}')
                plt.show()
                plt.pause(0.01)

        print('Without ceiling and floor')
        print(f'Average time per frame (with {num_frames} frames): {np.mean(time_taken_list)}')
        print(f'Standard deviation: {np.std(time_taken_list)}')

    if do_debug_plot:
        plt.ioff()
