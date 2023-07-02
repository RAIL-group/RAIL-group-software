import argparse
import environments
import vertexnav
import learning
import matplotlib.pyplot as plt
import numpy as np
import os


def data_gen_loop_dungeon(args, seed):
    """Loop through poses and write data to file."""
    WorldBuildingUnityBridge = environments.simulated.WorldBuildingUnityBridge

    with WorldBuildingUnityBridge(args.unity_path, sim_scale=0.15) as unity_bridge:
        print(f"Generating new world (seed: {seed})")
        world = vertexnav.environments.dungeon.DungeonWorld(hall_width=20,
                                                            inflate_ratio=0.30)
        unity_bridge.make_world(world)
        for counter in range(args.poses_per_world):
            pose = world.get_random_pose(min_signed_dist=2.0)
            datum = vertexnav.learning.get_vertex_datum_for_pose(
                pose,
                world,
                unity_bridge,
                min_range=2.0,
                max_range=args.max_range,
                num_range=args.num_range,
                num_bearing=args.num_bearing)

            if datum is None:
                continue

            if args.do_visualize_data:
                print("Visualizing Data")
                visualize_datum(datum)
                plt.show()

            # Write datum to file
            write_datum_to_pickle(args, counter, datum)
            counter += 1
            print(f"Saved pose: {args.seed}.{counter}")

        # Write a final file that indicates training is done
        plt.figure(figsize=(6, 6))
        vertexnav.plotting.plot_world(plt.gca(), world)
        plt.title(f'Seed: {args.seed}')
        plt.savefig(os.path.join(args.base_data_path, 'data',
                                 'training_env_plots',
                                 f'{args.data_plot_name}_{args.seed}.png'),
                    dpi=150)


def write_datum_to_pickle(args, counter, datum):
    save_dir = os.path.join(args.base_data_path, 'data')
    data_filename = os.path.join('pickles', f'dat_{args.seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(
        os.path.join(save_dir, data_filename), datum)

    csv_filename = f'{args.data_file_base_name}_{args.seed}.csv'
    with open(os.path.join(save_dir, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')


def visualize_datum(datum):
    """Visualize data."""
    fig = plt.figure(figsize=(12, 12))
    (ax1, ax2, ax3) = fig.subplots(3, 1)
    ax2.set_aspect('equal')

    ax1.clear()
    for circle in np.argwhere(datum['is_left_gap']):
        ax1.plot(circle[1] * 512 / 128, 128 / 2, '.', color='c')
    for circle in np.argwhere(datum['is_corner']):
        ax1.plot(circle[1] * 512 / 128, 128 / 2, '.', color='m')
    for circle in np.argwhere(datum['is_right_gap']):
        ax1.plot(circle[1] * 512 / 128, 128 / 2, '.', color='y')
    for circle in np.argwhere(datum['is_point_vertex']):
        ax1.plot(circle[1] * 512 / 128, 128 / 2, '.', color='w')
    ax1.imshow(datum['image'])

    ax2.clear()
    ax2.imshow(datum['depth'])

    ax3.clear()
    vert_data = np.zeros(list(datum['is_vertex'].shape) + [3])
    vert_data[datum['is_vertex'] > 0.5] = 1.0
    vert_data[:, :, 0][datum['is_left_gap'] > 0.5] = 0
    vert_data[:, :, 1][datum['is_corner'] > 0.5] = 0
    vert_data[:, :, 2][datum['is_right_gap'] > 0.5] = 0
    ax3.imshow(vert_data)

    plt.show()


def get_parser():
    """Define the command line arguments."""
    parser = argparse.ArgumentParser(description='Generate vertext data.')
    parser.add_argument('--xpassthrough', type=str, default='false')
    parser.add_argument('--poses_per_world', default=250, type=int)
    parser.add_argument('--environment', type=str, default='dungeon')
    parser.add_argument('--unity_path', type=str)
    parser.add_argument('--base_data_path', type=str, default='/data/')
    parser.add_argument('--data_file_base_name', type=str, required=True)
    parser.add_argument('--data_plot_name', type=str, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max_range',
                        type=int,
                        help='Max range for range in output grid.')
    parser.add_argument('--num_range',
                        type=int,
                        help='Number of range cells in output grid.')
    parser.add_argument('--num_bearing',
                        type=int,
                        help='Number of bearing cells in output grid.')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.do_visualize_data = args.xpassthrough == 'true'
    data_gen_loop_dungeon(args, args.seed)

    # for seed in range(args.seed_range[0], args.seed_range[1]):
    #     print("Seed: {}".format(seed))
    #     random.seed(seed, version=1)
    #     np.random.seed(seed)
    #     args.current_seed = seed

    #     if args.environment.lower() == 'dungeon':
    #     else:
    #         raise ValueError("Environment '{}' not recognized.".format(args.environment))
