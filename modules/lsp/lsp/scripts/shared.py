import argparse
import math


def get_command_line_parser():
    parser = argparse.ArgumentParser(
        description='Learning over Subgoals Planning (core)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--silence', action='store_true')
    parser.add_argument('--current_seed', required=True, type=int)
    parser.add_argument('--logfile_name', type=str, default='logfile.txt')

    parser.add_argument('--save_dir',
                        type=str,
                        required=True,
                        help='Directory in which to save the data')

    parser.add_argument('--unity_path',
                        type=str,
                        required=True,
                        help='Path to the Unity simulator.')

    group = parser.add_argument_group('Robot and Sensor Arguments')
    group.add_argument('--step_size',
                       type=float,
                       required=False,
                       default=1.8,
                       help='The step size for the robot motion')
    group.add_argument('--max_primitive_yaw',
                       type=float,
                       required=False,
                       default=math.pi / 3,
                       help='Maximum yaw robot can turn per step.')
    group.add_argument('--num_primitives',
                       type=int,
                       required=False,
                       default=32,
                       help='Base number used to generate motion primitives.')
    group.add_argument('--laser_max_range_m',
                       type=float,
                       required=False,
                       help='Max laser range (in meters)')
    group.add_argument('--laser_scanner_num_points',
                       type=int,
                       default=1024,
                       help='Number of points in simulated laser scan.')
    group.add_argument('--field_of_view_deg',
                       type=float,
                       required=False,
                       default=360.0,
                       help='Robot field of view (in degrees).')

    # Map Arguments
    group = parser.add_argument_group('Map Generation Arguments')
    group.add_argument('--map_type', type=str)
    group.add_argument(
        '--map_file',
        type=str,
        nargs='+',
        default=None,
        help='Image file(s) imported via the "loader" map type.')

    group = parser.add_argument_group('Mapping and Planning Arguments')
    group.add_argument(
        '--disable_known_grid_correction',
        action='store_true',
        help='Do not use the known grid to correct the observed map.')

    return parser
