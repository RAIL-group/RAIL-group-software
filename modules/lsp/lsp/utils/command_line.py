"""Helper functions for interpreting and displaying to the command line.

Any of the provided scripts will provide full documentation of the args and
what they do if "--help" is passed to the command line. Inspection of this file also yields this information.
"""

import argparse
import math


def yes_or_no(question):
    reply = str(input(question + " (y/n): ")).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Invalid input: please enter")


def print_frontier_data(frontier, num_leading_spaces=4, print_weights=False):
    if print_weights:
        s = "TRAIN  (%6.2f %6.2f) | P %.6f | DS %8.2f (%8.2f) | EX %8.2f (%8.2f)" % (
            frontier.get_centroid()[0], frontier.get_centroid()[1],
            frontier.prob_feasible, frontier.delta_success_cost,
            frontier.positive_weighting,
            frontier.exploration_cost,
            frontier.negative_weighting)
    else:
        s = "TRAIN  (%6.2f %6.2f) | P %.6f | DS %8.2f | EX %8.2f" % (
            frontier.get_centroid()[0], frontier.get_centroid()[1],
            frontier.prob_feasible, frontier.delta_success_cost,
            frontier.exploration_cost)
    print(num_leading_spaces * " " + s)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Compare the different approaches for navigation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_dir',
                        type=str,
                        required=True,
                        help='Directory in which to save the data')
    parser.add_argument(
        '--silence',
        action='store_true',
        help='[Depricated] If set, all non-error print statements are squashed.'
    )
    parser.add_argument('--unity_path',
                        default=None,
                        help='Path to Unity environment.')

    group = parser.add_argument_group('General Simulation Arguments')
    group.add_argument('--seed',
                       type=int,
                       nargs='+',
                       required=False,
                       default=None,
                       help='The seed for the random number generation')

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
                       default=10,
                       help='Base number used to generate motion primitives.')
    group.add_argument('--laser_max_range_m',
                       type=float,
                       required=False,
                       default=12.0,
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

    group = parser.add_argument_group('Mapping and Planning Arguments')
    group.add_argument('--base_resolution',
                       type=float,
                       required=False,
                       default=None,
                       help='The size of one sim grid cell')
    group.add_argument(
        '--inflation_radius_m',
        type=float,
        required=False,
        default=0.4,
        help='How much to inflate the grid (in units of meters).')
    group.add_argument(
        '--disable_known_grid_correction',
        action='store_true',
        help='Do not use the known grid to correct the observed map.')

    # Map Arguments
    group = parser.add_argument_group('Map Generation Arguments')
    group.add_argument('--map_type', type=str)
    group.add_argument(
        '--map_file',
        type=str,
        nargs='+',
        default=None,
        help='Image file(s) imported via the "loader" map type.')

    return parser
