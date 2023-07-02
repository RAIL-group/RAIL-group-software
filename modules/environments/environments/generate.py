"""Primary functions for dispatching map generation."""
from gridmap.utils import inflate_grid
from gridmap.planning import compute_cost_grid_from_position
import common


def _set_default_arguments(args, defaults):
    """Helper function to set default auguments for each map.

    Any argument not specified (or None) in the command line will be set as
    the default values in their respective environment.
    """
    for k, v in defaults.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)


def MapGenerator(args):

    if args.map_type.lower() == 'ploader':
        from . import pickle_loader
        _set_default_arguments(args, pickle_loader.DEFAULT_PARAMETERS)
        return pickle_loader.MapGenPLoader(args)
    elif args.map_type.lower() == 'maze':
        from . import guided_maze
        _set_default_arguments(args, guided_maze.DEFAULT_PARAMETERS)
        return guided_maze.MapGenMaze(args)
    elif args.map_type.lower() == 'office':
        from . import office
        _set_default_arguments(args, office.DEFAULT_PARAMETERS)
        return office.MapGenOffice(args)
    elif args.map_type.lower() == 'office2':
        from . import office2
        _set_default_arguments(args, office2.DEFAULT_PARAMETERS)
        return office2.MapGenOffice2(args)
    else:
        raise ValueError('Map type "%s" not recognized' % args.map_type)


def map_and_poses(args, num_attempts=1000, Pose=common.Pose):
    """Helper function that generates a map and feasible start end poses"""

    # Add some extra argumetns
    args.map_maze_path_width = 10
    args.map_maze_cell_dims = [8, 6]
    args.map_maze_wide_path_width = 14
    args.map_maze_all_wide = True

    # Generate a new map
    map_generator = MapGenerator(args)
    _, grid, map_data = map_generator.gen_map(random_seed=args.current_seed)

    # Initialize the sensor/robot variables
    inflation_radius = args.inflation_radius_m / args.base_resolution
    inflated_known_grid = inflate_grid(grid, inflation_radius=inflation_radius)

    # Get the poses (ensure they are connected)
    for _ in range(num_attempts):
        did_succeed, start, goal = map_generator.get_start_goal_poses()
        if not did_succeed:
            continue

        cost_grid, get_path = compute_cost_grid_from_position(
            inflated_known_grid, [goal.x, goal.y])
        did_plan, _ = get_path([start.x, start.y],
                               do_sparsify=False,
                               do_flip=False)
        if did_plan:
            break
    else:
        raise RuntimeError("Could not find a pair of poses that "
                           "connect during start/goal pose generation.")

    # A few other post-load-processing operations and arg-setting
    if map_data is None:
        map_data = dict()
    map_data['resolution'] = args.base_resolution
    map_data['x_offset'] = 0.0
    map_data['y_offset'] = 0.0

    if not hasattr(args, 'num_breadcrumb_elements'):
        args.num_breadcrumb_elements = 0
        args.breadcrumb_type = None

    return grid, map_data, start, goal
