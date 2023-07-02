"""Primary functions for dispatching map generation."""
from gridmap.utils import inflate_grid
from gridmap.planning import compute_cost_grid_from_position
import common
import lsp_gnn
import numpy as np
import random


def MapGenerator(args):
    if args.map_type.lower() == 'jshaped':
        return lsp_gnn.environments.jshaped.MapGenJshaped(args)
    elif args.map_type.lower() == 'new_office':
        return lsp_gnn.environments.parallel_hallway.MapGenParallel(args)
    else:
        raise ValueError('Map type "%s" not recognized' % args.map_type)


def map_and_poses(args, num_attempts=10, Pose=common.Pose):
    seed = args.current_seed
    random.seed(seed)
    np.random.seed(seed)
    """Helper function that generates a map and feasible start end poses"""

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

    return grid, map_data, start, goal
