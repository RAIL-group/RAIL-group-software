import gridmap
import lsp
import numpy as np


def compute_frontier_negative_weighting(
        occupancy_grid, known_grid, all_frontiers, chosen_frontier, start_pose,
        end_pose, exploration_cost):
    """How expensive it is to misclassify a negative frontier.
    This is simply the exploration cost. (Decided after much deliberation).
    """
    return exploration_cost


def compute_frontier_positive_weighting(occupancy_grid, known_grid,
                                        all_frontiers, chosen_frontier,
                                        start_pose, end_pose):
    """How expensive it is to misclassify a positive frontier.
    The positive weighting is computed by determining how expensive it would be
    to navigate to the goal if the fronteir were classified as negative. We
    first compute the cost function to plan if the chosen frontier is masked.
    We compute the distance to the farthest point in unknown space, double it
    and return that. This is more or less an upper bound on first-order-cost,
    but it works well in practice.
    """

    # Copy the occupancy grid
    known_grid = known_grid.copy()
    occupancy_grid = occupancy_grid.copy()

    # Mask the grid with chosen frontier (after inflation)
    pos_frontiers = [f for f in all_frontiers if f.prob_feasible > 0.5]
    masked_grid = lsp.core.mask_grid_with_frontiers(known_grid, pos_frontiers)

    cost_grid = gridmap.planning.compute_cost_grid_from_position(
        masked_grid, [start_pose.x, start_pose.y], only_return_cost_grid=True)

    # Compute the cost to the farthest unknown point
    cost_grid[occupancy_grid == lsp.constants.FREE_VAL] = 0
    cost_grid[occupancy_grid == lsp.constants.COLLISION_VAL] = 0
    cost_grid[cost_grid == float("inf")] = 0
    exploration_cost = 2 * np.amax(cost_grid)

    # Now compute the cost from the robot to the goal through the frontier
    return exploration_cost
