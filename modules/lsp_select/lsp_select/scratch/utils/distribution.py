import numpy as np
import gridmap
import common
import lsp


def get_cost_distribution(frontiers, distances):
    probs = [1]
    costs = [[distances['robot'][frontiers[0]], 0]]
    prev_cost_std = 0
    for i in range(len(frontiers)):
        f = frontiers[i]
        if i > 0:
            costs[-1][0] += distances['frontier'][frozenset([f, frontiers[i - 1]])]
        prev_cost_mean, prev_cost_std = costs[-1]
        success_cost_mean = prev_cost_mean + distances['goal'][f] + f.delta_success_cost
        success_cost_std = (f.delta_success_cost_std**2 + prev_cost_std**2)**0.5
        costs[-1] = [success_cost_mean, success_cost_std]
        prev_prob = probs[-1]
        probs[-1] *= f.prob_feasible
        probs.append(prev_prob * (1 - f.prob_feasible))
        costs.append([prev_cost_mean + f.exploration_cost, (prev_cost_std**2 + f.exploration_cost_std**2)**0.5])
    return np.array(probs), np.array(costs)


def corrupt_robot_pose(known_map, args):
    np.random.seed(args.current_seed)
    inflation_radius = args.inflation_radius_m / args.base_resolution
    inflated_mask = gridmap.utils.inflate_grid(known_map,
                                               inflation_radius=inflation_radius,
                                               collision_val=lsp.constants.COLLISION_VAL) < 1
    # Now sample a random point
    allowed_indices = np.where(inflated_mask)
    idx_start = np.random.randint(0, allowed_indices[0].size - 1)
    new_start_pose = common.Pose(x=allowed_indices[0][idx_start],
                                 y=allowed_indices[1][idx_start],
                                 yaw=2 * np.pi * np.random.rand())

    return new_start_pose
