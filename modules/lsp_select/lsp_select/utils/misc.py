import numpy as np
import lsp
import common
import gridmap


def corrupt_robot_pose(known_map, args):
    """Returns a randomly sampled pose in free space. Uses args.current_seed as random seed."""
    np.random.seed(args.current_seed)
    inflation_radius = args.inflation_radius_m / args.base_resolution
    inflated_mask = gridmap.utils.inflate_grid(known_map,
                                               inflation_radius=inflation_radius,
                                               collision_val=lsp.constants.COLLISION_VAL) < 1
    # Now sample a random point
    allowed_indices = np.where(inflated_mask)
    idx_start = np.random.randint(0, allowed_indices[0].size - 1)
    new_pose = common.Pose(x=allowed_indices[0][idx_start],
                           y=allowed_indices[1][idx_start],
                           yaw=2 * np.pi * np.random.rand())

    return new_pose
