import numpy as np
import random
import common
import lsp


def test_lsp_utils_get_oriented_input():
    """Test that the process by which we compute the input to the neural network
obeys a few properties."""
    random.seed(1234)

    num_bearing = 512

    class Subgoal(object):
        def get_centroid(self):
            return self.centroid

    # If the goal is at the centroid, the vectors should have special values
    # at the center of the image
    do_not_match_count = 0
    num_poses = 100
    for _ in range(num_poses):
        subgoal = Subgoal()
        subgoal.centroid = [10 * random.random(), 10 * random.random()]
        rpose = common.Pose(-5 * random.random(), -5 * random.random())
        gpose = common.Pose(subgoal.get_centroid()[0],
                            subgoal.get_centroid()[1])

        gvx_r, _ = lsp.utils.learning_vision.get_rel_goal_loc_vecs(
            rpose, gpose, num_bearing)
        gvx_s, _ = lsp.utils.learning_vision.get_rel_goal_loc_vecs(
            rpose, gpose, num_bearing, subgoal)

        cent_r = np.argmax(gvx_r)
        cent_s = np.argmax(gvx_s)

        assert cent_s - num_bearing // 2 == 0

        if not cent_r == cent_s:
            do_not_match_count += 1
            assert abs(cent_r - num_bearing // 2) > 0

    assert do_not_match_count >= 0.9 * num_poses
