import pytest
import mrlsp
import mrlsp_accel
import time
from util_function import current_state_2  # noqa: F401


def test_cost_action_values_without_sampling(current_state_2):  # noqa: F811
    '''This function tests whether hand calculated cost for two robot scenario is
    same as the cost from Q(state, action) function. This Q(state, action) function
    is not sampling based and calculates cost recursively.'''

    unexplored_frontiers = current_state_2['unexplored_frontiers_py']
    times = current_state_2['time']
    calculated_actions = current_state_2['calculated_actions']
    calculated_costs = current_state_2['calculated_costs']
    num_robots = current_state_2['num_robots']

    sigma1 = mrlsp.core.State(n=num_robots, Fu=unexplored_frontiers, m_t=times)

    actions = sigma1.get_actions()
    costs = [mrlsp.core.Q(sigma1, action) for action in actions]
    print(costs)
    for i, action in enumerate(actions):
        index = [x for x, y in enumerate(calculated_actions) if y[0] == action[0] and y[1] == action[1]]
        if index != []:
            assert pytest.approx(costs[i]) == calculated_costs[index[0]]


def test_actions_with_pouct_sampling_py(current_state_2):  # noqa: F811
    '''This test function tests whether the calculated action (python
    version) using sampling technique is same as hand calculated action
    (oracle truth) '''
    unexplored_frontiers = current_state_2['unexplored_frontiers_py']
    times = current_state_2['time']
    calculated_actions = current_state_2['calculated_actions']
    calculated_costs = current_state_2['calculated_costs']
    num_robots = current_state_2['num_robots']

    num_iterations = 5000
    start_time = time.time()
    best_action = mrlsp.pouct.find_best_joint_action(
        unexplored_frontiers=unexplored_frontiers, time=times, num_robots=num_robots, num_iterations=num_iterations)
    print(f"Time taken PY: {time.time() - start_time}")
    expected = calculated_actions[calculated_costs.index(min(calculated_costs))]
    print(f"best_action = {hash(best_action[0]), hash(best_action[1])}")
    print(f'expected_action = {hash(expected[0])}, {hash(expected[1])}')

    assert best_action in calculated_actions
    assert all([a == b for a, b in zip(best_action, expected)])


def test_actions_with_pouct_sampling_cpp(current_state_2):  # noqa: F811
    '''This test function tests whether the calculated action (cpp
    version) using sampling technique is same as hand calculated action
    (oracle truth) '''
    s_cpp = current_state_2['s_cpp']
    rd_cpp = current_state_2['rd_cpp']
    gd_cpp = current_state_2['gd_cpp']
    fd_cpp = current_state_2['fd_cpp']
    num_robots = current_state_2['num_robots']
    calculated_actions = current_state_2['calculated_actions']
    calculated_costs = current_state_2['calculated_costs']
    expected = calculated_actions[calculated_costs.index(min(calculated_costs))]

    num_iterations = 5000
    start_time = time.time()
    best_action = mrlsp_accel.find_best_joint_action_accel(num_robots, s_cpp, rd_cpp, gd_cpp, fd_cpp, num_iterations)
    print(f"Time taken CPP: {time.time() - start_time}")
    print(f'{best_action=}')
    print(f'expected_action = {hash(expected[0])}, {hash(expected[1])}')

    assert all([a == hash(b) for a, b in zip(best_action, expected)])
