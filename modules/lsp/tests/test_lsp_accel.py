import itertools
import numpy as np
import random
import time

import lsp
import pytest


def _get_random_subgoal_data(seed, num_subgoals):
    random.seed(seed)

    cost_exp = 100
    cost_ds = 30
    dist_goal = 200
    dist_robot = 20
    dist_frontier = 50

    # Make a bunch of random data
    subgoals = [
        lsp.core.Frontier(np.array([[ii, ii]]).T) for ii in range(num_subgoals)
    ]
    [
        s.set_props(prob_feasible=random.random(),
                    delta_success_cost=cost_exp * random.random(),
                    exploration_cost=cost_ds * random.random())
        for s in subgoals
    ]

    goal_distances = {s: dist_goal * random.random() for s in subgoals}
    robot_distances = {s: dist_robot * random.random() for s in subgoals}
    frontier_distances = {
        frozenset(pair): dist_frontier * random.random()
        for pair in itertools.combinations(subgoals, 2)
    }

    distances = {
        'goal': goal_distances,
        'robot': robot_distances,
        'frontier': frontier_distances
    }

    return subgoals, distances


@pytest.mark.parametrize("num_subgoals", [1, 2, 8, 10, 15])
def test_subgoal_ordering_calc(num_subgoals):
    subgoals, distances = _get_random_subgoal_data(8616, num_subgoals)

    stime = time.time()
    cpp_cost, cpp_ordering = lsp.core.get_lowest_cost_ordering(
        subgoals, distances)
    print(f"C++ Time [#{num_subgoals}]: {time.time() - stime}")

    stime = time.time()
    py_cost, py_ordering = lsp.core.get_lowest_cost_ordering_old(
        subgoals, distances)
    for s in py_ordering:
        print(s.prob_feasible)
    print(py_cost)
    print(f"Python Time [#{num_subgoals}]: {time.time() - stime}")

    assert abs(cpp_cost - py_cost) < 0.001
    for cs, ps in zip(cpp_ordering, py_ordering):
        assert cs == ps

    if num_subgoals == 1:
        return

    _, cpp_ordering_bw = lsp.core.get_lowest_cost_ordering_beginning_with(
        cpp_ordering[0], subgoals, distances)
    _, cpp_ordering_bw_alt = lsp.core.get_lowest_cost_ordering_beginning_with(
        cpp_ordering[0], cpp_ordering, distances)

    for cs, csb, csba in zip(cpp_ordering, cpp_ordering_bw,
                             cpp_ordering_bw_alt):
        assert cs == csb
        assert cs == csba

    cpp_cost_bad, cpp_ordering_bad = lsp.core.get_lowest_cost_ordering_beginning_with(
        cpp_ordering[-1], cpp_ordering, distances)

    assert cpp_cost_bad > cpp_cost
    assert cpp_ordering_bad[0] == cpp_ordering[-1]
    assert len(cpp_ordering_bad) == len(cpp_ordering)


@pytest.mark.parametrize("num_subgoals", [1, 2, 8, 10, 15])
def test_subgoal_ordering_contains_all(num_subgoals):
    subgoals, distances = _get_random_subgoal_data(8616, num_subgoals)
    for s in subgoals:
        s.prob_feasible = 1

    cost, ordering = lsp.core.get_lowest_cost_ordering(
        subgoals, distances)

    assert len(ordering) == num_subgoals
