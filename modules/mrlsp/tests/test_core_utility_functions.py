import pytest  # noqa: F401
import mrlsp
import mrlsp_accel
import itertools
from util_function import current_state_1, current_state_2, current_state_3  # noqa: F401
import util_function as test_shared_functions


# write a test for testing the cpp version of remaining_time by triangle fomulation and the python version
def test_remaining_frontier_time_by_triangle_formulation():
    all_a = [10, 10, 12, 14.9999, 150, 20, 21, 60]
    all_b = [5, 2, 22, 4.999999, 149.99, 21, 28, 27]
    all_c = [5, 15, 8, 20, 50.0, 29, 35, 33]
    all_time_travelled = [5, 8, 6, 6.98, 86, 17, 19.8, 50]

    for a, b, c, travel_time in zip(all_a, all_b, all_c, all_time_travelled):
        remaining_time_py = mrlsp.utils.utility.get_frontier_time_by_triangle_formation(a, b, c, travel_time)
        remaining_time_cpp = mrlsp_accel.get_frontier_time_by_triangle_formation_cpp(a, b, c, travel_time)
        assert remaining_time_cpp == remaining_time_py
        print(f'for a = {a}, b = {b}, c = {c}, travel_time = {travel_time}, \
                remaining_time_py = {remaining_time_py}, remaining_time_cpp = {remaining_time_cpp}')


# test progress and frontier from qt for cpp and python
def test_progress_and_frontier_from_qt(current_state_1):  # noqa: F811
    num_robots = current_state_1['num_robots']
    unexplored_frontiers = current_state_1['unexplored_frontiers_cpp']
    q_t_py = current_state_1['q_t']
    q_t_cpp = current_state_1['q_t_cpp']
    s_dict = current_state_1['s_dict']

    actual_progress = [10, 12]
    actual_frontiers = [unexplored_frontiers[0], unexplored_frontiers[1]]
    calculated_progress_py = []
    calculated_frontiers_py = []
    calculated_progress_cpp = []
    calculated_frontiers_cpp = []

    for i in range(num_robots):
        frontier, progress = mrlsp.core.find_progress_and_frontier_for_robot(q_t_py, i)
        calculated_frontiers_py.append(frontier)
        calculated_progress_py.append(progress)

    for i in range(num_robots):
        frontier, progress = mrlsp_accel.find_progress_and_frontier_for_robot_cpp(q_t_cpp, i)
        calculated_frontiers_cpp.append(s_dict[frontier])
        calculated_progress_cpp.append(progress)

    print(f'{calculated_progress_py=}, {actual_progress=}')
    print(f'{calculated_frontiers_py=}, {actual_frontiers=}')

    assert all([x == y for x, y in zip(calculated_progress_py, actual_progress)])
    assert all([x == y for x, y in zip(calculated_frontiers_py, actual_frontiers)])
    assert all([x == y for x, y in zip(calculated_progress_py, calculated_progress_cpp)])
    assert all([x == y for x, y in zip(calculated_frontiers_py, calculated_frontiers_cpp)])


def test_first_explored_frontier_and_time(current_state_1):  # noqa: F811
    num_robots = current_state_1['num_robots']
    unexplored_frontiers = current_state_1['unexplored_frontiers_cpp']
    time = current_state_1['time']

    a1 = unexplored_frontiers[0]
    a2 = unexplored_frontiers[1]
    a3 = unexplored_frontiers[2]

    s1 = mrlsp.core.State(n=num_robots, Fu=unexplored_frontiers, m_t=time)
    actions = [[a1, a2], [a1, a3], [a2, a1], [a2, a3], [a3, a1], [a3, a2]]
    calculated_frontiers = []
    actual_frontiers = [a1, a1, a1, a3, a1, a3]

    calculated_TI = []
    actual_TI = [13, 13, 17, 18, 17, 18]

    for action in actions:
        f_I, T_I = mrlsp.core.get_frontier_of_knowlege_and_time(s1, action)
        calculated_frontiers.append(f_I)
        calculated_TI.append(T_I)

    print(f'{calculated_frontiers=}, {actual_frontiers=}')
    print(f'{calculated_TI=}, {actual_TI=}')

    assert all([x == y for x, y in zip(calculated_frontiers, actual_frontiers)])
    assert all([x == y for x, y in zip(calculated_TI, actual_TI)])


# test first explored frontier and time for cpp and python
@pytest.mark.parametrize(
    "current_state_name, num_robots",
    [
        ("current_state_1", 1),
        ("current_state_1", 2),
        ("current_state_2", 1),
        ("current_state_2", 2),
        ("current_state_3", 1),
        ("current_state_3", 2),
        ("current_state_3", 3),
    ]
)
def test_first_explored_frontier_and_time_cpp(current_state_name, num_robots, request):
    current_state = request.getfixturevalue(current_state_name)
    num_robots = num_robots
    unexplored_frontiers = current_state['unexplored_frontiers_cpp']
    time = current_state['time']
    s_cpp = current_state['s_cpp']
    rd_cpp = current_state['rd_cpp']
    gd_cpp = current_state['gd_cpp']
    fd_cpp = current_state['fd_cpp']
    s_dict = current_state['s_dict']

    s1_py = mrlsp.core.State(n=num_robots, Fu=unexplored_frontiers, m_t=time)
    actions = [action for action in itertools.permutations(unexplored_frontiers, num_robots)]
    calculated_frontiers_py = []
    calculated_TI_py = []
    calculated_frontiers_cpp = []
    calculated_TI_cpp = []

    for action in actions:
        f_I, T_I = mrlsp.core.get_frontier_of_knowlege_and_time(s1_py, action)
        calculated_frontiers_py.append(f_I)
        calculated_TI_py.append(T_I)

    s1_cpp = mrlsp_accel.State_cpp(num_robots=num_robots, unexplored_frontiers=s_cpp, robot_distances=rd_cpp,
                                   goal_distances=gd_cpp, frontier_distances=fd_cpp)

    actions_cpp = [[hash(a) for a in action] for action in actions]

    for action in actions_cpp:
        f_I, T_I = mrlsp_accel.get_frontier_of_knowledge_and_time_cpp(s1_cpp, action)
        calculated_frontiers_cpp.append(s_dict[f_I])
        calculated_TI_cpp.append(T_I)

    print(f'{calculated_TI_py=}, {calculated_TI_cpp=}')

    # check that the python and cpp versions of the functions return the same results
    assert all([x == y for x, y in zip(calculated_frontiers_py, calculated_frontiers_cpp)])
    assert all([x == y for x, y in zip(calculated_TI_py, calculated_TI_cpp)])


# write a test for testing q_t for the cpp version and the python version
@pytest.mark.parametrize(
    "current_state_name, num_robots",
    [
        ("current_state_1", 1),
        ("current_state_1", 2),
        ("current_state_2", 1),
        ("current_state_2", 2),
        ("current_state_3", 1),
        ("current_state_3", 2),
        ("current_state_3", 3),
    ]
)
def test_q_t_and_time(current_state_name, num_robots, request):  # noqa: F811
    num_robots = num_robots
    current_state = request.getfixturevalue(current_state_name)
    unexplored_frontiers = current_state['unexplored_frontiers_cpp']
    time = current_state['time']
    s_cpp = current_state['s_cpp']
    rd_cpp = current_state['rd_cpp']
    gd_cpp = current_state['gd_cpp']
    fd_cpp = current_state['fd_cpp']
    s_dict = current_state['s_dict']

    s1_py = mrlsp.core.State(n=num_robots, Fu=set(unexplored_frontiers), m_t=time)
    s1_cpp = mrlsp_accel.State_cpp(num_robots=num_robots, unexplored_frontiers=s_cpp, robot_distances=rd_cpp,
                                   goal_distances=gd_cpp, frontier_distances=fd_cpp)
    s1_q_t_py = s1_py.q_t
    s1_q_t_cpp = s1_cpp.q_t

    # Test that intialization of q_t is same for python and cpp State classes
    # Test that q_t is same for python and cpp State classes
    for prog_py, prog_cpp in zip(s1_q_t_py, s1_q_t_cpp):
        for f_py, p_py in prog_py.items():
            assert prog_cpp[hash(f_py)] == p_py

    actions = [action for action in itertools.permutations(unexplored_frontiers, num_robots)]
    for action in actions:
        test_shared_functions.check_py_and_cpp_q_t_match(s1_py, s1_cpp, action, s_dict)


# write a test to test move robots
@pytest.mark.parametrize(
    "current_state_name, num_robots",
    [
        ("current_state_1", 1),
        ("current_state_1", 2),
        ("current_state_2", 1),
        ("current_state_2", 2),
        ("current_state_3", 1),
        ("current_state_3", 2),
        ("current_state_3", 3),
    ]
)
def test_move_robots(current_state_name, num_robots, request):  # noqa: F811
    num_robots = num_robots
    current_state = request.getfixturevalue(current_state_name)
    unexplored_frontiers = current_state['unexplored_frontiers_cpp']
    time = current_state['time']
    s_cpp = current_state['s_cpp']
    rd_cpp = current_state['rd_cpp']
    gd_cpp = current_state['gd_cpp']
    fd_cpp = current_state['fd_cpp']
    s_dict = current_state['s_dict']

    s1_py = mrlsp.core.State(n=num_robots, Fu=set(unexplored_frontiers), m_t=time)
    s1_cpp = mrlsp_accel.State_cpp(num_robots=num_robots, unexplored_frontiers=s_cpp, robot_distances=rd_cpp,
                                   goal_distances=gd_cpp, frontier_distances=fd_cpp)

    # actions = [(a1, a2), (a1, a3), (a2, a1), (a2, a3), (a3, a1), (a3, a2)]
    actions = [action for action in itertools.permutations(unexplored_frontiers, num_robots)]

    for action in actions:
        test_shared_functions.check_py_and_cpp_move_robots_match(s1_py, s1_cpp, action, s_dict)


def test_tree_states(current_state_2):  # noqa: F811
    '''This test checks all the states of a tree from the current state (current_state_2)
    for every actions available from current_state_2.'''
    unexplored_frontiers = current_state_2['unexplored_frontiers_py']
    time = current_state_2['time']
    num_robots = current_state_2['num_robots']
    list_of_unexplored_frontiers = current_state_2['unexplored_frontiers_cpp']

    s_dict = current_state_2['s_dict']
    s_cpp = current_state_2['s_cpp']
    rd_cpp = current_state_2['rd_cpp']
    gd_cpp = current_state_2['gd_cpp']
    fd_cpp = current_state_2['fd_cpp']

    a1 = list_of_unexplored_frontiers[0]
    a2 = list_of_unexplored_frontiers[1]

    # Root of the tree
    s1_py = mrlsp.core.State(n=num_robots, Fu=set(unexplored_frontiers), m_t=time)
    s1_cpp = mrlsp_accel.State_cpp(num_robots=num_robots, unexplored_frontiers=s_cpp, robot_distances=rd_cpp,
                                   goal_distances=gd_cpp, frontier_distances=fd_cpp)

    test_shared_functions.check_py_and_cpp_states_match(s1_py, s1_cpp, s_dict)

    actions_s1 = [(a1, a2), (a2, a1)]

    # Children of the root (1a)
    print("Root -> Child 1a")
    action1a_py = actions_s1[0]
    action1a_cpp = (hash(action1a_py[0]), hash(action1a_py[1]))
    s_1a_succ_py, s_1a_fail_py, f_I_py, T_I_py, goal_reached_py = mrlsp.core.move_robots(s1_py, action1a_py)
    s_1a_succ_cpp, s_1a_fail_cpp, f_I_cpp, T_I_cpp, goal_reached_cpp = mrlsp_accel.move_robots_cpp(s1_cpp, action1a_cpp)
    assert T_I_py == T_I_cpp
    assert T_I_py == 13
    assert f_I_cpp == hash(f_I_py)
    assert goal_reached_py == goal_reached_cpp
    print("Success")
    test_shared_functions.check_py_and_cpp_states_match(s_1a_succ_py, s_1a_succ_cpp, s_dict)
    print("Failure")
    test_shared_functions.check_py_and_cpp_states_match(s_1a_fail_py, s_1a_fail_cpp, s_dict)

    # Expanding child 1a (success)
    actions_s2 = [(a1, a2)]
    print("Child 1a (success) -> Child 2a")
    action2a_py = actions_s2[0]
    action2a_cpp = (hash(action2a_py[0]), hash(action2a_py[1]))
    s_2a_succ_py, s_2a_fail_py, f_I_py, T_I_py, goal_reached_py = mrlsp.core.move_robots(s_1a_succ_py, action2a_py)
    s_2a_succ_cpp, s_2a_fail_cpp, f_I_cpp, T_I_cpp, goal_reached_cpp = mrlsp_accel.move_robots_cpp(
        s_1a_succ_cpp, action2a_cpp)
    assert T_I_py == T_I_cpp
    assert T_I_py == 5
    assert f_I_cpp == hash(f_I_py)
    assert f_I_py == a2
    assert goal_reached_py == goal_reached_cpp
    print("Success")
    test_shared_functions.check_py_and_cpp_states_match(s_2a_succ_py, s_2a_succ_cpp, s_dict)
    print("Failure")
    test_shared_functions.check_py_and_cpp_states_match(s_2a_fail_py, s_2a_fail_cpp, s_dict)

    # Expanding child 1a (failure)
    actions_s2 = [(a2, a2)]
    print("Child 1a (fail) -> Child 2a")
    action2a_py = actions_s2[0]
    action2a_cpp = (hash(action2a_py[0]), hash(action2a_py[1]))
    s_2b_succ_py, s_2b_fail_py, f_I_py, T_I_py, goal_reached_py = mrlsp.core.move_robots(s_1a_fail_py, action2a_py)
    s_2b_succ_cpp, s_2b_fail_cpp, f_I_cpp, T_I_cpp, goal_reached_cpp = mrlsp_accel.move_robots_cpp(
        s_1a_fail_cpp, action2a_cpp)
    assert T_I_py == T_I_cpp
    assert T_I_py == 5
    assert f_I_cpp == hash(f_I_py)
    assert f_I_py == a2
    assert goal_reached_py == goal_reached_cpp
    print("Success")
    test_shared_functions.check_py_and_cpp_states_match(s_2b_succ_py, s_2b_succ_cpp, s_dict)
    print("Failure")
    test_shared_functions.check_py_and_cpp_states_match(s_2b_fail_py, s_2b_fail_cpp, s_dict)

    # Children of the root (1b)
    print("Root -> Child 1b")
    action1b_py = actions_s1[1]
    action1b_cpp = (hash(action1b_py[0]), hash(action1b_py[1]))
    s_1b_succ_py, s_1b_fail_py, f_I_py, T_I_py, goal_reached_py = mrlsp.core.move_robots(s1_py, action1b_py)
    s_1b_succ_cpp, s_1b_fail_cpp, f_I_cpp, T_I_cpp, goal_reached_cpp = mrlsp_accel.move_robots_cpp(s1_cpp, action1b_cpp)
    assert T_I_py == T_I_cpp
    assert T_I_py == 17
    assert f_I_cpp == hash(f_I_py)
    assert f_I_py == a1
    assert goal_reached_py == goal_reached_cpp
    test_shared_functions.check_py_and_cpp_states_match(s_1b_succ_py, s_1b_succ_cpp, s_dict)
    test_shared_functions.check_py_and_cpp_states_match(s_1b_fail_py, s_1b_fail_cpp, s_dict)

    # Expanding child 1b (success)
    actions_s2 = [(a2, a1)]
    print("Child 1b (success) -> Child 2a")
    action2a_py = actions_s2[0]
    action2a_cpp = (hash(action2a_py[0]), hash(action2a_py[1]))
    s_2a_succ_py, s_2a_fail_py, f_I_py, T_I_py, goal_reached_py = mrlsp.core.move_robots(s_1b_succ_py, action2a_py)
    s_2a_succ_cpp, s_2a_fail_cpp, f_I_cpp, T_I_cpp, goal_reached_cpp = mrlsp_accel.move_robots_cpp(
        s_1b_succ_cpp, action2a_cpp)
    assert T_I_py == T_I_cpp
    assert T_I_py == 5
    assert f_I_cpp == hash(f_I_py)
    assert f_I_py == a2
    assert goal_reached_py == goal_reached_cpp
    test_shared_functions.check_py_and_cpp_states_match(s_2a_succ_py, s_2a_succ_cpp, s_dict)
    test_shared_functions.check_py_and_cpp_states_match(s_2a_fail_py, s_2a_fail_cpp, s_dict)

    actions_s2 = [(a2, a2)]
    print("Child 1b (fail) -> Child 2a")
    action2b_py = actions_s2[0]
    action2b_cpp = (hash(action2b_py[0]), hash(action2b_py[1]))
    s_2b_succ_py, s_2b_fail_py, f_I_py, T_I_py, goal_reached_py = mrlsp.core.move_robots(s_1b_fail_py, action2b_py)
    s_2b_succ_cpp, s_2b_fail_cpp, f_I_cpp, T_I_cpp, goal_reached_cpp = mrlsp_accel.move_robots_cpp(
        s_1b_fail_cpp, action2b_cpp)
    assert T_I_py == T_I_cpp
    assert T_I_py == 5
    assert f_I_cpp == hash(f_I_py)
    assert f_I_py == a2
    assert goal_reached_py == goal_reached_cpp
    test_shared_functions.check_py_and_cpp_states_match(s_2b_succ_py, s_2b_succ_cpp, s_dict)
    test_shared_functions.check_py_and_cpp_states_match(s_2b_fail_py, s_2b_fail_cpp, s_dict)
