import mrlsp
import mrlsp_accel
import pytest
import itertools
from dummy_frontier import DummyFrontier


@pytest.fixture
def current_state_1():
    num_robots = 2
    a1 = DummyFrontier(q_f=[305, 409], label='a1')
    a2 = DummyFrontier(q_f=[105, 209], label='a2')
    a3 = DummyFrontier(q_f=[505, 609], label='a3')

    a1.set_props(
        prob_feasible=0.9,
        delta_success_cost=100,
        exploration_cost=5
    )
    a2.set_props(
        prob_feasible=0.3,
        delta_success_cost=20,
        exploration_cost=10,
    )
    a3.set_props(
        prob_feasible=0.7,
        delta_success_cost=30,
        exploration_cost=3,
    )

    new_time = {
        'frontier': {frozenset([a1, a2]): 10, frozenset([a2, a3]): 5, frozenset([a1, a3]): 5},
        'robot1': {a1: 8, a2: 12, a3: 15},
        'robot2': {a1: 12, a2: 8, a3: 15},
        'goal': {a1: 0, a2: 0, a3: 0}
    }
    unexplored_frontiers = {a1, a2, a3}
    unexplored_frontiers_cpp = [a1, a2, a3]
    q_t = [{a1: 10, a2: 0, a3: 0}, {a1: 0.0, a2: 12, a3: 0}]

    # for cpp
    s_dict = {hash(s): s for s in unexplored_frontiers}
    s_cpp = [
        mrlsp_accel.FrontierDataMR(s.prob_feasible, s.delta_success_cost,
                                   s.exploration_cost, hash(s),
                                   s.is_from_last_chosen) for s in unexplored_frontiers
    ]
    rd_cpp = {(i, hash(s)): new_time[f'robot{i+1}'][s] for i in range(num_robots) for s in unexplored_frontiers}
    gd_cpp = {hash(s): new_time['goal'][s] for s in unexplored_frontiers}
    fd_cpp = {(hash(sp[0]), hash(sp[1])): new_time['frontier'][frozenset(sp)]
              for sp in itertools.permutations(unexplored_frontiers, 2)}
    q_t_cpp = [{hash(a1): 10.0, hash(a2): 0.0, hash(a3): 0.0}, {hash(a1): 0.0, hash(a2): 12, hash(a3): 0.0}]

    # return [[num_robots, unexplored_frontiers, new_time, q_t],
    #         [s_cpp, rd_cpp, gd_cpp, fd_cpp, q_t_cpp],
    #         [s_dict]]

    state_parameters = {'num_robots': num_robots,
                        'unexplored_frontiers_py': unexplored_frontiers,
                        'unexplored_frontiers_cpp': unexplored_frontiers_cpp,
                        'time': new_time,
                        's_cpp': s_cpp,
                        'rd_cpp': rd_cpp,
                        'gd_cpp': gd_cpp,
                        'fd_cpp': fd_cpp,
                        's_dict': s_dict,
                        'q_t': q_t,
                        'q_t_cpp': q_t_cpp,
                        'has_data_for_python': True,
                        'has_data_for_cpp': True,
                        }

    return state_parameters


@pytest.fixture
def current_state_2():
    '''Only in current state 2, we have hand calculated values.'''
    num_robots = 2
    a1 = DummyFrontier(q_f=[305, 409], label='a1')
    a2 = DummyFrontier(q_f=[105, 209], label='a2')
    a3 = DummyFrontier(q_f=[505, 609], label='a3')

    a1.set_props(
        prob_feasible=0.9,
        delta_success_cost=100,
        exploration_cost=5
    )
    a2.set_props(
        prob_feasible=0.3,
        delta_success_cost=20,
        exploration_cost=10,
    )
    a3.set_props(
        prob_feasible=0.7,
        delta_success_cost=30,
        exploration_cost=3,
    )

    new_time = {
        'frontier': {frozenset([a1, a2]): 10},
        'robot1': {a1: 8, a2: 12},
        'robot2': {a1: 12, a2: 8},
        'goal': {a1: 0, a2: 0}
    }
    calculated_actions = [(a1, a2),
                          (a2, a1)]

    calculated_costs = [78.4, 82.4]
    unexplored_frontiers = {a1, a2}

    unexplored_frontiers_cpp = [a1, a2]
    s_dict = {hash(s): s for s in unexplored_frontiers_cpp}
    s_cpp = [
        mrlsp_accel.FrontierDataMR(s.prob_feasible, s.delta_success_cost,
                                   s.exploration_cost, hash(s),
                                   s.is_from_last_chosen) for s in unexplored_frontiers_cpp
    ]
    rd_cpp = {(i, hash(s)): new_time[f'robot{i+1}'][s] for i in range(num_robots) for s in unexplored_frontiers_cpp}
    gd_cpp = {hash(s): new_time['goal'][s] for s in unexplored_frontiers_cpp}
    fd_cpp = {(hash(sp[0]), hash(sp[1])): new_time['frontier'][frozenset(sp)]
              for sp in itertools.permutations(unexplored_frontiers_cpp, 2)}

    # return [
    #     [unexplored_frontiers, new_time, calculated_actions, calculated_costs, num_robots, unexplored_frontiers_cpp],
    #     [s_dict, s_cpp, rd_cpp, gd_cpp, fd_cpp]
    # ]

    state_parameters = {'num_robots': num_robots,
                        'unexplored_frontiers_py': unexplored_frontiers,
                        'unexplored_frontiers_cpp': unexplored_frontiers_cpp,
                        'time': new_time,
                        's_cpp': s_cpp,
                        'rd_cpp': rd_cpp,
                        'gd_cpp': gd_cpp,
                        'fd_cpp': fd_cpp,
                        's_dict': s_dict,
                        'calculated_actions': calculated_actions,
                        'calculated_costs': calculated_costs,
                        'has_data_for_python': True,
                        'has_data_for_cpp': True}

    return state_parameters


@pytest.fixture
def current_state_3():
    num_robots = 3
    a1 = DummyFrontier(q_f=[305, 409], label='a1')
    a2 = DummyFrontier(q_f=[105, 209], label='a2')
    a3 = DummyFrontier(q_f=[505, 609], label='a3')
    a4 = DummyFrontier(q_f=[705, 809], label='a4')
    a5 = DummyFrontier(q_f=[905, 1009], label='a5')

    a1.set_props(
        prob_feasible=0.9,
        delta_success_cost=100,
        exploration_cost=5
    )
    a2.set_props(
        prob_feasible=0.3,
        delta_success_cost=20,
        exploration_cost=10,
    )
    a3.set_props(
        prob_feasible=0.7,
        delta_success_cost=30,
        exploration_cost=15,
    )
    a4.set_props(
        prob_feasible=0.5,
        delta_success_cost=50,
        exploration_cost=12,
    )
    a5.set_props(
        prob_feasible=0.1,
        delta_success_cost=60,
        exploration_cost=20,
    )

    new_time = {
        'frontier': {frozenset([a1, a2]): 10, frozenset([a1, a3]): 20, frozenset([a1, a4]): 40, frozenset([a1, a5]): 60,
                     frozenset([a2, a3]): 25, frozenset([a2, a4]): 30, frozenset([a2, a5]): 37,
                     frozenset([a3, a4]): 33, frozenset([a3, a5]): 60,
                     frozenset([a4, a5]): 27},
        'robot1': {a1: 8, a2: 12, a3: 20, a4: 40, a5: 60},
        'robot2': {a1: 12, a2: 8, a3: 20, a4: 40, a5: 60},
        'robot3': {a1: 20, a2: 20, a3: 8, a4: 40, a5: 60},
        'goal': {a1: 0, a2: 0, a3: 0, a4: 0, a5: 0}
    }
    unexplored_frontiers = {a1, a2, a3, a4, a5}

    unexplored_frontiers_cpp = [a1, a2, a3, a4, a5]
    s_dict = {hash(s): s for s in unexplored_frontiers_cpp}
    s_cpp = [
        mrlsp_accel.FrontierDataMR(s.prob_feasible, s.delta_success_cost,
                                   s.exploration_cost, hash(s),
                                   s.is_from_last_chosen) for s in unexplored_frontiers_cpp
    ]
    rd_cpp = {(i, hash(s)): new_time[f'robot{i+1}'][s] for i in range(num_robots) for s in unexplored_frontiers_cpp}
    gd_cpp = {hash(s): new_time['goal'][s] for s in unexplored_frontiers_cpp}
    fd_cpp = {(hash(sp[0]), hash(sp[1])): new_time['frontier'][frozenset(sp)]
              for sp in itertools.permutations(unexplored_frontiers_cpp, 2)}

    # return [[unexplored_frontiers, new_time, num_robots, unexplored_frontiers_cpp],
    #         [s_dict, s_cpp, rd_cpp, gd_cpp, fd_cpp]]

    state_parameters = {'num_robots': num_robots,
                        'unexplored_frontiers_py': unexplored_frontiers,
                        'unexplored_frontiers_cpp': unexplored_frontiers_cpp,
                        'time': new_time,
                        's_cpp': s_cpp,
                        'rd_cpp': rd_cpp,
                        'gd_cpp': gd_cpp,
                        'fd_cpp': fd_cpp,
                        's_dict': s_dict,
                        'has_data_for_python': True,
                        'has_data_for_cpp': True}

    return state_parameters


@pytest.fixture
def cpp_state_bug1():
    '''This exact data (except for frontier hash number) resulted in segmentation fault in mrlsp_accel.'''
    frontier1_hash = -1111111111111111
    frontier2_hash = -2222222222222222
    frontier3_hash = -3333333333333333
    frontier4_hash = -4444444444444444
    frontier5_hash = -5555555555555555
    frontier6_hash = -6666666666666666
    frontier7_hash = -7777777777777777

    frontier1 = mrlsp_accel.FrontierDataMR(0.7924796342849731, 169.36832580710956,
                                           188.08245849609375, frontier1_hash, False)
    frontier2 = mrlsp_accel.FrontierDataMR(0.15180489420890808, 301.71375278452655,
                                           106.37687683105469, frontier2_hash, False)
    frontier3 = mrlsp_accel.FrontierDataMR(0.5383244752883911, 169.77446062963244,
                                           58.89070510864258, frontier3_hash, False)
    frontier4 = mrlsp_accel.FrontierDataMR(0.48599353432655334, 207.92903140640095,
                                           96.38992309570312, frontier4_hash, False)
    frontier5 = mrlsp_accel.FrontierDataMR(0.604091227054596, 135.59115753860436,
                                           83.71237182617188, frontier5_hash, False)
    frontier6 = mrlsp_accel.FrontierDataMR(0.06415274739265442, 329.7541229349233,
                                           83.29269409179688, frontier6_hash, False)
    frontier7 = mrlsp_accel.FrontierDataMR(0.779825747013092, 237.1676590301899,
                                           271.9759521484375, frontier7_hash, False)

    unexplored_frontiers = [frontier1, frontier2, frontier3, frontier4, frontier5, frontier6, frontier7]
    robot_distances = {
        (0, frontier1_hash): 57.07106781186547, (0, frontier2_hash): 27.72792206135786,
        (0, frontier3_hash): 41.870057685088796, (0, frontier4_hash): 47.97056274847713,
        (0, frontier5_hash): 55.79898987322331, (0, frontier6_hash): 12.242640687119286,
        (0, frontier7_hash): 55.556349186104036, (1, frontier1_hash): 51.727922061357845,
        (1, frontier2_hash): 51.4558441227157, (1, frontier3_hash): 49.89949493661165,
        (1, frontier4_hash): 62.62741699796948, (1, frontier5_hash): 18.071067811865476,
        (1, frontier6_hash): 50.07106781186546, (1, frontier7_hash): 19.48528137423857
    }

    goal_distances = {
        frontier1_hash: 95.65685424949237,
        frontier2_hash: 174.49242404917499,
        frontier3_hash: 121.89444430272815,
        frontier4_hash: 149.35028842544392,
        frontier5_hash: 83.87005768508875,
        frontier6_hash: 173.59797974644675,
        frontier7_hash: 135.5218613006977
    }

    frontier_distances = {
        (frontier1_hash, frontier2_hash): 61.31370849898474, (frontier2_hash, frontier1_hash): 61.31370849898474,
        (frontier1_hash, frontier3_hash): 61.828427124746185, (frontier3_hash, frontier1_hash): 61.828427124746185,
        (frontier1_hash, frontier4_hash): 68.82842712474618, (frontier4_hash, frontier1_hash): 68.82842712474618,
        (frontier1_hash, frontier5_hash): 49.55634918610403, (frontier5_hash, frontier1_hash): 49.55634918610403,
        (frontier1_hash, frontier6_hash): 59.65685424949237, (frontier6_hash, frontier1_hash): 59.65685424949237,
        (frontier1_hash, frontier7_hash): 79.41421356237309, (frontier7_hash, frontier1_hash): 79.41421356237309,
        (frontier2_hash, frontier3_hash): 45.798989873223306, (frontier3_hash, frontier2_hash): 45.798989873223306,
        (frontier2_hash, frontier4_hash): 48.48528137423857, (frontier4_hash, frontier2_hash): 48.48528137423857,
        (frontier2_hash, frontier5_hash): 61.183766184073534, (frontier5_hash, frontier2_hash): 61.183766184073534,
        (frontier2_hash, frontier6_hash): 35.89949493661167, (frontier6_hash, frontier2_hash): 35.89949493661167,
        (frontier2_hash, frontier7_hash): 61.071067811865476, (frontier7_hash, frontier2_hash): 61.071067811865476,
        (frontier3_hash, frontier4_hash): 16.48528137423857, (frontier4_hash, frontier3_hash): 16.48528137423857,
        (frontier3_hash, frontier5_hash): 59.3847763108502, (frontier5_hash, frontier3_hash): 59.3847763108502,
        (frontier3_hash, frontier6_hash): 49.21320343559642, (frontier6_hash, frontier3_hash): 49.21320343559642,
        (frontier3_hash, frontier7_hash): 20.242640687119284, (frontier7_hash, frontier3_hash): 20.242640687119284,
        (frontier4_hash, frontier5_hash): 68.69848480983494, (frontier5_hash, frontier4_hash): 68.69848480983494,
        (frontier4_hash, frontier6_hash): 51.89949493661165, (frontier6_hash, frontier4_hash): 51.89949493661165,
        (frontier4_hash, frontier7_hash): 13.82842712474619, (frontier7_hash, frontier4_hash): 13.82842712474619,
        (frontier5_hash, frontier6_hash): 60.69848480983499, (frontier6_hash, frontier5_hash): 60.69848480983499,
        (frontier5_hash, frontier7_hash): 76.97056274847715, (frontier7_hash, frontier5_hash): 76.97056274847715,
        (frontier6_hash, frontier7_hash): 64.48528137423858, (frontier7_hash, frontier6_hash): 64.48528137423858,
    }

    num_robots = 2
    num_iterations = 5000

    # return [num_robots, unexplored_frontiers, robot_distances, goal_distances, frontier_distances, num_iterations]

    state_parameters = {'num_robots': num_robots,
                        's_dict': None,
                        's_cpp': unexplored_frontiers,
                        'rd_cpp': robot_distances,
                        'gd_cpp': goal_distances,
                        'fd_cpp': frontier_distances,
                        'num_iterations': num_iterations,
                        'has_data_for_python': False,
                        'has_data_for_cpp': True}

    return state_parameters


def check_py_and_cpp_states_match(s1_py, s1_cpp, s_dict):
    # Print everything for cpp state
    print("####################################################")
    print(s_dict)
    print("CPP Version")
    print(f'Num robots: {s1_cpp.num_robots}')
    print(f'Unexplored frontiers: {[s for s in s1_cpp.unexplored_frontiers_hash]}')
    print(f'Robot distances: {s1_cpp.robot_distances}')

    print(f'goal_frontiers: {s1_cpp.goal_frontiers}')
    print(f'Q_t: {s1_cpp.q_t}')
    print("---------------------------------------------------")
    print("Python Version")
    print(f'Num robots: {s1_py.n}')
    print(f'Unexplored frontiers: {[s for s in s1_py.Fu]}')
    for i in range(s1_py.n):
        print(f"Robot{i+1} distance:", s1_py.time[f'robot{i+1}'])

    print(f"goal_frontiers: {s1_py.goal_frontiers}")
    print(f'Q_t: {s1_py.q_t}')
    print("####################################################")

    # checks everything of the state
    assert s1_py.n == s1_cpp.num_robots
    assert len(s1_py.Fu) == len(s1_cpp.unexplored_frontiers)

    # check if every frontier in unexplored frontier of python is in cpp
    assert len(s1_py.Fu) == len(s1_cpp.unexplored_frontiers_hash)
    for f in s1_py.Fu:
        assert hash(f) in s1_cpp.unexplored_frontiers_hash

    # check if all the goal frontier of python is in cpp
    assert len(s1_py.goal_frontiers) == len(s1_cpp.goal_frontiers)
    for f in s1_py.goal_frontiers:
        assert hash(f) in s1_cpp.goal_frontiers

    # check robot distance
    for pair, distance in s1_cpp.robot_distances.items():
        robot = pair[0]
        frontier_hash = pair[1]
        assert pytest.approx(s1_py.time[f'robot{robot+1}'][s_dict[frontier_hash]]) == distance

    # check q_t
    for i in range(s1_py.n):
        for frontier_hash, q in s1_cpp.q_t[i].items():
            assert s1_py.q_t[i][s_dict[frontier_hash]] == q

    # check if the actions are same
    actions_cpp = s1_cpp.get_actions()
    actions_py = s1_py.get_actions()
    assert len(actions_cpp) == len(actions_py)


def check_py_and_cpp_q_t_match(s1_py, s1_cpp, action, s_dict):
    print("--------------------------------------------------")
    print(f"For action: ({[hash(a) for a in action]})")
    action1_py = action
    action1_cpp = [hash(a) for a in action]
    f_I_py, T_I_py = mrlsp.core.get_frontier_of_knowlege_and_time(s1_py, action1_py)
    f_I_cpp, T_I_cpp = mrlsp_accel.get_frontier_of_knowledge_and_time_cpp(s1_cpp, action1_cpp)
    assert f_I_cpp == hash(f_I_py)
    assert T_I_cpp == T_I_py
    q_t_py, residue_py = s1_py.find_q_t_for_action(T_I_py, action1_py)
    q_t_cpp, residue_cpp = mrlsp_accel.find_q_t_for_action_cpp(s1_cpp, T_I_cpp, action1_cpp)

    # Test that q_t is same for python and cpp State classes
    for prog_py, prog_cpp in zip(q_t_py, q_t_cpp):
        for f_py, p_py in prog_py.items():
            assert prog_cpp[hash(f_py)] == p_py
    q_t_py_hash = []
    for i, q_t in enumerate(q_t_py):
        q_t_dict = {hash(f): p for f, p in q_t.items()}
        q_t_py_hash.append(q_t_dict)
    print(f'q_t_py = {q_t_py_hash}')
    print(f'{q_t_cpp=}')
    print(f'residue_py = {residue_py}, residue_cpp = {residue_cpp}')

    time_py = s1_py.get_time_from_q_t(q_t_py, residue_py)
    distance_cpp = mrlsp_accel.get_time_from_qt_cpp(s1_cpp, q_t_cpp, residue_cpp)
    print(f'{time_py=}')
    print(f'{distance_cpp=}')
    for robot_frontier_pair, distance in distance_cpp.items():
        robot = robot_frontier_pair[0]
        frontier_hash = robot_frontier_pair[1]
        assert pytest.approx(time_py[f'robot{robot+1}'][s_dict[frontier_hash]]) == distance


def check_py_and_cpp_move_robots_match(s1_py, s1_cpp, action, s_dict):
    print(f"\nFor action {[hash(a) for a in action]}")
    action_py = action
    action_cpp = [hash(a) for a in action]
    success_state_py, failure_state_py, f_I_py, T_I_py, goal_reached_py = mrlsp.core.move_robots(s1_py, action_py)
    success_state_cpp, failure_state_cpp, f_I_cpp, T_I_cpp, goal_reached_cpp = mrlsp_accel.move_robots_cpp(
        s1_cpp, action_cpp)

    assert f_I_cpp == hash(f_I_py)
    assert T_I_cpp == T_I_py
    assert goal_reached_cpp == goal_reached_py
    assert len(success_state_cpp.unexplored_frontiers) == len(success_state_py.Fu)
    assert len(failure_state_cpp.unexplored_frontiers) == len(failure_state_py.Fu)
    assert len(success_state_cpp.goal_frontiers) == len(success_state_py.goal_frontiers)
    assert len(failure_state_cpp.goal_frontiers) == len(failure_state_py.goal_frontiers)
    assert len(success_state_cpp.unexplored_frontiers_hash) == len(success_state_py.Fu)
    assert len(failure_state_cpp.unexplored_frontiers) == len(failure_state_py.Fu)
    for f in success_state_py.Fu:
        assert hash(f) in success_state_cpp.unexplored_frontiers_hash
    for f in failure_state_py.Fu:
        assert hash(f) in failure_state_cpp.unexplored_frontiers_hash
    for f in success_state_py.goal_frontiers:
        assert hash(f) in success_state_cpp.goal_frontiers

    # check robot distances in success_state
    print("Success state distances:")
    for pair, distance in success_state_cpp.robot_distances.items():
        robot = pair[0]
        frontier_hash = pair[1]
        assert pytest.approx(success_state_py.time[f'robot{robot+1}'][s_dict[frontier_hash]]) == distance
        print(f"Robot {robot+1}, frontier {frontier_hash}: Distance = {distance}")

    # check robot distances in failure_state
    print("Failure state distances:")
    for pair, distance in failure_state_cpp.robot_distances.items():
        robot = pair[0]
        frontier_hash = pair[1]
        assert pytest.approx(failure_state_py.time[f'robot{robot+1}'][s_dict[frontier_hash]]) == distance
        print(f"Robot {robot+1}, frontier {frontier_hash}: Distance = {distance}")

    # check q_t for cpp and python for success state
    success_q_t_py = success_state_py.q_t
    success_q_t_cpp = success_state_cpp.q_t
    for s_q_t_py, s_q_t_cpp in zip(success_q_t_py, success_q_t_cpp):
        for f_py, p_py in s_q_t_py.items():
            assert s_q_t_cpp[hash(f_py)] == p_py

    # check q_t for cpp and python for failure state
    failure_q_t_py = failure_state_py.q_t
    failure_q_t_cpp = failure_state_cpp.q_t
    for f_q_t_py, f_q_t_cpp in zip(failure_q_t_py, failure_q_t_cpp):
        for f_py, p_py in f_q_t_py.items():
            assert f_q_t_cpp[hash(f_py)] == p_py
