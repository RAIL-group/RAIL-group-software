import mrlsp_accel
from util_function import current_state_1  # noqa: F401


def test_state_variables(current_state_1):  # noqa: F811
    '''Only current_state_1 has data to test this function'''
    num_robots = current_state_1['num_robots']
    unexplored_frontiers = current_state_1['unexplored_frontiers_cpp']
    s_cpp = current_state_1['s_cpp']
    rd_cpp = current_state_1['rd_cpp']
    gd_cpp = current_state_1['gd_cpp']
    fd_cpp = current_state_1['fd_cpp']

    a1 = unexplored_frontiers[0]

    s1_cpp = mrlsp_accel.State_cpp(num_robots=num_robots, unexplored_frontiers=s_cpp, robot_distances=rd_cpp,
                                   goal_distances=gd_cpp, frontier_distances=fd_cpp)

    # check that the state is properly initialized
    assert s1_cpp.num_robots == num_robots
    assert s1_cpp.unexplored_frontiers == s_cpp
    assert s1_cpp.robot_distances == rd_cpp

    s2_cpp = s1_cpp.copy_state()
    '''
    In the copy state, the unexplored_frontiers and the robot distances are copied by value,
    but the goal distances and frontier distances are copied by reference.'''

    assert s2_cpp.num_robots == num_robots
    s2_cpp.remove_frontier_from_unexplored_frontiers(hash(a1))
    s2_cpp.add_frontier_to_goal_frontiers(hash(a1))
    # The number of frontiers in s2_cpp is decreased
    assert len(s2_cpp.unexplored_frontiers) == 2
    # The number of frontiers in s1_cpp is not changed
    assert len(s1_cpp.unexplored_frontiers) == 3
    # The number of goal frontiers in s2_cpp is increased
    assert len(s2_cpp.goal_frontiers) == 1
    # The number of goal frontiers in s1_cpp is not changed
    assert len(s1_cpp.goal_frontiers) == 0

    # change the robot distance in s2_cpp and see if it changes in s1_cpp
    s2_cpp.change_rd_for_robot(0, hash(a1), 100)

    print(f'{s1_cpp.unexplored_frontiers=}, \n {s2_cpp.unexplored_frontiers=}')
    print(f'{s1_cpp.goal_frontiers=}, \n {s2_cpp.goal_frontiers=}')
    print(f'{s1_cpp.robot_distances=}, \n {s2_cpp.robot_distances=}')

    assert s2_cpp.robot_distances[(0, hash(a1))] != s1_cpp.robot_distances[(0, hash(a1))]
