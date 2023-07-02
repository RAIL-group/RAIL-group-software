from util_function import current_state_1, current_state_2, current_state_3, cpp_state_bug1   # noqa: F401
import mrlsp
import mrlsp_accel
import time
import pytest


@pytest.mark.parametrize(
    "current_state_name, num_robots",
    [
        ("cpp_state_bug1", 1),
        ("cpp_state_bug1", 2),
        ("current_state_1", 1),
        ("current_state_1", 2),
        ("current_state_2", 1),
        ("current_state_2", 2),
        ("current_state_3", 1),
        ("current_state_3", 2),
        ("current_state_3", 3),
    ])
def test_mrlsp_works_with_multiple_robots(current_state_name, num_robots, request):  # noqa: F811
    '''NOTE: cpp_state_bug1, current_state_1 & 2 doesn't have 3 robot support because of robot distances
    and frontier distances '''
    current_state = request.getfixturevalue(current_state_name)
    num_robots = num_robots

    has_python_data = current_state['has_data_for_python']
    has_cpp_data = current_state['has_data_for_cpp']

    if has_python_data:
        unexplored_frontiers = current_state['unexplored_frontiers_py']
        times = current_state['time']
        start_time = time.time()
        best_action_py = mrlsp.pouct.find_best_joint_action(
            unexplored_frontiers=unexplored_frontiers, time=times, num_robots=num_robots, num_iterations=5000)
        print(f"Time taken PY: {time.time() - start_time}")
        print(f'Joint action py: {[hash(b) for b in best_action_py]}')

    if has_cpp_data:
        # s_dict = current_state['s_dict']
        s_cpp = current_state['s_cpp']
        rd_cpp = current_state['rd_cpp']
        gd_cpp = current_state['gd_cpp']
        fd_cpp = current_state['fd_cpp']
        start_time = time.time()
        best_action_cpp = mrlsp_accel.find_best_joint_action_accel(num_robots, s_cpp, rd_cpp, gd_cpp, fd_cpp, 5000)
        print(f"Time taken CPP: {time.time() - start_time}")
        print(f'Joint action cpp: {best_action_cpp}')
