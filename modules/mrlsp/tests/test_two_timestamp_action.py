from dummy_frontier import DummyFrontier
from mrlsp import pouct


def test_two_timestamp():
    a1 = DummyFrontier(q_f=[126.28, 71.78], label='a1')
    a2 = DummyFrontier(q_f=[57.16, 41.66], label='a2')
    a3 = DummyFrontier(q_f=[20.21, 103.71], label='a3')
    a4 = DummyFrontier(q_f=[93.28, 38.78], label='a4')

    a1.set_props(
        prob_feasible=0.9704260230064392,
        delta_success_cost=18.546121597,
        exploration_cost=394.71551513671875
    )
    a2.set_props(
        prob_feasible=0.0044766939245164394,
        delta_success_cost=46.356998444,
        exploration_cost=95.10801696777344,
    )
    a3.set_props(
        prob_feasible=0.01118100993335247,
        delta_success_cost=47.328475952,
        exploration_cost=108.231201171875,
    )
    a4.set_props(
        prob_feasible=0.22330205142498016,
        delta_success_cost=29.703948975,
        exploration_cost=97.69618225097656,
    )

    timestep_24 = {
        'frontier': {frozenset([a1, a2]): 77.0121933088197, frozenset([a1, a3]): 124.28427124746185,
                     frozenset([a1, a4]): 87.08326112068515,
                     frozenset([a2, a3]): 67.65685424949237, frozenset([a2, a4]): 33.38477631085024,
                     frozenset([a3, a4]): 42.31370849898474},
        'robot1': {a1: 30.62741699796953, a2: 66.76955262170043, a3: 111.0416305603426, a4: 72.42640687119281},
        'robot2': {a1: 72.18376618407353, a2: 29.213203435596434, a3: 67.82842712474618, a4: 29.213203435596434},
        'goal': {a1: 30.69848480983501, a2: 128.3847763108502, a3: 162.25483399593924, a4: 127.38477631085023}
    }

    timestep_25 = {
        'frontier': {frozenset([a1, a2]): 77.0121933088197, frozenset([a1, a3]): 124.28427124746185,
                     frozenset([a1, a4]): 87.08326112068515,
                     frozenset([a2, a3]): 67.65685424949237, frozenset([a2, a4]): 33.38477631085024,
                     frozenset([a3, a4]): 42.31370849898474},
        'robot1': {a1: 30.213203435596434, a2: 66.35533905932733, a3: 110.6274169979695, a4: 72.0121933088197},
        'robot2': {a1: 72.76955262170043, a2: 27.798989873223338, a3: 67.24264068711928, a4: 28.62711699796953},
        'goal': {a1: 30.69848480983501, a2: 128.3847763108502, a3: 162.25483399593924, a4: 127.38477631085023}
    }

    unexplored_frontiers = {a1, a2, a3, a4}
    best_action_t24 = pouct.find_best_joint_action(unexplored_frontiers=unexplored_frontiers,
                                                   time=timestep_24, num_iterations=1000)
    best_action_t25 = pouct.find_best_joint_action(unexplored_frontiers=unexplored_frontiers,
                                                   time=timestep_25, num_iterations=1000)

    assert best_action_t24[0] == a1
    assert best_action_t25[0] == a1
