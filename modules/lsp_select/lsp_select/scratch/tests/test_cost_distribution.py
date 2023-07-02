import numpy as np
import lsp
import lsp_select


def test_cost_distribution():
    f1 = lsp.core.Frontier(points=np.array([[1, 2, 3], [2, 3, 4]]))
    f1.set_props(prob_feasible=0.8,
                 delta_success_cost=11,
                 exploration_cost=33,
                 delta_success_cost_std=2,
                 exploration_cost_std=3)
    f2 = lsp.core.Frontier(points=np.array([[3, 4, 5], [5, 6, 7]]))
    f2.set_props(prob_feasible=0.2,
                 delta_success_cost=14,
                 exploration_cost=26,
                 delta_success_cost_std=4,
                 exploration_cost_std=5)
    f3 = lsp.core.Frontier(points=np.array([[6, 7, 8], [7, 8, 9]]))
    f3.set_props(prob_feasible=0.3,
                 delta_success_cost=15,
                 exploration_cost=27,
                 delta_success_cost_std=6,
                 exploration_cost_std=7)
    distances = {
        'frontier': {frozenset([f1, f2]): 17, frozenset([f1, f3]): 12, frozenset([f2, f3]): 10},
        'robot': {f1: 5, f2: 8, f3: 6},
        'goal': {f1: 21, f2: 23, f3: 25}
    }
    probs, costs = lsp_select.utils.distribution.get_cost_distribution([f1, f2, f3], distances)

    actual_probs = [f1.prob_feasible,
                    (1 - f1.prob_feasible) * f2.prob_feasible,
                    (1 - f1.prob_feasible) * (1 - f2.prob_feasible) * f3.prob_feasible,
                    (1 - f1.prob_feasible) * (1 - f2.prob_feasible) * (1 - f3.prob_feasible)]
    assert np.allclose(probs, actual_probs)

    actual_costs_mean = [distances['robot'][f1] + distances['goal'][f1] + f1.delta_success_cost,
                         distances['robot'][f1] + f1.exploration_cost + distances['frontier'][frozenset([f1, f2])]
                         + distances['goal'][f2] + f2.delta_success_cost,
                         distances['robot'][f1] + f1.exploration_cost + distances['frontier'][frozenset([f1, f2])]
                         + f2.exploration_cost + distances['frontier'][frozenset([f2, f3])]
                         + distances['goal'][f3] + f3.delta_success_cost,
                         distances['robot'][f1] + f1.exploration_cost + distances['frontier'][frozenset([f1, f2])]
                         + f2.exploration_cost + distances['frontier'][frozenset([f2, f3])]
                         + f3.exploration_cost]
    assert np.allclose(costs[:, 0], actual_costs_mean)

    actual_costs_std = [f1.delta_success_cost_std,
                        (f1.exploration_cost_std**2 + f2.delta_success_cost_std**2)**0.5,
                        (f1.exploration_cost_std**2 + f2.exploration_cost_std**2 + f3.delta_success_cost_std**2)**0.5,
                        (f1.exploration_cost_std**2 + f2.exploration_cost_std**2 + f3.exploration_cost_std**2)**0.5]
    assert np.allclose(costs[:, 1], actual_costs_std)
