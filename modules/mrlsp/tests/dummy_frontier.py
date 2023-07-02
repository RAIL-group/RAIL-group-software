import numpy as np


class DummyFrontier():
    def __init__(self, q_f, label=None):
        self.prob_feasible = 0.0
        self.exploration_cost = 0.0
        self.delta_success_cost = 0.0
        self.q_f = np.array(q_f)
        self.label = label
        self.is_from_last_chosen = False
        self.hash = hash(self.q_f.tobytes())

    def set_props(self, prob_feasible, exploration_cost, delta_success_cost):
        self.prob_feasible = prob_feasible
        self.exploration_cost = exploration_cost
        self.delta_success_cost = delta_success_cost

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)
