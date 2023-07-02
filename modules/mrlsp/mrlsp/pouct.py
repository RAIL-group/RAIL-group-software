import numpy as np
from scipy.stats import bernoulli
import copy
from mrlsp.core import move_robots, State
from mrlsp.utils.utility import find_action_index


# just some prime numers so it wont get repeated
NULL = 573213
FAILURE = 0
SUCCESS = 1


class Node():
    def __init__(self, start_state, parent=None, prev_action=None, T_I=0, goal_reached=False):
        self.state = start_state
        self.parent = parent
        self.prev_action = prev_action
        if parent is not None:
            self.cost = T_I + parent.cost
        else:
            self.cost = T_I
        self.goal_reached = goal_reached

        self.actions = self.state.get_actions()
        self.action_values = [[] for _ in range(len(self.actions))]
        self.action_n = [0 for _ in range(len(self.actions))]
        self.unexplored_actions = copy.copy(self.actions)
        '''One node can have two children, either f_I leads to the goal or not
        the 1th index is for success, and 0th index is for failure'''
        self.children = [{0: NULL, 1: NULL} for _ in range(len(self.actions))]
        '''One node can have two children, either f_I leads to the goal or not
        the 1th index is for success, and 0th index is for failure'''
        self.children = [{FAILURE: NULL, SUCCESS: NULL} for _ in range(len(self.actions))]

        '''This stores the outcome of node so that the computation is saved'''
        self.children_properties = [{'outcome': {FAILURE: NULL, SUCCESS: NULL},
                                    'properties': {'f_I': None, 'T_I': None, 'goal_reached': None}, 'has_value': False}
                                    for _ in range(len(self.actions))]

        # Update if this is terminal node
        if not self.goal_reached:
            state_unexplored_frontiers = list(self.state.Fu)
            if len(state_unexplored_frontiers) == 0:
                # Updated this
                if len(self.state.goal_frontiers) != 0:
                    all_actions = self.state.get_actions()
                    all_action_cost = []
                    for a in all_actions:
                        all_robot_cost = []
                        for i, f in enumerate(a):
                            cost_to_goal = self.state.time[f'robot{i+1}'][f] + (
                                f.delta_success_cost + self.state.time['goal'][f]) - self.state.q_t[i][f]
                            all_robot_cost.append(cost_to_goal)
                        all_action_cost.append(min(all_robot_cost))
                    self.cost += min(all_action_cost)
                    self.goal_reached = True
                else:
                    '''When no frontier leads to the goal, and no unexplored frontiers left,
                    assume that the parent node unexplored frontier leads to the goal, and add
                    the success cost of that frontier.'''
                    f = list(self.parent.state.Fu)[0]
                    all_robot_cost = []
                    for i in range(self.state.n):
                        cost_to_goal = self.parent.state.time[f'robot{i+1}'][f] + (
                            f.delta_success_cost + self.state.time['goal'][f]) - self.state.q_t[i][f]
                        all_robot_cost.append(cost_to_goal)
                    self.cost += min(all_robot_cost)
                    self.goal_reached = True

    def update_child_properties(self, action_idx, success_state, failure_state, f_I, T_I, goal_reached):
        self.children_properties[action_idx]['has_value'] = True
        self.children_properties[action_idx]['outcome'][SUCCESS] = success_state
        self.children_properties[action_idx]['outcome'][FAILURE] = failure_state
        self.children_properties[action_idx]['properties']['f_I'] = f_I
        self.children_properties[action_idx]['properties']['T_I'] = T_I
        self.children_properties[action_idx]['properties']['goal_reached'] = goal_reached

    def get_child_properties(self, action_idx):
        if self.children_properties[action_idx]['has_value'] is True:
            success_state = self.children_properties[action_idx]['outcome'][SUCCESS]
            failure_state = self.children_properties[action_idx]['outcome'][FAILURE]
            f_I = self.children_properties[action_idx]['properties']['f_I']
            T_I = self.children_properties[action_idx]['properties']['T_I']
            goal_reached = self.children_properties[action_idx]['properties']['goal_reached']
            return success_state, failure_state, f_I, T_I, goal_reached
        else:
            return None

    @property
    def is_fully_explored(self):
        return len(self.unexplored_actions) == 0

    @property
    def is_terminal_state(self):
        return self.goal_reached

    def find_rollout_cost(self):
        '''Rollout cost from 'self' state'''
        if self.goal_reached:
            return self.cost
        # Use a heuristic to find the best policy and calculate the cost of that policy randomly
        cost_for_robots = []
        for i in range(self.state.n):
            unexplored_frontiers = list(self.state.Fu)
            Q = min([self.state.time[f'robot{i+1}'][s] + self.state.time['goal'][s] - self.state.q_t[i][s] +
                    (1 - s.prob_feasible) * s.exploration_cost for s in unexplored_frontiers])
            '''If there are frontiers which leads to the goal, then the cost to goal from these frontier
            for the robot is the minimum of distance in which the robots can reach the goal'''
            if len(self.state.goal_frontiers) != 0:
                Q_g = min([self.state.time[f'robot{i+1}'][s] + (self.state.time['goal'][s] +
                          s.delta_success_cost) - self.state.q_t[i][s] for s in self.state.goal_frontiers])
                Q_p = Q
                Q = min(Q_g, Q_p)
            cost_for_robots.append(Q + self.cost)
        lower_bound_cost = np.min(cost_for_robots)
        return lower_bound_cost


def find_best_joint_action(unexplored_frontiers, time, num_robots=2, num_iterations=500):
    sigma1 = State(n=num_robots, Fu=unexplored_frontiers, m_t=time)
    """pouct core loop"""
    # Start by creating the root of the tree.
    root = Node(start_state=sigma1)
    # Loop through MCTS iterations.
    for _ in range(num_iterations):
        # One step of MCTS iteration
        leaf = traverse(root)
        simulation_result = rollout(leaf)
        backpropagate(leaf, simulation_result)

    return best_action(root)


def traverse(node):
    '''While the root is fully explored, pick the best uct node.'''
    while node.is_fully_explored and not node.is_terminal_state:
        action = best_uct_action(node)
        action_idx = find_action_index(action, node.actions)
        # check if the action from the parent node leads to the same node
        success_state, failure_state, f_I, T_I, goal_reached = node.get_child_properties(action_idx)
        if f_I is not None:
            success = bernoulli(f_I.prob_feasible).rvs(1)[0]
        else:
            if T_I is not None:
                # If f_I is none, and T_I is not None, then goal is reached
                success = 1
            else:
                '''This is case when both action taken can reach goal'''
                print("Something has to be done here UP!!")
        if node.children[action_idx][success] is not NULL:
            node = node.children[action_idx][success]
        else:
            new_node_state = node.children_properties[action_idx]['outcome'][success]
            new_node = Node(new_node_state, parent=node,
                            prev_action=action, T_I=T_I, goal_reached=goal_reached)
            # add this 'new node' to the parent's child
            node.children[action_idx][success] = new_node
            # set this new node as node
            node = new_node

    '''If the node is terminal state, or goal_state return the node'''
    if node.is_terminal_state:
        return node

    '''If the node is not terminal node, or goal node, the node is first discovered.'''
    # 1. Pick a new action from the unexplored actions from the node
    action = node.unexplored_actions.pop()
    action_idx = find_action_index(action, node.actions)
    # If the child node is already expanded, use the previously saved information
    if node.children_properties[action_idx]['has_value'] is True:
        success_state, failure_state, f_I, T_I, goal_reached = node.get_child_properties(action_idx)
    else:
        success_state, failure_state, f_I, T_I, goal_reached = move_robots(node.state, action)
        node.update_child_properties(action_idx, success_state, failure_state, f_I, T_I, goal_reached)

    if f_I is not None:
        success = bernoulli(f_I.prob_feasible).rvs(1)[0]
    else:
        if T_I is not None:
            # If f_I is none, and T_I is not None, then goal is reached
            success = 1
        else:
            '''This is case when both action taken can reach goal'''
            print("Something has to be done here!!")

    new_node_state = node.children_properties[action_idx]['outcome'][success]

    # 2. Create a new child (new leaf)
    new_node = Node(new_node_state, parent=node,
                    prev_action=action, T_I=T_I, goal_reached=goal_reached)
    # 3. Add that child to the list of children
    node.children[action_idx][success] = new_node
    # 4. return that new child
    return new_node


def rollout(node):
    return node.find_rollout_cost()


def backpropagate(node, simulation_result):
    '''Update the node and it's parent (via recursion). We are updating node parents'
    properties because we find best action from a node rather than best node using uct'''
    if node.parent is None:
        return
    action_idx = find_action_index(node.prev_action, node.parent.actions)
    node.parent.action_n[action_idx] += 1
    node.parent.action_values[action_idx].append(simulation_result)
    backpropagate(node.parent, simulation_result)


def best_action(root):
    '''When done sampling, pick the action which has been visited most'''

    def heuristic_cost(robot, f):
        return root.state.time[f'robot{robot+1}'][f]

    if root.state.n == 1:
        max_n = np.max(root.action_n)
        max_n_index = [i for i, j in enumerate(root.action_n) if j == max_n or j == max_n - 1]
        best_action_index = max_n_index[np.argmax([heuristic_cost(robot=0, f=root.actions[i][0]) for i in max_n_index])]
        best_action = root.actions[best_action_index]
    else:
        max_n = np.max(root.action_n)
        max_n_index = [i for i, j in enumerate(root.action_n) if j == max_n or j == max_n - 1]
        h_cost = []
        for i in max_n_index:
            cost = 0
            for j in range(root.state.n):
                cost += heuristic_cost(j, root.actions[i][j])
            h_cost.append(cost)
        best_action_index = max_n_index[np.argmin(h_cost)]
        best_action = root.actions[best_action_index]
    return best_action


def best_uct_action(node, C=500):
    """Pick the best action according to the UCB/UCT algorithm"""
    actions = node.actions
    values = node.action_values
    n = node.action_n
    Q = []
    for i, _ in enumerate(actions):
        UCB = np.sum(values[i]) / n[i] - C * np.sqrt((np.log(sum(n))) / (n[i]))
        Q.append(UCB)
    action = actions[np.argmin(Q)]
    return action
