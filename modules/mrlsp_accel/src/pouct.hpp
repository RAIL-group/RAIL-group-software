#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

typedef std::shared_ptr<State> StatePtr;

struct ChildrenProperties {
  StatePtr failure_state;
  StatePtr success_state;
  long f_I = -1;
  double T_I = -1;
  bool goal_reached = false;
  bool has_value = false;

  ChildrenProperties() = default;

  void update_child_properties(StatePtr success_state, StatePtr failure_state,
                               long f_I, double T_I, bool goal_reached) {
    this->failure_state = failure_state;
    this->success_state = success_state;
    this->f_I = f_I;
    this->T_I = T_I;
    this->goal_reached = goal_reached;
    this->has_value = true;
  }

  const std::tuple<long, double, bool> get_child_properties() {
    return std::make_tuple(f_I, T_I, goal_reached);
  }

  StatePtr get_state(bool success) {
    if (success)
      return success_state;
    else
      return failure_state;
  }
};

struct Node {
  StatePtr state;
  std::shared_ptr<Node> parent;
  std::vector<long> prev_action;
  double cost;
  bool goal_reached;
  std::vector<std::vector<long>> actions;
  std::vector<std::vector<long>> unexplored_actions;
  std::vector<std::shared_ptr<Node>> children;
  std::vector<std::shared_ptr<ChildrenProperties>> children_properties;
  std::vector<double> action_values;
  std::vector<int> action_n;
  int action_n_total;
  Node(StatePtr current_state, std::shared_ptr<Node> p = nullptr,
       std::vector<long> p_a = {}, double T_I = 0, bool g_r = false) {
    state = current_state;
    parent = p;
    prev_action = p_a;
    if (parent == nullptr) {
      cost = T_I;
    } else {
      cost = T_I + parent->cost;
    }
    this->goal_reached = g_r;

    // implement get_actions() function
    actions = state->get_actions();

    // copy actions to unexplored_actions
    unexplored_actions = actions;
    action_n_total = 0;
    for (int i = 0; i < actions.size(); i++) {
      // initialize action_values to 0
      action_values.push_back(0);
      // initialize action_n to 0
      action_n.push_back(0);

      // initialize success and failure state as nullptr in children for size of
      // actions
      children.push_back(nullptr);
      children.push_back(nullptr);

      std::shared_ptr<ChildrenProperties> child_properties =
          std::make_shared<ChildrenProperties>();
      children_properties.push_back(child_properties);
    }

    if (!goal_reached) {
      std::vector<long> state_unexplored_frontiers =
          state->unexplored_frontiers_hash;
      if (state_unexplored_frontiers.size() == 0) {
        if (state->goal_frontiers.size() != 0) {
          std::vector<double> all_action_cost;
          for (auto &action : actions) {
            std::vector<double> all_robot_cost;
            for (int i = 0; i < action.size(); i++) {
              long f = action[i];
              double cost_to_goal =
                  state->robot_distances.at(std::pair<int, long>(i, f)) +
                  state->hash_to_frontier[f]->delta_success_cost +
                  state->goal_distances->at(f) - state->q_t[i][f];
              all_robot_cost.push_back(cost_to_goal);
            }
            all_action_cost.push_back(*std::min_element(all_robot_cost.begin(),
                                                        all_robot_cost.end()));
          }
          cost +=
              *std::min_element(all_action_cost.begin(), all_action_cost.end());
          goal_reached = true;
        } else {
          /* When no frontier leads to the goal, and no unexplored frontiers
          left, assume that the parent node unexplored frontier leads to the
          goal, and add the success cost of that frontier. */
          long f = parent->state->unexplored_frontiers_hash[0];
          std::vector<double> all_robot_cost;
          for (int i = 0; i < state->num_robots; i++) {
            double cost_to_goal =
                parent->state->robot_distances.at(std::pair<int, long>(i, f)) +
                parent->state->hash_to_frontier[f]->delta_success_cost +
                state->goal_distances->at(f) - state->q_t[i][f];
            all_robot_cost.push_back(cost_to_goal);
          }
          cost +=
              *std::min_element(all_robot_cost.begin(), all_robot_cost.end());
          goal_reached = true;
        }
      }
    }
  }

  void clear_children() {
    for (auto const &child : children) {
      if (child != nullptr) {
        child->clear_children();
      }
    }
    children.clear();
  }

  bool is_fully_explored() const { return (unexplored_actions.size() == 0); }

  bool is_terminal_state() const { return goal_reached; }

  double find_rollout_cost() const {
    /* Rollout cost from current state'*/
    if (goal_reached)
      return cost;

    double Q = 1e25;
    for (int i = 0; i < state->num_robots; i++) {
      for (auto &s : state->unexplored_frontiers_hash) {
        Q = std::min(Q, state->robot_distances.at(std::pair<int, long>(i, s)) +
                            state->goal_distances->at(s) - state->q_t[i][s] +
                            (1 - state->hash_to_frontier[s]->prob_feasible) *
                                state->hash_to_frontier[s]->exploration_cost);
      }

      /* If there are frontiers which lead to the goal, then the cost to goal
      from these frontiers for the robot is the minimum of distance in which the
      robots can reach the goal */
      for (auto &s : state->goal_frontiers) {
        Q = std::min(Q, state->robot_distances.at(std::pair<int, long>(i, s)) +
                            (state->goal_distances->at(s) +
                             state->hash_to_frontier[s]->delta_success_cost) -
                            state->q_t[i][s]);
      }
    }

    // return minimum cost
    return Q + cost;
  }
};

inline double rollout(const std::shared_ptr<Node> &node) {
  return node->find_rollout_cost();
}

void backpropagate(std::shared_ptr<Node> node, double simulation_result) {
  /* Update the node and it's parent. We are updating node parents' properties
  because we find best action from a node rather than best node using uct */
  while (node->parent != nullptr) {
    int action_idx =
        find_action_index(node->prev_action, node->parent->actions);
    node->parent->action_n[action_idx] += 1;
    node->parent->action_n_total += 1;
    node->parent->action_values[action_idx] += simulation_result;
    node = node->parent;
  }
}

inline double heuristic_cost(const std::shared_ptr<Node> &node, int robot,
                             long frontier) {
  return node->state->robot_distances[std::make_pair(robot, frontier)];
}

std::vector<long> best_action(std::shared_ptr<Node> root) {
  /* When done sampling, pick the action which has been visited most*/

  // find indices with maximum visits, and one less than maximum visits
  std::vector<int> max_n_index = findIndex(root->action_n);
  std::vector<double> heuristic_costs;
  int best_action_index;
  if (root->state->num_robots == 1) {
    // find heuristic_cost for all action with max_n_index in root.actions
    // index of lowest heuristic cost
    for (auto &index : max_n_index) {
      heuristic_costs.push_back(
          heuristic_cost(root, 0, root->actions[index][0]));
    }
    best_action_index = max_n_index[std::min_element(heuristic_costs.begin(),
                                                     heuristic_costs.end()) -
                                    heuristic_costs.begin()];
  } else {
    for (auto &index : max_n_index) {
      double cost = 0;
      for (int i = 0; i < root->state->num_robots; i++) {
        cost += heuristic_cost(root, i, root->actions[index][i]);
      }
      heuristic_costs.push_back(cost);
    }
    best_action_index = max_n_index[std::min_element(heuristic_costs.begin(),
                                                     heuristic_costs.end()) -
                                    heuristic_costs.begin()];
  }
  return root->actions[best_action_index];

  // Update: We can combine 1 robot and mutliple robot. Code inside else
  // statement works for both cases
}

const std::vector<long> &best_uct_action(const std::shared_ptr<Node> &node,
                                         const double &C = 500) {
  /*Pick the best action according to the UCB/UCT algorithm*/
  std::vector<double> Q;
  const std::vector<int> &n = node->action_n;

  for (int i = 0; i < n.size(); i++) {
    Q.push_back(node->action_values[i] / n[i] -
                C * sqrt(log(node->action_n_total) / n[i]));
  }

  return node->actions[std::min_element(Q.begin(), Q.end()) - Q.begin()];
}

// Before writing traverse, write Node class
std::shared_ptr<Node> traverse(std::shared_ptr<Node> node) {
  std::vector<long> action;
  std::mt19937 rng{std::random_device{}()};
  int action_idx;
  std::shared_ptr<Node> current_node = node;

  // While the node is fully explored, pick the best uct node.
  while (current_node->is_fully_explored() &&
         !current_node->is_terminal_state()) {
    action = best_uct_action(current_node);
    action_idx = find_action_index(action, current_node->actions);
    auto [f_I, T_I, goal_reached] =
        current_node->children_properties[action_idx]->get_child_properties();
    bool success;
    if (f_I != -1) {
      std::bernoulli_distribution d(
          current_node->state->hash_to_frontier[f_I]->prob_feasible);
      success = d(rng);
    } else {
      if (T_I != -1) {
        // if f_I is -1 and T_I is not -1, then goal is reached
        success = 1;
      } else {
        // This is case when both action taken can reach goal
        std::cout << "Something has to be done here UP!!" << std::endl;
      }
    }
    if (current_node->children[2 * action_idx + success] != nullptr) {
      current_node = current_node->children[2 * action_idx + success];
    } else {
      StatePtr new_node_state =
          current_node->children_properties[action_idx]->get_state(success);
      std::shared_ptr<Node> new_node = std::make_shared<Node>(
          new_node_state, current_node, action, T_I, goal_reached);
      // add this new node to parent's children
      current_node->children[2 * action_idx + success] = new_node;
      // set current_node to new_node
      current_node = new_node;
    }
  }

  /* If the node is terminal state, or goal state, return the node */
  if (current_node->is_terminal_state())
    return current_node;

  /* If the node is not terminal node, or goal node, the node is first
   * discovered*/
  // 1. Pick a new action from the first action of unexplored actions of the
  // node and remove it from unexplored actions
  action = current_node->unexplored_actions[0];
  current_node->unexplored_actions.erase(
      current_node->unexplored_actions.begin());
  // If the child node is already expanded, use the previously saved information
  // from children_properties
  action_idx = find_action_index(action, current_node->actions);

  // initialize success_state, failure_state, f_I, T_I, goal_reached
  long f_I;
  double T_I;
  bool goal_reached;
  bool success;
  if (current_node->children_properties[action_idx]->has_value) {
    auto [f_I_temp, T_I_temp, goal_reached_temp] =
        current_node->children_properties[action_idx]->get_child_properties();
    f_I = f_I_temp;
    T_I = T_I_temp;
    goal_reached = goal_reached_temp;
  } else {
    // change state_ptr to state
    State state = *current_node->state;
    auto [success_state_ptr, failure_state_ptr, f_I_temp, T_I_temp,
          goal_reached_temp] = move_robots(state, action);
    f_I = f_I_temp;
    T_I = T_I_temp;
    goal_reached = goal_reached_temp;
    // update child properties
    current_node->children_properties[action_idx]->update_child_properties(
        success_state_ptr, failure_state_ptr, f_I, T_I, goal_reached);
  }

  if (f_I != -1) {
    // bernoulli sample
    std::bernoulli_distribution d(
        current_node->state->hash_to_frontier[f_I]->prob_feasible);
    success = d(rng);
  } else {
    if (T_I != -1) {
      // if f_I is -1 and T_I is not -1, then goal is reached
      success = 1;
    } else {
      // This is case when both action taken can reach goal
      std::cout << "Something has to be done here DOWN!!" << std::endl;
    }
  }

  StatePtr new_node_state =
      current_node->children_properties[action_idx]->get_state(success);
  // 2. create a new child (new leaf) shared pointer
  std::shared_ptr<Node> new_node = std::make_shared<Node>(
      new_node_state, current_node, action, T_I, goal_reached);
  // 3. add the child to the list of children
  current_node->children[2 * action_idx + success] = new_node;
  // 4. return the child

  return new_node;
}

std::vector<long> find_best_joint_action_accel(
    const int num_robots, std::vector<FrontierDataMRPtr> unexplored_frontiers,
    std::map<std::pair<int, long>, double> robot_distances,
    std::map<long, double> goal_distances,
    std::map<std::pair<long, long>, double> frontier_distances,
    int num_iterations) {
  // Make State shared_ptr for using all the arguments
  StatePtr sigma1 =
      std::make_shared<State>(num_robots, unexplored_frontiers, robot_distances,
                              goal_distances, frontier_distances);
  // pouct core loop
  // Start by creating the root of the tree
  std::shared_ptr<Node> root = std::make_shared<Node>(sigma1);
  // Loop through MCTS iterations
  for (int i = 0; i < num_iterations; i++) {
    // One step of MCTS iteration
    std::shared_ptr<Node> leaf = traverse(root);
    double simulation_result = rollout(leaf);
    backpropagate(leaf, simulation_result);
  }
  auto act = best_action(root);
  root->clear_children();
  return act;
}
