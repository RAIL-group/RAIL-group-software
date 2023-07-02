#include <map>
#include <vector>
#include <utility>


struct FrontierData {
  double prob_feasible;
  double delta_success_cost;
  double exploration_cost;
  long hash_id;
  bool is_from_last_chosen;

  FrontierData(double prob_feasible,
               double delta_success_cost,
               double exploration_cost,
               long hash_id,
               bool is_from_last_chosen)
      : prob_feasible(prob_feasible),
        delta_success_cost(delta_success_cost),
        exploration_cost(exploration_cost),
        hash_id(hash_id),
        is_from_last_chosen(is_from_last_chosen) {}

  long get_hash() const { return hash_id; }
};


typedef std::shared_ptr<FrontierData> FrontierDataPtr;


struct FState {
  std::vector<long> frontier_id_list;
  double cost;
  double prob;

  FState(const FrontierDataPtr &new_frontier,
         const std::map<long, double> &robot_distances,
         const std::map<long, double> &goal_distances,
         const std::map<std::pair<long, long>, double> &frontier_distances) {
    // This is the first frontier, so the robot must accumulate a cost of getting to the frontier
    // Known cost (travel to frontier)
    double p = new_frontier->prob_feasible;
    double sc = new_frontier->delta_success_cost + goal_distances.at(new_frontier->hash_id);
    double ec = new_frontier->exploration_cost;
    double kc = robot_distances.at(new_frontier->hash_id);
    // if (new_frontier->is_from_last_chosen) {
    //   kc -= 10.0;
    // }

    // Update the state properties
    cost = kc + p * sc + (1 - p) * ec;
    prob = 1 - p;
    frontier_id_list.push_back(new_frontier->hash_id);
  }

  FState(const FState &old_state,
         const FrontierDataPtr &new_frontier,
         const std::map<long, double> &robot_distances,
         const std::map<long, double> &goal_distances,
         const std::map<std::pair<long, long>, double> &frontier_distances) :
      frontier_id_list(old_state.frontier_id_list),
      cost(old_state.cost),
      prob(old_state.prob) {
    // Compute some intermediate properties
    double p = new_frontier->prob_feasible;
    double sc = new_frontier->delta_success_cost + goal_distances.at(new_frontier->hash_id);
    double ec = new_frontier->exploration_cost;
  
    double kc = frontier_distances.at(std::pair<long, long>(
        new_frontier->hash_id, frontier_id_list.back()));

    cost += prob * (kc + p * sc + (1 - p) * ec);
    prob *= (1 - p);
    frontier_id_list.push_back(new_frontier->hash_id);
  }

  bool operator<(const FState &other) {
    return cost < other.cost;
  }

};


FState get_lowest_cost_ordering_sub(
    const FState &prev_state,
    const std::vector<FrontierDataPtr> &frontiers,
    const std::map<long, double> &robot_distances,
    const std::map<long, double> &goal_distances,
    const std::map<std::pair<long, long>, double> &frontier_distances,
    double *bound) {
  if (frontiers.size() == 1) {
    FState state(prev_state,
                 frontiers[0],
                 robot_distances,
                 goal_distances,
                 frontier_distances);
    *bound = std::min(*bound, state.cost);
    return state;
  }

  if (prev_state.cost > *bound) {
    return prev_state;
  }

  // If the cost is sufficiently low, avoid exhaustive planning, since it will make only
  // an insignificant cost difference at the expense of computation. However, we append
  // the remaining frontiers to the policy so that the length of the policy is the same
  // as the number of frontiers (and none are thrown away), important for some
  // applications.
  if (prev_state.prob < 1.0e-12) {
    std::vector<FrontierDataPtr> sub_frontiers;
    std::copy(frontiers.begin() + 1, frontiers.end(), std::back_inserter(sub_frontiers));
    FState state = FState(prev_state,
                          frontiers[0],
                          robot_distances,
                          goal_distances,
                          frontier_distances);
    return get_lowest_cost_ordering_sub(
        state,
        sub_frontiers,
        robot_distances,
        goal_distances,
        frontier_distances,
        bound);
  }

  std::vector<FState> best_states;
  for (auto f_it = frontiers.begin(); f_it != frontiers.end(); ++f_it) {
    std::vector<FrontierDataPtr> sub_frontiers;
    std::copy(frontiers.begin(), f_it, std::back_inserter(sub_frontiers));
    std::copy(f_it + 1, frontiers.end(), std::back_inserter(sub_frontiers));

    FState state = FState(prev_state,
                          *f_it,
                          robot_distances,
                          goal_distances,
                          frontier_distances);
    best_states.push_back(get_lowest_cost_ordering_sub(
        state,
        sub_frontiers,
        robot_distances,
        goal_distances,
        frontier_distances,
        bound));
  }
  return *(std::min_element(best_states.begin(), best_states.end()));
}


std::pair<double, std::vector<long>> get_lowest_cost_ordering(
    const std::vector<FrontierDataPtr> &frontiers,
    const std::map<long, double> &robot_distances,
    const std::map<long, double> &goal_distances,
    const std::map<std::pair<long, long>, double> &frontier_distances) {
  std::vector<FState> best_states;
  double bound = 1.0e10;
  for (auto f_it = frontiers.begin(); f_it != frontiers.end(); ++f_it) {
    std::vector<FrontierDataPtr> sub_frontiers;
    std::copy(frontiers.begin(), f_it, std::back_inserter(sub_frontiers));
    std::copy(f_it + 1, frontiers.end(), std::back_inserter(sub_frontiers));

    FState state = FState(*f_it,
                          robot_distances,
                          goal_distances,
                          frontier_distances);
    if (sub_frontiers.size() == 0) {
      best_states.push_back(state);
      break;
    }

    best_states.push_back(get_lowest_cost_ordering_sub(
        state,
        sub_frontiers,
        robot_distances,
        goal_distances,
        frontier_distances,
        &bound));
  }
  auto sout = std::min_element(best_states.begin(), best_states.end());
  return std::make_pair(sout->cost, sout->frontier_id_list);
}


std::pair<double, std::vector<long>> get_lowest_cost_ordering_beginning_with(
    const FrontierDataPtr &frontier_of_interest,
    const std::vector<FrontierDataPtr> &frontiers,
    const std::map<long, double> &robot_distances,
    const std::map<long, double> &goal_distances,
    const std::map<std::pair<long, long>, double> &frontier_distances) {
  std::vector<FState> best_states;
  double bound = 1.0e10;

  FState state = FState(frontier_of_interest,
                        robot_distances,
                        goal_distances,
                        frontier_distances);

  FState sout = get_lowest_cost_ordering_sub(
      state,
      frontiers,
      robot_distances,
      goal_distances,
      frontier_distances,
      &bound);
  return std::make_pair(sout.cost, sout.frontier_id_list);
}


std::pair<double, std::vector<long>> get_lowest_cost_ordering_not_beginning_with(
    const FrontierDataPtr &frontier_to_avoid,
    const std::vector<FrontierDataPtr> &frontiers,
    const std::map<long, double> &robot_distances,
    const std::map<long, double> &goal_distances,
    const std::map<std::pair<long, long>, double> &frontier_distances) {
  std::vector<FState> best_states;
  double bound = 1.0e10;
  for (auto f_it = frontiers.begin(); f_it != frontiers.end(); ++f_it) {
    if ((*f_it)->hash_id == frontier_to_avoid->hash_id) {
      continue;
    }

    std::vector<FrontierDataPtr> sub_frontiers;
    std::copy(frontiers.begin(), f_it, std::back_inserter(sub_frontiers));
    std::copy(f_it + 1, frontiers.end(), std::back_inserter(sub_frontiers));

    FState state = FState(*f_it,
                          robot_distances,
                          goal_distances,
                          frontier_distances);
    if (sub_frontiers.size() == 0) {
      best_states.push_back(state);
      break;
    }

    best_states.push_back(get_lowest_cost_ordering_sub(
        state,
        sub_frontiers,
        robot_distances,
        goal_distances,
        frontier_distances,
        &bound));
  }
  auto sout = std::min_element(best_states.begin(), best_states.end());
  return std::make_pair(sout->cost, sout->frontier_id_list);
}
