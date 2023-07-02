#include <cmath>
#include <iostream>
#include <map>
#include <vector>

namespace set {
std::vector<long> u(std::vector<long> &a, std::vector<long> &b) {
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::vector<long> result;
  std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                 std::back_inserter(result));
  return result;
}
} // namespace set

double get_frontier_time_by_triangle_formation(double a, double b, double c,
                                               double time_travelled) {
  double epsilon = 0.1;
  if (a + b < c) {
    double new_time = time_travelled * c / (a + b);
    return new_time;
  } else if (b > a + c) {
    double new_time = time_travelled * c / (a + b);
    return new_time;
  } else if (a > b + c) {
    double new_time = time_travelled * c / (a + b);
    return new_time;
  }
  // else if absolute of sum of a and b - c is less than epsilon
  else if (std::abs(a + b - c) <= epsilon) {
    // the frontier is horizontally aligned (in positive direction)
    double new_time = c - time_travelled;
    return new_time;
  } else if (std::abs(a + c - b) <= epsilon) {
    // the frontier is horizontally aligned (in negative direction)
    double new_time = c + time_travelled;
    return new_time;
  } else if (std::abs(b + c - a) <= epsilon) {
    // all the frontiers lie in the same line
    double new_time = std::abs(c - time_travelled);
    return new_time;
  }

  std::vector<double> frontier_point;
  if (a > 0) {
    double x = (c * c - b * b + a * a) / (2 * a);
    double y = sqrt(c * c - x * x);
    // array of x and y
    frontier_point = {x, y};
  } else {
    std::cout << "First side of triangle not > 0" << std::endl;
  }

  std::vector<double> new_point = {time_travelled, 0};
  double new_time = sqrt(pow(new_point[0] - frontier_point[0], 2) +
                         pow(new_point[1] - frontier_point[1], 2));
  return new_time;
}

int find_action_index(const std::vector<long> &action,
                      const std::vector<std::vector<long>> &all_actions) {
  int action_index = -1;
  for (int i = 0; i < all_actions.size(); i++) {
    if (all_actions[i] == action) {
      action_index = i;
      break;
    }
  }
  return action_index;
}

// Find the index of  maximum value and one less than maximum value in vector v
template <typename T> std::vector<int> findIndex(std::vector<T> const &v) {
  int max_value = *std::max_element(v.begin(), v.end());
  std::vector<int> indices;
  auto it = v.begin();
  while ((it = std::find_if(it, v.end(), [&](T const &e) {
            return (e == max_value || e == max_value - 1);
          })) != v.end()) {
    indices.push_back(std::distance(v.begin(), it));
    it++;
  }
  return indices;
}

// A function get all combination of a vector of type long
std::vector<std::vector<long>> get_combinations(std::vector<long> &v, int k) {
  std::vector<std::vector<long>> result;
  std::vector<long> combination;
  std::function<void(int)> get_combinations_helper = [&](int start) {
    if (combination.size() == k) {
      result.push_back(combination);
      return;
    }
    for (int i = start; i < v.size(); ++i) {
      combination.push_back(v[i]);
      get_combinations_helper(i + 1);
      combination.pop_back();
    }
  };
  get_combinations_helper(0);
  return result;
}

// A function that takes combination and return all permutations of it
std::vector<std::vector<long>> get_permutations(std::vector<long> v) {
  std::vector<std::vector<long>> result;
  do {
    result.push_back(v);
  } while (std::next_permutation(v.begin(), v.end()));
  return result;
}

// A function that combines get_combinations and get_permutations to return all
// permutations by doing combinations
std::vector<std::vector<long>> get_all_permutations(std::vector<long> &v,
                                                    int k) {
  std::vector<std::vector<long>> result;
  std::vector<std::vector<long>> combinations = get_combinations(v, k);
  for (auto &combination : combinations) {
    std::vector<std::vector<long>> permutations = get_permutations(combination);
    result.insert(result.end(), permutations.begin(), permutations.end());
  }
  return result;
}

// A function for permutations with replacement
std::vector<std::vector<long>>
get_permutations_with_replacement(std::vector<long> &v, int k) {
  std::vector<std::vector<long>> result;
  std::vector<long> combination(k);
  std::function<void(int)> get_permutations_with_replacement_helper =
      [&](int start) {
        if (start == k) {
          result.push_back(combination);
          return;
        }
        for (int i = 0; i < v.size(); ++i) {
          combination[start] = v[i];
          get_permutations_with_replacement_helper(start + 1);
        }
      };
  get_permutations_with_replacement_helper(0);
  return result;
}

// A funtion that combines get_permutations or get_permutations with replacement
// according to number of frontiers and number of robots
std::vector<std::vector<long>>
get_action_combinations(std::vector<long> v, int repeat,
                        bool same_action = false) {
  /* If same action = false, permutations without replacement
  If no.of frontiers > no of robot, permute the frontiers*/
  if (!same_action && v.size() >= repeat) {
    return get_all_permutations(v, repeat);
  }
  /* If same action = true, permutations with replacement
  If no.of frontiers < no.of robots, make sure some robot explore 'same'
  frontier*/
  std::vector<std::vector<long>> actions =
      get_permutations_with_replacement(v, repeat);
  /* If same_action = false, i.e you want all robot to prevent exploring 'same'
  frontier 'as much as possible'. Hence Maximum allowed same frontiers = no. of
  frontiers / robot*/
  if (!same_action && v.size() > 1) {
    std::vector<std::vector<long>> final_action;
    // same_action_max as ceil of no. of frontiers / no. of robots
    int same_action_max = std::ceil(float(repeat) / v.size());
    // if number of same frontiers is less than or equal to same_action_max, add
    // to final_action
    for (auto action : actions) {
      // find the count of maximum number of times an element gets repeated in
      // action
      int max_count = 0;
      bool all_elements = true;
      for (auto element : v) {
        int count = std::count(action.begin(), action.end(), element);
        if (count == 0) {
          all_elements = false;
          break;
        }
        if (count > max_count) {
          max_count = count;
        }
      }
      if (max_count <= same_action_max && all_elements) {
        final_action.push_back(action);
      }
    }
    return final_action;
  }
  return actions;
}
