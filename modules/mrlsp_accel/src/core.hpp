#include "mrlsp_utility.hpp"
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

struct FrontierDataMR {
  double prob_feasible;
  double delta_success_cost;
  double exploration_cost;
  long hash_id;
  bool is_from_last_chosen;

  FrontierDataMR(double prob_feasible, double delta_success_cost,
                 double exploration_cost, long hash_id,
                 bool is_from_last_chosen)
      : prob_feasible(prob_feasible), delta_success_cost(delta_success_cost),
        exploration_cost(exploration_cost), hash_id(hash_id),
        is_from_last_chosen(is_from_last_chosen) {}

  long get_hash() const { return hash_id; }
};

typedef std::shared_ptr<FrontierDataMR> FrontierDataMRPtr;
typedef std::vector<std::map<long, double>> ListOfRobotsProgress;

const std::pair<long, double>
find_progress_and_frontier_for_robot(const ListOfRobotsProgress &q_t,
                                     const int robot_id) {
  auto el = *std::max_element(q_t[robot_id].begin(), q_t[robot_id].end(),
                              [](const std::pair<long, double> &a,
                                 const std::pair<long, double> &b) -> bool {
                                return a.second < b.second;
                              });
  if (el.second > 0) {
    return el;
  } else {
    return std::make_pair<long, double>(-1, 0);
  }
}

// define class named State
struct State {
  int num_robots;
  std::vector<FrontierDataMRPtr> unexplored_frontiers;
  std::vector<long> unexplored_frontiers_hash;
  std::map<std::pair<int, long>, double> robot_distances;
  std::shared_ptr<std::map<long, double>> goal_distances;
  std::shared_ptr<std::map<std::pair<long, long>, double>> frontier_distances;
  std::vector<long> goal_frontiers;
  // initialize a vector q_t that stores progress for each robot for each frontier
  ListOfRobotsProgress q_t;
  // initialize a mapping from hash to frontier to retrieve frontier data from hash
  std::map<long, FrontierDataMRPtr> hash_to_frontier;

  State(const int n, std::vector<FrontierDataMRPtr> &Fu,
        std::map<std::pair<int, long>, double> r_d, std::map<long, double> &g_d,
        std::map<std::pair<long, long>, double> &f_d) {
    goal_distances = std::make_shared<std::map<long, double>>();
    frontier_distances =
        std::make_shared<std::map<std::pair<long, long>, double>>();

    for (auto &element : g_d) {
      goal_distances->insert(element);
    }
    for (auto &element : f_d) {
      frontier_distances->insert(element);
    }
    num_robots = n;
    unexplored_frontiers = Fu;
    this->robot_distances = r_d;

    for (int i = 0; i < num_robots; i++) {
      std::map<long, double> progress;
      for (auto &f : Fu) {
        progress[f->hash_id] = 0;
      }
      q_t.push_back(progress);
    }
    for (int i = 0; i < unexplored_frontiers.size(); i++) {
      long f_hash = unexplored_frontiers[i]->hash_id;
      unexplored_frontiers_hash.push_back(f_hash);
      hash_to_frontier[f_hash] = unexplored_frontiers[i];
    }
  }

  // define a function to remove a frontier f with hash_id f_hash from
  // unexplored_frontiers
  void remove_frontier_from_unexplored_frontiers(long f_hash) {
    for (int i = 0; i < unexplored_frontiers.size(); i++) {
      if (unexplored_frontiers_hash[i] == f_hash) {
        unexplored_frontiers.erase(unexplored_frontiers.begin() + i);
        unexplored_frontiers_hash.erase(unexplored_frontiers_hash.begin() + i);
        break;
      }
    }
  }

  // define a function to add a frontier f with hash_id f_hash to goal_frontiers
  void add_frontier_to_goal_frontiers(long f_hash) {
    goal_frontiers.push_back(f_hash);
  }

  // define a function to change robot_distances for robot r to frontier f with
  // hash_id f_hash to distance new_rd
  void change_rd_for_robot(int robot_id, long f_hash, double new_rd) {
    robot_distances[std::make_pair(robot_id, f_hash)] = new_rd;
  }

  // define a function copy_state that returns a copy of the current state
  State copy_state() {
    State s(*this);
    return s;
  }

  const std::vector<std::vector<long>> get_actions() {
    auto restrict_action_according_to_progress =
        [&](std::vector<std::vector<long>> all_actions) {
          /*This function restricts the action so that if the robots are making
          progress towards some frontier, then the other robot revealing some
          frontier doesn't interrupt the ongoing action*/
          std::vector<std::vector<long>> final_action;
          bool save = false;
          for (auto &action : all_actions) {
            for (int i = 0; i < action.size(); i++) {
              auto [frontier, progress] =
                  find_progress_and_frontier_for_robot(q_t, i);
              // If the frontier is not in unexplored frontier, then the
              // frontier is just explored
              if (std::find(unexplored_frontiers_hash.begin(),
                            unexplored_frontiers_hash.end(),
                            frontier) == unexplored_frontiers_hash.end()) {
                /* If the frontier is not in goal frontier, then the frontier
                doesn't lead to the goal and the robot needs to return back from
                that frontier. So the robot has not 'made' any progress towards
                any frontier that might reveal goal.*/
                if (std::find(goal_frontiers.begin(), goal_frontiers.end(),
                              frontier) == goal_frontiers.end()) {
                  frontier = -1;
                }
              }

              if (frontier == -1) {
                save = true;
                continue;
              } else {
                if (action[i] != frontier) {
                  save = false;
                  continue;
                } else {
                  save = true;
                  continue;
                }
              }
            }
            if (save)
              final_action.push_back(action);
          }
          return final_action;
        };
    std::vector<long> all_unexplored_frontiers = unexplored_frontiers_hash;
    std::vector<long> frontiers_to_explore =
        set::u(unexplored_frontiers_hash, goal_frontiers);
    if (num_robots == 1) {
      std::vector<std::vector<long>> all_actions;
      for (auto &f : frontiers_to_explore) {
        std::vector<long> action;
        action.push_back(f);
        all_actions.push_back(action);
      }
      return all_actions;
    }
    std::vector<std::vector<long>> actions =
        get_action_combinations(frontiers_to_explore, num_robots);
    return restrict_action_according_to_progress(actions);
  }
};

std::pair<long, double>
get_frontier_of_knowledge_and_time(const State &s,
                                   const std::vector<long> &action) {
  std::vector<long> goal_frontiers;
  bool can_reach_goal = false;
  long f_I;
  double T_I;
  if (s.goal_frontiers.size() != 0) {
    can_reach_goal = true;
    goal_frontiers = s.goal_frontiers;
  }
  std::vector<double> all_TI;
  std::vector<long> all_frontiers;
  for (int i = 0; i < action.size(); i++) {
    if (can_reach_goal &&
        std::find(goal_frontiers.begin(), goal_frontiers.end(), action[i]) !=
            goal_frontiers.end()) {
      continue;
    } else {
      FrontierDataMRPtr f = s.hash_to_frontier.at(action[i]);
      double Ti =
          s.robot_distances.at(std::pair<int, long>(i, action[i])) +
          std::min((f->delta_success_cost + s.goal_distances->at(action[i])),
                   f->exploration_cost) -
          s.q_t[i].at(action[i]);
      all_TI.push_back(Ti);
      all_frontiers.push_back(action[i]);
    }
  }
  if (all_TI.size() == 0) {
    // no unexplored frontiers have been added; i.e both actions are goal
    // frontiers
    return std::make_pair(-1, -1);
  }
  f_I = all_frontiers[std::min_element(all_TI.begin(), all_TI.end()) -
                      all_TI.begin()];
  // T_I is the minimum of all_TI
  T_I = *std::min_element(all_TI.begin(), all_TI.end());
  return std::make_pair(f_I, T_I);
}

std::pair<ListOfRobotsProgress,
          std::vector<std::map<std::pair<long, long>, double>>>
find_q_t_for_action(State &s, const double &T_I,
                    const std::vector<long> &action) {
  ListOfRobotsProgress q_t;
  std::vector<std::map<std::pair<long, long>, double>> residue_time;

  // get union of unexplored frontiers and goal frontiers of State s in
  // frontiers_to_keep_track
  std::vector<long> all_unexplored_frontiers = s.unexplored_frontiers_hash;
  std::vector<long> frontiers_to_keep_track =
      set::u(all_unexplored_frontiers, s.goal_frontiers);

  for (int i = 0; i < s.num_robots; i++) {
    std::map<long, double> progress_for_robot; // q_t_dict in python version
    std::map<std::pair<long, long>, double>
        residue_time_for_robot; // residue_time in python version
    auto [prev_frontier, prev_progress] =
        find_progress_and_frontier_for_robot(s.q_t, i);
    long current_frontier = action[i];
    // for each frontier in frontiers_to_keep_track
    for (auto &f : frontiers_to_keep_track) {
      /* If the current frontier is not the frontier that the robot is moving
      towards, then no progress is made towards that frontier (set time to 0),
      and the current frontier is also not the previous frontier (do nothing)*/
      if (f != current_frontier) {
        // we don't want to set the previous frontier time as 0 right away
        if (f != prev_frontier) {
          progress_for_robot[f] = 0;
        }
      } else {
        // if in previous state, the robot was not exploring any frontier
        double time_to_current_frontier;
        if (prev_frontier == -1) {
          time_to_current_frontier =
              s.robot_distances.at(std::pair<int, long>(i, current_frontier));
          // if the robot has entered that frontier
          if (T_I > time_to_current_frontier) {
            progress_for_robot[f] = T_I - time_to_current_frontier;
          } else {
            // Scenario of residue while moving towards frontier f
            progress_for_robot[f] = 0;
            residue_time_for_robot[std::pair<long, long>(prev_frontier, f)] =
                T_I;
          }
        } else if (current_frontier == prev_frontier) {
          progress_for_robot[f] = prev_progress + T_I;
        } else {
          /* If the time of knowledge > the progress made on previous frontier
          then the robot gets out of the frontier */
          if (T_I >= prev_progress) {
            double out_and_explore_time = T_I - prev_progress;
            double inter_frontier_time = s.frontier_distances->at(
                std::pair<long, long>(f, prev_frontier));
            progress_for_robot[prev_frontier] = 0;
            if (out_and_explore_time > inter_frontier_time) {
              progress_for_robot[f] =
                  out_and_explore_time - inter_frontier_time;
            } else {
              progress_for_robot[f] = 0;
              residue_time_for_robot[std::pair<long, long>(prev_frontier, f)] =
                  out_and_explore_time;
            }
          } else {
            /* else the robot is currently coming out of the same frontier it
             * was exploring before */
            progress_for_robot[prev_frontier] = prev_progress - T_I;
            progress_for_robot[f] = 0;
          }
        }
      }
    }
    q_t.push_back(progress_for_robot);
    residue_time.push_back(residue_time_for_robot);
  }
  return std::make_pair(q_t, residue_time);
}

std::map<std::pair<int, long>, double> get_time_from_qt(
    std::shared_ptr<State> s, ListOfRobotsProgress &new_q_t,
    std::vector<std::map<std::pair<long, long>, double>> residue_time) {
  std::map<std::pair<int, long>, double> new_time;
  std::map<std::pair<int, long>, double> old_time = s->robot_distances;
  std::vector<long> all_unexplored_frontiers = s->unexplored_frontiers_hash;
  std::vector<long> frontiers_to_keep_track =
      set::u(all_unexplored_frontiers, s->goal_frontiers);
  for (int i = 0; i < s->num_robots; i++) {
    // initialize robot distance
    std::map<std::pair<int, long>, double> r_d;
    auto [prev_frontier, prev_progress] =
        find_progress_and_frontier_for_robot(new_q_t, i);
    if (prev_frontier == -1) {
      // check if there is resiude time for this robot
      if (!residue_time[i].empty()) {
        /* Assumption: Construct a triangle using where robot was coming from,
        where robot is going, and where robot needs to go, and find the distance
        from that triangle. The results will never be negative. */
        long from_frontier = residue_time[i].begin()->first.first;
        long to_frontier = residue_time[i].begin()->first.second;
        double time_travelled = residue_time[i].begin()->second;
        /*If the robot is in 'known' space and not coming out from frontier then
        the time is deduced from the old robot-frontier time. This happens when
        the subgoal is assigned to the robot, but other robot knows about
        "frontier of knowledge" before the current robot gets to see the
        frontier that it is assigned to.*/
        if (from_frontier == -1) {
          double time_to_frontier = 0;
          // a is the distance from robot position to the frontier that it is
          // assigned to, before the robot moved
          double a = old_time[std::pair<int, long>(i, to_frontier)];
          // Handle edge case
          if (a == 0) {
            // If the robot just reached frontier it was assigned to explore,
            // and belief state changed
            for (auto &f : frontiers_to_keep_track) {
              if (f == to_frontier) {
                time_to_frontier = 0;
              } else {
                time_to_frontier = s->frontier_distances->at(
                    std::pair<long, long>(f, to_frontier));
              }
              r_d[std::pair<int, long>(i, f)] = time_to_frontier;
            }
          } else {
            for (auto &f : frontiers_to_keep_track) {
              if (f == to_frontier) {
                time_to_frontier =
                    old_time[std::pair<int, long>(i, f)] - time_travelled;

              } else {
                /* 'b' is the time from the frontier that the robot was
                   previously assigned to, and the current frontier 'f' */
                double b = s->frontier_distances->at(
                    std::pair<long, long>(f, to_frontier));
                /* 'c' is the time from robot position to the frontier,
                    which we are currently calculating, before the robot moved
                 */
                double c = old_time[std::pair<int, long>(i, f)];

                time_to_frontier = get_frontier_time_by_triangle_formation(
                    a, b, c, time_travelled);
              }
              r_d[std::pair<int, long>(i, f)] = time_to_frontier;
            }
          }
        } else {
          /*If the robot is in 'known' space but by coming out from another
             frontier. In this case, the time from the previous frontier to all
             the other frontier - outside time is the robot's time to reach
             other frontiers.*/
          double a = s->frontier_distances->at(
              std::pair<long, long>(from_frontier, to_frontier));
          for (auto &f : frontiers_to_keep_track) {
            bool f_is_to_frontier = f == to_frontier;
            bool f_is_from_frontier = f == from_frontier;
            if (f_is_to_frontier || f_is_from_frontier) {
              if (f_is_from_frontier) {
                r_d[std::pair<int, long>(i, f)] = time_travelled;
              } else {
                r_d[std::pair<int, long>(i, f)] =
                    s->frontier_distances->at(
                        std::pair<long, long>(from_frontier, f)) -
                    time_travelled;
              }
            } else {
              double b = s->frontier_distances->at(
                  std::pair<long, long>(f, to_frontier));
              double c = s->frontier_distances->at(
                  std::pair<long, long>(f, from_frontier));
              r_d[std::pair<int, long>(i, f)] =
                  get_frontier_time_by_triangle_formation(a, b, c,
                                                          time_travelled);
            }
          }
        }
      } else {
        // Update in q_t and not here.
        std::cout << "Not sure what to do here !!" << std::endl;
      }

    } else {
      for (auto &f : frontiers_to_keep_track) {
        double time_to_frontier;
        // time for the robot to explore same frontier that it is making
        // progress towards
        if (f == prev_frontier) {
          time_to_frontier = 0;
        } else {
          time_to_frontier =
              prev_progress + s->frontier_distances->at(
                                  std::pair<long, long>(f, prev_frontier));
        }
        r_d[std::pair<int, long>(i, f)] = time_to_frontier;
      }
    }
    new_time.insert(r_d.begin(), r_d.end());
  }
  return new_time;
  // Update maybe: May not need old_time
}

std::tuple<std::shared_ptr<State>, std::shared_ptr<State>, long, double, bool>
move_robots(const State &s, std::vector<long> &action) {
  auto failure_state = std::make_shared<State>(s);
  auto [f_I, T_I] = get_frontier_of_knowledge_and_time(s, action);
  bool goal_reached = false;
  if (f_I == -1) {
    /*if f_I and T_I are -1, both the action lead to the goal
    i.e success_cost is the minimum of two robots reaching the goal*/
    std::vector<double> all_time_to_goal;
    for (int i = 0; i < action.size(); i++) {
      FrontierDataMRPtr f = s.hash_to_frontier.at(action[i]);
      double time_to_goal =
          s.robot_distances.at(std::pair<int, long>(i, action[i])) +
          (f->delta_success_cost + s.goal_distances->at(action[i])) -
          s.q_t.at(i).at(action[i]);
      all_time_to_goal.push_back(time_to_goal);
    }
    T_I = *std::min_element(all_time_to_goal.begin(), all_time_to_goal.end());
    goal_reached = true;
  }
  auto [new_q_t, residue_time] =
      find_q_t_for_action(*failure_state, T_I, action);
  failure_state->q_t = new_q_t;
  auto success_state = std::make_shared<State>(*failure_state);
  if (f_I != -1) {
    // remove f_I from unexplored frontier in both success state and failure
    // state
    failure_state->remove_frontier_from_unexplored_frontiers(f_I);
    success_state->remove_frontier_from_unexplored_frontiers(f_I);
    // add f_I to goal frontier only in success state
    success_state->add_frontier_to_goal_frontiers(f_I);
  }
  success_state->robot_distances =
      get_time_from_qt(success_state, new_q_t, residue_time);
  failure_state->robot_distances =
      get_time_from_qt(failure_state, new_q_t, residue_time);

  /* Check for goal reached:
  If the robot progress along a frontier is greater than or equal to delta
  success cost, goal is reached. */

  std::vector<double> T_I_list;
  for (int i = 0; i < s.num_robots; i++) {
    auto [frontier, progress] =
        find_progress_and_frontier_for_robot(new_q_t, i);
    if (progress != 0) {
      double d_sc = s.hash_to_frontier.at(frontier)->delta_success_cost;
      double toreachgoal = d_sc + s.goal_distances->at(frontier);
      double epsilon = 1.0;
      if (toreachgoal - progress <= epsilon) {
        double rem = progress - (d_sc + s.goal_distances->at(frontier));
        T_I_list.push_back(T_I - rem + epsilon);
        goal_reached = true;
      }
    }
  }
  if (T_I_list.size() != 0) {
    T_I = *std::min_element(T_I_list.begin(), T_I_list.end());
  }
  return std::make_tuple(success_state, failure_state, f_I, T_I, goal_reached);
}
