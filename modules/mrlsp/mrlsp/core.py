import numpy as np
import copy
import mrlsp


class State():
    def __init__(self, n, Fu, m_t=None):
        self.n = n
        self.Fu = copy.copy(Fu)
        self.time = mrlsp.utils.utility.copy_dictionary(m_t)
        self.q_t = []
        self.goal_frontiers = set()
        for i in range(self.n):
            progress = {}
            for f in Fu:
                progress[f] = 0
            self.q_t.append(progress)

    def copy_state(self):
        new_state = State(self.n, self.Fu, self.time)
        new_state.q_t = []
        for q in self.q_t:
            new_state.q_t.append(copy.copy(q))
        new_state.goal_frontiers = copy.copy(self.goal_frontiers)
        return new_state

    def get_actions(self):
        def restrict_action_according_to_progress(all_actions):
            '''This function restricts the action so that if the robots are making progress towards
            some frontier, then the other robot revealing some frontier doesn't interrupt the ongoing
            action'''
            final_action = []
            save = False
            for c in all_actions:
                for i, action in enumerate(c):
                    frontier, _ = find_progress_and_frontier_for_robot(self.q_t, i)
                    # If the frontier is not in unexplored frontier, then the frontier is just explored
                    if frontier not in self.Fu:
                        if frontier not in self.goal_frontiers:
                            ''' If the frontier is not in goal frontier, then the frontier doesn't lead to the goal
                            and the robot needs to return back from that frontier. So, the robot has not 'made' any
                            progress towards any frontier that might reveal goal.'''
                            frontier = None
                        else:
                            '''After exploring, the frontier can either lead to the goal; in that case, the
                            action contains that frontier. So no need to set None'''
                            pass

                    if frontier is None:
                        save = True
                        continue
                    else:
                        if frontier != action:
                            save = False
                            continue
                        else:
                            save = True
                            continue
                if save:
                    final_action.append(c)
            return final_action

        frontiers_to_explore = self.Fu.union(self.goal_frontiers)
        if self.n == 1:
            return [[f] for f in frontiers_to_explore]

        actions = mrlsp.utils.utility.get_action_combination(frontiers_to_explore, self.n)
        # Make sure that if a robot is making progress along some frontier, it doesn't change action
        final_action = restrict_action_according_to_progress(actions)
        return final_action
        # return actions

    def find_q_t_for_action(self, T_I, action):
        new_q_t = []
        residue_time = []
        # update for all the robots
        for i in range(self.n):
            q_t_dict = {}
            residue_dict = {}  # used for storing reside
            prev_frontier, progress_prev_frontier = find_progress_and_frontier_for_robot(self.q_t, i)
            current_frontier = action[i]
            frontiers_to_keep_track = self.Fu.union(self.goal_frontiers)
            for f in frontiers_to_keep_track:
                '''
                if the current frontier is not the frontier that the robot is moving towards
                then no progress is made towards that frontier (set time to 0), and the current
                frontier is also not the previous frontier (do nothing)
                '''
                if f != current_frontier:
                    # We don't want to set the previous frontier time as 0 right now.
                    if f != prev_frontier:
                        q_t_dict[f] = 0
                else:
                    # If in previous state, robot was not exploring any frontier
                    if prev_frontier is None:
                        time_to_current_frontier = self.time[f"robot{i+1}"][f]
                        # If the robot has entered that frontier
                        if T_I > time_to_current_frontier:
                            q_t_dict[f] = T_I - time_to_current_frontier
                        else:
                            # Scenario of residue while moving towards frontier f
                            # residue_time = T_I
                            q_t_dict[f] = 0
                            residue_dict['from'] = prev_frontier
                            residue_dict['to'] = f
                            residue_dict['time'] = T_I

                    elif current_frontier == prev_frontier:
                        q_t_dict[f] = progress_prev_frontier + T_I
                    else:
                        '''
                        If the time of knowledge > the progress made on previous frontier then the robot gets out
                        of the frontier
                        '''
                        if T_I >= progress_prev_frontier:
                            out_and_explore_time = T_I - progress_prev_frontier
                            inter_frontier_time = self.time['frontier'][frozenset([f, prev_frontier])]
                            q_t_dict[prev_frontier] = 0
                            if out_and_explore_time > inter_frontier_time:
                                q_t_dict[f] = out_and_explore_time - inter_frontier_time
                            else:
                                # TODO: Add scenario of residue while moving towards frontier f
                                # residue is used in calculation in robot-frontier time
                                # residue_time = inter_frontier_time - out_and_explore_time
                                q_t_dict[f] = 0
                                residue_dict['from'] = prev_frontier
                                residue_dict['to'] = f
                                residue_dict['time'] = out_and_explore_time
                        else:
                            '''
                            else the robot is currently coming out of the same frontier it was exploring before
                            '''
                            q_t_dict[f] = 0
                            q_t_dict[prev_frontier] = progress_prev_frontier - T_I
            new_q_t.append(q_t_dict)
            residue_time.append(residue_dict)

        return new_q_t, residue_time

    def get_time_from_q_t(self, new_q_t, residue_time):
        new_time = {}
        new_time['frontier'] = copy.copy(self.time['frontier'])
        new_time['goal'] = copy.copy(self.time['goal'])
        old_time = mrlsp.utils.utility.copy_dictionary(self.time)
        frontiers_to_keep_track = self.Fu.union(self.goal_frontiers)
        for i in range(self.n):
            time_dict = {}
            prev_frontier, progress = find_progress_and_frontier_for_robot(new_q_t, i)
            if prev_frontier is None:
                # TODO: Check if there is residue distance
                if len(residue_time[i]) != 0:
                    '''
                    Assumption1: If the robot is making progress towards a frontier, its making progress
                    towards all the other frontiers
                    Updated Assumption: Construct a triangle using where robot was coming from, where robot is
                    going, and where robot needs to go, and find the distance from that triangle. The results will
                    never be negative.
                    '''
                    from_frontier = residue_time[i]['from']
                    to_frontier = residue_time[i]['to']
                    time_travelled = residue_time[i]['time']

                    '''
                    If the robot is in 'known' space and not coming out from frontier then the time is
                    deduced from the old robot-frontier time.
                    This happens when the subgoal is assigned to the robot, but other robot knows about
                    "frontier of knowledge" before the current robot gets to see the frontier that it is
                    assigned to.
                    '''
                    if from_frontier is None:
                        ''' 'a' is the distance from robot position to the frontier
                        that it is assigned to, before the robot moved'''
                        a = old_time[f'robot{i + 1}'][to_frontier]
                        # HANDLE EDGE CASE:
                        if a == 0:
                            # If the robot just reached frontier it was assigned to explore, and belief state changed
                            for f in frontiers_to_keep_track:
                                if f == to_frontier:
                                    time_to_frontier = 0
                                else:
                                    time_to_frontier = old_time['frontier'][frozenset([to_frontier, f])]
                                time_dict[f] = time_to_frontier
                        else:
                            for f in frontiers_to_keep_track:
                                if f == to_frontier:
                                    time_to_frontier = old_time[f'robot{i + 1}'][f] - time_travelled
                                else:
                                    ''' 'b' is the time from the frontier that the robot was previously
                                    assigned to, and the current frontier 'f' '''
                                    b = old_time['frontier'][frozenset([to_frontier, f])]
                                    ''' 'c' is the time from robot position to the frontier,
                                    which we are currently calculating, before the robot moved'''
                                    c = old_time[f'robot{i + 1}'][f]

                                    time_to_frontier = mrlsp.utils.utility.get_frontier_time_by_triangle_formation(
                                        a, b, c, time_travelled)
                                time_dict[f] = time_to_frontier

                    else:
                        '''
                        If the robot is in 'known' space but by coming out from another frontier. In this case,
                        the time from the previous frontier to all the other frontier - outside time is the
                        robot's time to reach other frontiers.
                        '''
                        a = old_time['frontier'][frozenset([from_frontier, to_frontier])]
                        for f in frontiers_to_keep_track:
                            # TODO: Check if we need this
                            if f == from_frontier or f == to_frontier:
                                if f == from_frontier:
                                    time_dict[f] = time_travelled
                                else:
                                    time_dict[f] = old_time['frontier'][frozenset([from_frontier, f])] - time_travelled
                            else:
                                b = old_time['frontier'][frozenset([to_frontier, f])]
                                c = old_time['frontier'][frozenset([from_frontier, f])]
                                time_to_frontier = mrlsp.utils.utility.get_frontier_time_by_triangle_formation(
                                    a, b, c, time_travelled)
                                time_dict[f] = time_to_frontier
                else:
                    # Update in q_t and not here.
                    raise AssertionError('Not sure what to do here !!')

            else:
                for f in frontiers_to_keep_track:
                    # time for the robot to explore same frontier that it is making progress towards
                    if prev_frontier == f:
                        time_to_frontier = 0
                    # for any other frontier, come out of the currently exploring frontier and go towards that frontier
                    else:
                        time_to_frontier = progress + self.time["frontier"][frozenset([f, prev_frontier])]
                    time_dict[f] = time_to_frontier

            new_time[f"robot{i+1}"] = time_dict
        return new_time


def move_robots(state, action):
    failure_state = state.copy_state()
    f_I, T_I = get_frontier_of_knowlege_and_time(state, action)
    goal_reached = False
    if f_I is None:
        '''if f_I and T_I is none, both the action leads to the goal.
        i.e success_cost is the minimum of two robots reaching the goal.'''
        # NOTE: after this, one of the robot reaches the goal, and success cost should be 0.
        all_time_to_goal = []
        for i, f in enumerate(action):
            time_to_goal = state.time[f'robot{i+1}'][f] + \
                (f.delta_success_cost + state.time['goal'][f]) - state.q_t[i][f]
            all_time_to_goal.append(time_to_goal)
        time_to_goal = min(all_time_to_goal)
        T_I = time_to_goal
        goal_reached = True

    new_q_t, residue_time = failure_state.find_q_t_for_action(T_I, action)

    failure_state.q_t = new_q_t

    if f_I is not None:
        failure_state.Fu.remove(f_I)

    success_state = failure_state.copy_state()
    if f_I is not None:
        success_state.goal_frontiers.add(f_I)

    success_state.time = success_state.get_time_from_q_t(new_q_t=new_q_t, residue_time=residue_time)
    failure_state.time = failure_state.get_time_from_q_t(new_q_t=new_q_t, residue_time=residue_time)

    '''
    Check for goal reached:
    If the robot progress along a frontier is greater than or equal to delta
    success cost, goal is reached.
    What to do if progress is greater than delta success cost?
    decrease T_I (cost) and q_t by the increment amount'''
    T_I_list = []
    for i in range(state.n):
        frontier, progress = find_progress_and_frontier_for_robot(new_q_t, i)
        if (progress != 0):
            toreachgoal = frontier.delta_success_cost + state.time['goal'][frontier]
            epsilon = 1
            if (toreachgoal - progress <= epsilon):
                rem = progress - (frontier.delta_success_cost + state.time['goal'][frontier])
                T_I_list.append(T_I - rem + epsilon)
                goal_reached = True
        # no need to update intermediate state properties, because goal is reached, and cost is returned
    if len(T_I_list) != 0:
        T_I = min(T_I_list)
    return success_state, failure_state, f_I, T_I, goal_reached


def find_progress_and_frontier_for_robot(q_t, i):
    robot_q_t = q_t[i]
    frontier = None
    progress = 0
    for f in robot_q_t:
        if robot_q_t[f] != 0:
            frontier = f
            progress = robot_q_t[f]
            break
    return frontier, progress


def get_frontier_of_knowlege_and_time(state, action):
    # find if the state is a state from which goal can be reached
    can_reach_goal = False
    if len(state.goal_frontiers) != 0:
        can_reach_goal = True
        goal_frontiers = state.goal_frontiers
    # list to store all the time
    all_TI = []
    all_frontiers = []
    # Iterate over the joint action
    for i, f in enumerate(action):
        if can_reach_goal and f in goal_frontiers:
            continue
        Ti = state.time[f"robot{i+1}"][f] + \
            min(f.delta_success_cost + state.time['goal'][f], f.exploration_cost) - state.q_t[i][f]
        all_TI.append(Ti)
        all_frontiers.append(f)

    if len(all_TI) == 0:
        # no unexplored frontiers have been added; i.e both actions are goal frontiers
        f_I, T_I = None, None
        return f_I, T_I

    f_I = all_frontiers[np.argmin(all_TI)]
    T_I = min(all_TI)

    return f_I, T_I


def update_frontier_properties_for_multirobot(planner, subgoals_initial, distance_mr):
    # remove subgoals that are covered by known space, i.e they don't lead to the goal.
    subgoals = set([copy.copy(s) for s in subgoals_initial if distance_mr['goal'][s] < 100000])
    sg_not_leading_to_goal = subgoals_initial - subgoals
    for sg in subgoals:
        # find the robot which is close to the subgoal 'sg'.
        distance_to_sg = [distance_mr[f'robot{i+1}'][sg] for i in range(len(planner))]
        robot_idx = np.argmin(distance_to_sg)
        for f in planner[robot_idx].subgoals:
            if f == sg:
                sg.prob_feasible = f.prob_feasible
                sg.exploration_cost = f.exploration_cost
                sg.delta_success_cost = f.delta_success_cost

        if sg.exploration_cost <= 0:
            sg.exploration_cost = -1 * sg.exploration_cost
        if sg.delta_success_cost <= 0:
            sg.delta_success_cost = -1 * sg.delta_success_cost

    return subgoals, sg_not_leading_to_goal


def Q(state, action):
    # If only one frontier is left to explore
    if len(state.Fu) == 1:
        return get_final_state_cost(state)

    new_state_success, new_state_failure, f_I, T_I, goal_reached = move_robots(state, action)

    if goal_reached:
        return T_I

    success_cost = min([Q(new_state_success, action)
                        for action in new_state_success.get_actions()])

    exploration_cost = min([Q(new_state_failure, action)
                            for action in new_state_failure.get_actions()])

    return (T_I + f_I.prob_feasible * success_cost +
            (1 - f_I.prob_feasible) * exploration_cost)


def get_final_state_cost(state):
    '''This is used to compute the cost of state where there are 1 unexplored frontier
    '''
    actions = state.get_actions()
    if (len(state.goal_frontiers) != 0):
        '''In case there are one or more frontier that leads to the goal, there
        are lots of action combination which is possible. After an action is taken,
        and the robots are moved, the new state has no frontiers left to explore. Hence we
        can calculate the cost for all actions in this function alone without use of recursion'''
        final_costs = []
        for action in actions:
            new_state_success, new_state_failure, f_I, T_I, goal_reached = move_robots(state, action)
            # Success: get new state and expected cost
            if goal_reached:
                success_cost = 0
            else:
                success_state_actions = new_state_success.get_actions()
                all_action_cost = []
                for a in success_state_actions:
                    all_robots_cost = []
                    for i, f in enumerate(a):
                        sc = new_state_success.time[f'robot{i+1}'][f] + \
                            (f.delta_success_cost + state.time['goal'][f]) - new_state_success.q_t[i][f]
                        all_robots_cost.append(sc)
                    all_action_cost.append(min(all_robots_cost))
                success_cost = min(all_action_cost)

            # Failure: get new state and expected cost
            if goal_reached:
                exploration_cost = 0
            else:
                failure_state_actions = new_state_failure.get_actions()
                all_action_cost = []
                for a in failure_state_actions:
                    all_robot_cost = []
                    for i, f in enumerate(a):
                        ec = new_state_failure.time[f'robot{i+1}'][f] + \
                            (f.delta_success_cost + state.time['goal'][f]) - new_state_failure.q_t[i][f]
                        all_robot_cost.append(ec)
                    all_action_cost.append(min(all_robot_cost))
                exploration_cost = min(all_action_cost)
            if goal_reached:
                final_costs.append(T_I)
            else:
                cost = T_I + f_I.prob_feasible * success_cost + (1 - f_I.prob_feasible) * exploration_cost
                final_costs.append(cost)

        return min(final_costs)
    else:
        '''
        If no frontier leads to the goal up to this point then the current frontier leads to the goal
        Return the minimum cost (time) for all the robot to reach the goal through that frontier.
        '''
        all_cost = []
        action = actions[0]
        for i, f in enumerate(action):
            sc = state.time[f'robot{i+1}'][f] + (f.delta_success_cost + state.time['goal'][f]) - state.q_t[i][f]
            all_cost.append(sc)
        return min(all_cost)


def find_best_joint_action(num_robots, unexplored_frontiers, time):
    # update_frontier_properties_for_multirobot(unexplored_frontiers)
    sigma = State(n=num_robots, Fu=unexplored_frontiers, m_t=time)
    actions = sigma.get_actions()
    costs = [Q(sigma, action) for action in actions]
    return actions[np.argmin(costs)]
