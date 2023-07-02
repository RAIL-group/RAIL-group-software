import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import textwrap

import gridmap
from gridmap.constants import COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL

# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


class SubgoalDiff(object):
    def __init__(self, subgoal_prop_base, subgoal_prop_upd):
        self.prob_feasible_diff = (subgoal_prop_upd.prob_feasible -
                                   subgoal_prop_base.prob_feasible).item()
        self.delta_success_cost_diff = (
            subgoal_prop_upd.delta_success_cost -
            subgoal_prop_base.delta_success_cost).item()
        self.exploration_cost_diff = (
            subgoal_prop_upd.exploration_cost -
            subgoal_prop_base.exploration_cost).item()


class Explanation(object):
    def __init__(self,
                 subgoals,
                 subgoal_ind_dict,
                 base_datum,
                 base_subgoal_props,
                 updated_datum,
                 updated_subgoal_props,
                 delta_subgoal_data,
                 partial_grid,
                 inflated_partial_grid,
                 goal_pose,
                 robot_pose,
                 limit_num=-1):
        # Want to store the subgoal properties and datum before and after the
        # updates. Also store the 'delta_subgoal_data' which will tell us which
        # of the subgoal properties we allowed the derivative to update. Also
        # the map, from which it can compute the paths.
        self.subgoals = subgoals
        self.subgoal_ind_dict = subgoal_ind_dict
        self.base_datum = base_datum
        self.base_subgoal_props = base_subgoal_props
        self.updated_datum = updated_datum
        self.updated_subgoal_props = updated_subgoal_props
        self.delta_subgoal_data = delta_subgoal_data
        self.partial_grid = partial_grid
        self.inflated_partial_grid = inflated_partial_grid
        self.goal_pose = goal_pose
        self.robot_pose = robot_pose
        self.limit_num = limit_num

    def get_subgoal_prop_changes(self):
        return {
            k: SubgoalDiff(self.base_subgoal_props[k],
                           self.updated_subgoal_props[k])
            for k in self.base_subgoal_props.keys()
        }

    def visualize(self, plot_name=None, show_plot=False):
        # Some setup
        ind_p_ind_dict = {}
        prop_changes = self.get_subgoal_prop_changes()
        num_subgoals = len(
            self.updated_datum['target_subgoal_policy']['policy'])
        import matplotlib.cm
        cmap = matplotlib.cm.get_cmap('viridis')
        colors = [cmap(ii / num_subgoals) for ii in range(num_subgoals)]

        print("Base policy for chosen subgoal:")
        for p_ind, ind in enumerate(
                self.base_datum['backup_subgoal_policy']['policy']):
            if (ind, 'prob_feasible') not in self.delta_subgoal_data.keys():
                print("ERROR: dsd")
                continue

            ind_p_ind_dict[ind] = p_ind
            sp_base = self.base_subgoal_props[ind]

            print(
                f"ind: {p_ind:3d} "
                f"| Ps: {sp_base.prob_feasible:.4f} "
                f"[D: {prop_changes[ind].prob_feasible_diff:4f}] "
                f"| rank: {self.delta_subgoal_data[(ind, 'prob_feasible')].rank:3d}"
            )
            print(
                f"         "
                f"| Rs: {sp_base.delta_success_cost:.4f} "
                f"[D: {prop_changes[ind].delta_success_cost_diff:4f}] "
                f"| rank: {self.delta_subgoal_data[(ind, 'delta_success_cost')].rank:3d}"
            )
            print(
                f"         "
                f"| Re: {sp_base.exploration_cost:.4f} "
                f"[D: {prop_changes[ind].exploration_cost_diff:4f}] "
                f"| rank: {self.delta_subgoal_data[(ind, 'exploration_cost')].rank:3d}"
            )

        print("Updated policy for query subgoal.")
        for ind in self.updated_datum['target_subgoal_policy']['policy']:
            if (ind, 'prob_feasible') not in self.delta_subgoal_data.keys():
                print("ERROR: dsd")
                continue

            if ind in ind_p_ind_dict.keys():
                p_ind = ind_p_ind_dict[ind]
            else:
                p_ind = len(ind_p_ind_dict.keys())
                ind_p_ind_dict[ind] = p_ind
            sp_upd = self.updated_subgoal_props[ind]

            dsd_ps = self.delta_subgoal_data[(ind, 'prob_feasible')]
            dsd_rs = self.delta_subgoal_data[(ind, 'delta_success_cost')]
            dsd_re = self.delta_subgoal_data[(ind, 'exploration_cost')]

            print(f"ind: {p_ind:3d} "
                  f"| Ps: {sp_upd.prob_feasible:.4f} "
                  f"| rank: {dsd_ps.rank:3d} [{dsd_ps.rank < self.limit_num}] "
                  f"[{dsd_ps.net_data_cost_fraction:4f}]")
            print(f"         "
                  f"| Rs: {sp_upd.delta_success_cost:.4f} "
                  f"| rank: {dsd_rs.rank:3d} [{dsd_rs.rank < self.limit_num}] "
                  f"[{dsd_rs.net_data_cost_fraction:4f}]")
            print(f"         "
                  f"| Re: {sp_upd.exploration_cost:.4f} "
                  f"| rank: {dsd_re.rank:3d} [{dsd_re.rank < self.limit_num}] "
                  f"[{dsd_re.net_data_cost_fraction:4f}]")

        # Now show the subgoal image data
        height = 1.25 * num_subgoals + 0 * 10 + 2
        bsp_ind = 1
        tsp_ind = 2
        if show_plot:
            fig = plt.figure(dpi=300, figsize=(25 / 4, height / 4))
        else:
            fig = plt.figure(dpi=300, figsize=(25, height))

        # Initialize the plots
        gspec_left = plt.GridSpec(ncols=4,
                                  nrows=num_subgoals + 1,
                                  hspace=0.4,
                                  width_ratios=[1.0, 0.5, 0.5, 1.0])
        gspec_right = gspec_left
        gspec_maps = plt.GridSpec(ncols=4,
                                  nrows=num_subgoals + 1,
                                  hspace=0.1,
                                  wspace=0.1,
                                  width_ratios=[1, 0.9, 0.9, 1])
        axs = [[[] for _ in range(4)] for _ in range(num_subgoals)]

        do_include_policy = True

        # Add the map plots
        ax = fig.add_subplot(gspec_maps[:-1, 0])
        self._add_map_plot(ax, colors, 'base', do_include_policy)
        ax = fig.add_subplot(gspec_maps[:-1, -1])
        self._add_map_plot(ax, colors, 'updated', do_include_policy)
        # Base plots
        for ii, ind in enumerate(
                self.base_datum['backup_subgoal_policy']['policy']):
            ax = fig.add_subplot(gspec_left[ii, bsp_ind])
            axs[ii][bsp_ind] = ax
            self._fill_subgoal_subplot(ax,
                                       ind,
                                       ii,
                                       f"{str(ii)}",
                                       color=colors[ii],
                                       data_src='base',
                                       do_include_policy=do_include_policy)

        # Updated plots
        for ii, ind in enumerate(
                self.updated_datum['target_subgoal_policy']['policy']):
            ax = fig.add_subplot(gspec_right[ii, tsp_ind])
            axs[ii][tsp_ind] = ax
            self._fill_subgoal_subplot(ax,
                                       ind,
                                       ii,
                                       f"{chr(ii+97)}",
                                       color=colors[ii],
                                       data_src='updated',
                                       do_include_policy=do_include_policy)

        if do_include_policy:
            self._draw_policy_annotations('base', bsp_ind, colors, fig, axs)
            self._draw_policy_annotations('updated', tsp_ind, colors, fig, axs)

        # Now we draw lines between the different plots
        # First, the lines across the different policies
        for ii, ind in enumerate(
                self.updated_datum['target_subgoal_policy']['policy']):
            if not do_include_policy:
                break

            p_ind = ind_p_ind_dict[ind]
            ax_base = axs[p_ind][bsp_ind]
            ax_upd = axs[ii][tsp_ind]
            con = ConnectionPatch(xyA=(1.0, 0.5),
                                  coordsA=ax_base.transAxes,
                                  xyB=(0.0, 0.5),
                                  coordsB=ax_upd.transAxes)
            fig.add_artist(con)

        # == Generate plain text ==
        # Sort the subgoals by rank
        ordered_subgoal_inds = [
            ind
            for ind in self.updated_datum['target_subgoal_policy']['policy']
        ]
        ordered_subgoal_inds.sort(key=lambda ind: min([
            self.delta_subgoal_data[(ind, 'prob_feasible')].rank, self.
            delta_subgoal_data[(ind, 'exploration_cost')].rank, self.
            delta_subgoal_data[(ind, 'delta_success_cost')].rank
        ]))

        sc_dict = {
            ind: self.updated_datum['target_subgoal_policy']
            ['success_distances'][ii]
            for ii, ind in enumerate(
                self.updated_datum['target_subgoal_policy']['policy'])
        }

        p_base_ind = ind_p_ind_dict[self.base_datum['backup_subgoal_policy']
                                    ['policy'][0]]
        p_target_ind = ind_p_ind_dict[self.base_datum['target_subgoal_policy']
                                      ['policy'][0]]

        explanation = []
        for ind in ordered_subgoal_inds:
            p_ind = ind_p_ind_dict[ind]
            # Prob Feasible

            dsd_ps = self.delta_subgoal_data[(ind, 'prob_feasible')]
            dsd_rs = self.delta_subgoal_data[(ind, 'delta_success_cost')]
            dsd_re = self.delta_subgoal_data[(ind, 'exploration_cost')]

            sp_base = self.base_subgoal_props[ind]
            sp_updated = self.updated_subgoal_props[ind]
            sp_changes = prop_changes[ind]

            ps_rank = self.delta_subgoal_data[(ind, 'prob_feasible')].rank
            re_rank = self.delta_subgoal_data[(ind, 'exploration_cost')].rank
            rs_rank = self.delta_subgoal_data[(ind, 'delta_success_cost')].rank

            if min([ps_rank, re_rank, rs_rank]) >= self.limit_num:
                continue

            # Prob feasible
            ps_base = sp_base.prob_feasible
            ps_updated = sp_updated.prob_feasible
            dps = sp_changes.prob_feasible_diff
            if abs(dps) <= 0.1:
                adj_str = "slightly "
            elif abs(dps) >= 0.4:
                adj_str = "significantly "
            else:
                adj_str = ""
            change_str = "more likely" if dps > 0 else "less likely"
            change_val_str = "up from" if dps > 0 else "down from"
            ps_dat = [
                ps_rank,
                (f"that Subgoal {p_ind} were {adj_str}{change_str} to lead to "
                 f"the goal ({100*ps_updated:2.0f}%, "
                 f"{change_val_str} {100*ps_base:2.0f}%)")
            ]

            # Delta Success Cost
            sc = sc_dict[ind]
            rs_base = sp_base.delta_success_cost + sc
            rs_updated = sp_updated.delta_success_cost + sc
            drs = sp_changes.delta_success_cost_diff
            if abs(dps) <= 10:
                adj_str = "slightly "
            elif abs(dps) >= 50:
                adj_str = "significantly "
            else:
                adj_str = ""
            change_str = "increased" if drs > 0 else "decreased"
            change_val_str = "up from" if drs > 0 else "down from"
            rs_dat = [
                rs_rank,
                (f"that the cost of getting to the goal through "
                 f"Subgoal {p_ind} were {adj_str}{change_str} "
                 f"({rs_updated/10:.1f} meters, "
                 f"{change_val_str} {rs_base/10:.1f} meters)")
            ]

            # Exploration Cost
            re_base = sp_base.exploration_cost
            re_updated = sp_updated.exploration_cost
            dre = sp_changes.exploration_cost_diff
            if abs(dps) <= 10:
                adj_str = "slightly "
            elif abs(dps) >= 50:
                adj_str = "significantly "
            else:
                adj_str = ""
            change_str = "increased" if dre > 0 else "decreased"
            change_val_str = "up from" if dre > 0 else "down from"
            re_dat = [
                re_rank,
                (f"that the cost of revealing a dead end beyond "
                 f"Subgoal {p_ind} were {adj_str}{change_str} "
                 f"({re_updated/10:.1f} meters, "
                 f"{change_val_str} {re_base/10:.1f} meters)")
            ]

            explain_components = [ps_dat, rs_dat, re_dat]
            explain_components.sort(key=lambda ec: ec[0])
            explain_components = [
                ec[1] for ec in explain_components if ec[0] < self.limit_num
            ]
            explanation.append(" and ".join(explain_components))

        explanation = (f"I would have prefered Subgoal {p_target_ind} " +
                       f"over Subgoal {p_base_ind} if I instead believed " +
                       "; and believed ".join(explanation) + ".")
        ax = fig.add_subplot(gspec_maps[-1, :])
        ax.set_xlim([0, 1])
        ax.set_axis_off()
        explanation = textwrap.fill(explanation.replace("%", "\\%"), width=120)
        ax.text(0.1,
                0.9,
                explanation,
                ha='left',
                va='top',
                wrap=True,
                fontsize=15)

        if plot_name is not None:
            print(f"Saving figure: {plot_name}")
            plt.savefig(plot_name)
        elif show_plot:
            plt.show()

    def _add_map_plot(self, ax, colors, policy_name, do_include_policy=True):
        # Set up
        ind_subgoal_dict = {ind: s for s, ind in self.subgoal_ind_dict.items()}
        if policy_name == 'updated':
            policy = self.updated_datum['target_subgoal_policy']['policy']
            subgoal_props = self.updated_subgoal_props
        elif policy_name == 'base':
            policy = self.base_datum['backup_subgoal_policy']['policy']
            subgoal_props = self.base_subgoal_props
        else:
            raise ValueError('policy_name not supported.')

        plotting_grid = np.zeros(self.partial_grid.shape)
        plotting_grid[self.partial_grid == COLLISION_VAL] = 0.0
        plotting_grid[self.partial_grid == FREE_VAL] = 1.0
        plotting_grid[self.partial_grid == UNOBSERVED_VAL] = 0.7
        ax.imshow(plotting_grid, origin='lower', cmap='gray')
        ax.set_title("The robot's partial map of the world\n"
                     "Robot (R), Goal (G), and Subgoals\n"
                     "Also includes the agent's plan.")

        # Plot the paths
        planning_grid = self.inflated_partial_grid.copy()
        planning_grid[self.partial_grid == UNOBSERVED_VAL] = COLLISION_VAL
        for subgoal in self.subgoals:
            planning_grid[subgoal.points[0, :],
                          subgoal.points[1, :]] = FREE_VAL

        poses = (
            [[self.robot_pose.x, self.robot_pose.y]] +
            [ind_subgoal_dict[ind].get_frontier_point() for ind in policy])
        probs = [subgoal_props[ind].prob_feasible for ind in policy]
        path_dat = []
        cpd = 1.0
        cpds = []
        for ii, (ps, pe, prob) in enumerate(zip(poses, poses[1:], probs)):
            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                planning_grid, [pe[0], pe[1]], use_soft_cost=True)
            did_plan, path = get_path([ps[0], ps[1]],
                                      do_sparsify=True,
                                      do_flip=True)
            path_dat.append([path, colors[ii], float(cpd)])
            cpd *= (1 - prob)
            cpds.append(float(cpd))

        # Plot in reverse to 'first' paths go 'on top'
        if not do_include_policy:
            path_dat = path_dat[:1]
            path_dat[0][2] = 1.0

        for path, color, cpd in path_dat[::-1]:
            if len(path) < 2:
                continue

            ax.plot(path[1, :] - 0.5,
                    path[0, :] - 0.5,
                    color=color,
                    alpha=0.8,
                    linewidth=self._get_linewidth_from_cpd(cpd))

        # Plot the paths to the goal
        planning_grid = self.inflated_partial_grid.copy()
        planning_grid[self.partial_grid == FREE_VAL] = COLLISION_VAL
        for subgoal in self.subgoals:
            planning_grid[subgoal.points[0, :],
                          subgoal.points[1, :]] = FREE_VAL

        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            planning_grid, [self.goal_pose.x, self.goal_pose.y],
            use_soft_cost=True)
        path_dat = []

        for color, cpd, ps in zip(colors, cpds, poses[1:]):
            did_plan, path = get_path([ps[0], ps[1]],
                                      do_sparsify=True,
                                      do_flip=False)
            path_dat.append([path, color, cpd])

        # Plot in reverse to 'first' paths go 'on top'
        if not do_include_policy:
            path_dat = path_dat[:1]
            path_dat[0][2] = 1.0

        for path, color, cpd in path_dat[::-1]:
            if len(path) < 2:
                continue
            ax.plot(path[1, :] - 0.5,
                    path[0, :] - 0.5,
                    color=color,
                    alpha=0.6,
                    linestyle='dotted',
                    linewidth=self._get_linewidth_from_cpd(cpd))

        # Plot the badges
        ax.text(self.robot_pose.y,
                self.robot_pose.x,
                "R",
                bbox={
                    "boxstyle": "round",
                    "edgecolor": "black",
                    "facecolor": "white",
                    "linewidth": 2
                },
                ha='center',
                va='center',
                transform=ax.transData)
        ax.text(self.goal_pose.y,
                self.goal_pose.x,
                "G",
                bbox={
                    "boxstyle": "round",
                    "edgecolor": "black",
                    "facecolor": "green",
                    "linewidth": 2
                },
                ha='center',
                va='center',
                transform=ax.transData)
        for ii, ind in enumerate(policy):
            subgoal = ind_subgoal_dict[ind]
            s_cent = subgoal.get_frontier_point()
            label = chr(ii + 97) if policy_name == 'updated' else str(ii)
            ax.text(s_cent[1],
                    s_cent[0],
                    label,
                    bbox={"boxstyle": "round",
                          "edgecolor": colors[ii],
                          "facecolor": "white",
                          "linewidth": 2},
                    transform=ax.transData)

    def _fill_subgoal_subplot(self, ax, ind, num, label, color, data_src,
                              do_include_policy):
        ind_subgoal_dict = {ind: s for s, ind in self.subgoal_ind_dict.items()}
        subgoal = ind_subgoal_dict[ind]

        # Add the image
        ax.imshow(subgoal.nn_input_data['image'])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        sp_changes = self.get_subgoal_prop_changes()

        if data_src == 'base':
            offset = -0.55
            sp = self.base_subgoal_props[ind]
            sc = self.base_datum['backup_subgoal_policy']['success_distances'][
                num]
        elif data_src == 'updated':
            offset = 1.05
            sp = self.updated_subgoal_props[ind]
            sc = self.updated_datum['target_subgoal_policy'][
                'success_distances'][num]
        else:
            raise ValueError("'data_src' must be either 'base' or 'updated'.")

        ax.plot([0.5, 0.5], [1.0, 0.0],
                transform=ax.transAxes,
                color=color,
                alpha=0.3)
        ax.text(0.5,
                0.95,
                label,
                bbox={
                    "boxstyle": "round",
                    "edgecolor": color,
                    "facecolor": "white",
                    "linewidth": 2
                },
                ha='center',
                va='center',
                transform=ax.transAxes)

        if not do_include_policy:
            return

        if self.limit_num > 0:
            is_ps_diff = self.delta_subgoal_data[(
                ind, 'prob_feasible')].rank < self.limit_num
            is_rs_diff = self.delta_subgoal_data[(
                ind, 'delta_success_cost')].rank < self.limit_num
            is_re_diff = self.delta_subgoal_data[(
                ind, 'exploration_cost')].rank < self.limit_num
        else:
            is_ps_diff = True
            is_rs_diff = True
            is_re_diff = True

        if is_ps_diff:
            ps_diff = sp_changes[ind].prob_feasible_diff
            ps_str = f"$\\mathbf{{ P_s = {sp.prob_feasible:0.3f} }}$"
            if ps_diff >= 0 and data_src == 'base':
                ps_str += f"\\ [$\\Delta\\uparrow${abs(ps_diff):0.3f}]"
            elif ps_diff < 0 and data_src == 'base':
                ps_str += f"\\ [$\\Delta\\downarrow${abs(ps_diff):0.3f}]"
        else:
            ps_str = f"$P_s = {sp.prob_feasible:0.3f}$"

        if is_rs_diff:
            rs_diff = sp_changes[ind].delta_success_cost_diff
            rs_str = f"$\\mathbf{{ R_s = {sp.delta_success_cost + sc:0.2f} }}$"
            if rs_diff >= 0 and data_src == 'base':
                rs_str += f"\\ [$\\Delta\\uparrow${abs(rs_diff):0.2f}]"
            elif rs_diff < 0 and data_src == 'base':
                rs_str += f"\\ [$\\Delta\\downarrow${abs(rs_diff):0.2f}]"
        else:
            rs_str = f"$R_s = {sp.delta_success_cost + sc:0.2f}$"

        if is_re_diff:
            re_diff = sp_changes[ind].exploration_cost_diff
            re_str = f"$\\mathbf{{ R_e = {sp.exploration_cost:0.2f} }}$"
            if re_diff >= 0 and data_src == 'base':
                re_str += f"\\ [$\\Delta\\uparrow${abs(re_diff):0.2f}]"
            elif re_diff < 0 and data_src == 'base':
                re_str += f"\\ [$\\Delta\\downarrow${abs(re_diff):0.2f}]"
        else:
            re_str = f"$R_e = {sp.exploration_cost:0.2f}$"

        head_str = f"Subgoal: $s_{label}$"

        plt.text(offset, 0.85, head_str, transform=ax.transAxes)
        plt.text(offset, 0.6, ps_str, transform=ax.transAxes)
        plt.text(offset, 0.4, rs_str, transform=ax.transAxes)
        plt.text(offset, 0.2, re_str, transform=ax.transAxes)

    def _draw_policy_annotations(self, policy_name, policy_ind, colors, fig,
                                 axs):
        # Next draw the annotations between each set of plots
        if policy_name == 'base':
            policy_data = self.base_datum['backup_subgoal_policy']
            subgoal_props = self.base_subgoal_props
        else:
            policy_data = self.updated_datum['target_subgoal_policy']
            subgoal_props = self.updated_subgoal_props

        policy = policy_data['policy']
        probs = [subgoal_props[ind].prob_feasible for ind in policy]

        cpd = 1.0
        for ii, (cost, prob, color) in enumerate(
                zip(policy_data['failure_distances'], probs, colors[1:])):
            cpd *= (1 - prob)
            # Add the line
            ax_a = axs[ii][policy_ind]
            ax_b = axs[ii + 1][policy_ind]
            con = ConnectionPatch(xyA=(0.1, 0.0),
                                  coordsA=ax_a.transAxes,
                                  xyB=(0.1, 1.0),
                                  coordsB=ax_b.transAxes,
                                  arrowstyle='->',
                                  linewidth=2 *
                                  self._get_linewidth_from_cpd(cpd),
                                  color=color)
            fig.add_artist(con)

            # Now add the text
            xy = 0.5 * (ax_a.transAxes.transform(
                (0.12, 0.0)) + ax_b.transAxes.transform((0.12, 1.0)))
            plt.text(xy[0], xy[1], f"Travel Cost: {cost:0.2f}", transform=None)

    @classmethod
    def _get_linewidth_from_cpd(_, cpd):
        return max(0.5, 2 + np.log10(float(cpd)))
