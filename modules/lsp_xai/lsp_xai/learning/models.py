from collections import namedtuple
import itertools
import numpy as np
import torch
import torch.nn as nn
from learning.logging import tensorboard_plot_decorator

import lsp_xai
from lsp.learning.models.shared import EncoderNBlocks


class Subgoal(object):
    def __init__(self, prob_feasible, delta_success_cost, exploration_cost,
                 id):
        self.prob_feasible = prob_feasible
        self.delta_success_cost = delta_success_cost
        self.exploration_cost = exploration_cost
        self.id = int(id)
        self.is_from_last_chosen = False

    def __hash__(self):
        return self.id


SubgoalPropDat = namedtuple('SubgoalPropDat', [
    'ind', 'prop_name', 'delta', 'weight', 'delta_cost', 'delta_cost_fraction',
    'net_data_cost_fraction', 'rank'
])


def compute_subgoal_props(is_feasibles,
                          delta_success_costs,
                          exploration_costs,
                          subgoal_data,
                          ind_mapping,
                          device='cpu',
                          callback_dict=None,
                          limit_subgoals_num=-1,
                          delta_subgoal_data=None):
    # Populate variables of relevance
    subgoal_props = {}

    delta_success_neg = []
    exploration_neg = []

    softplus = torch.nn.Softplus()

    if callback_dict is not None:

        def save_grad(ind, name):
            def hook(grad):
                callback_dict[(ind, name)] = grad

            return hook

    # Populate the storage lists
    for ind, subgoal_datum in subgoal_data.items():
        counter = ind_mapping[ind]
        is_feasible = is_feasibles[counter]
        delta_success_cost = delta_success_costs[counter]
        exploration_cost = exploration_costs[counter]
        delta_success_neg.append(softplus(-delta_success_cost).cpu())
        exploration_neg.append(softplus(-exploration_cost).cpu())

        # Detatch if requested
        if limit_subgoals_num >= 0:
            if delta_subgoal_data[(
                    ind, 'prob_feasible')].rank >= limit_subgoals_num:
                is_feasible = is_feasible.detach()
            if delta_subgoal_data[(
                    ind, 'delta_success_cost')].rank >= limit_subgoals_num:
                delta_success_cost = delta_success_cost.detach()
            if delta_subgoal_data[(
                    ind, 'exploration_cost')].rank >= limit_subgoals_num:
                exploration_cost = exploration_cost.detach()

        if callback_dict is not None:
            is_feasible.register_hook(save_grad(ind, 'prob_feasible'))
            delta_success_cost.register_hook(
                save_grad(ind, 'delta_success_cost'))
            exploration_cost.register_hook(save_grad(ind, 'exploration_cost'))

        subgoal_props[ind] = Subgoal(
            prob_feasible=is_feasible.cpu(),
            delta_success_cost=delta_success_cost.cpu(),
            exploration_cost=exploration_cost.cpu(),
            id=ind)

    return subgoal_props, delta_success_neg, exploration_neg


class ExpNavVisLSP(nn.Module):
    name = "ExpNavVisLSP"

    def __init__(self, args=None, num_outputs=3):
        super(ExpNavVisLSP, self).__init__()

        # Store arguments
        self._args = args

        # Initialize the blocks
        self.enc_1 = EncoderNBlocks(3, 16, num_layers=2)
        self.enc_2 = EncoderNBlocks(16, 16, num_layers=2)
        self.enc_3 = EncoderNBlocks(16 + 4, 32, num_layers=2)
        self.enc_4 = EncoderNBlocks(32, 32, num_layers=2)
        self.enc_5 = EncoderNBlocks(32, 32, num_layers=2)
        self.enc_6 = EncoderNBlocks(32, 32, num_layers=2)

        # Initialize remaining layers
        self.conv_1x1 = nn.Conv2d(32, 4, kernel_size=1, bias=False)
        self.fc_outs = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.01),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32, momentum=0.01),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 16, bias=False),
            nn.BatchNorm1d(16, momentum=0.01),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, num_outputs),
        )

        # Not used; remains for backwards compatability reasons
        self.goal_bn = nn.BatchNorm2d(2, momentum=0.001)

    def forward(self, batch, device='cpu'):
        ind_mapping = {}
        for counter, ind in enumerate(batch['subgoal_data'].keys()):
            ind_mapping[ind] = counter

        # Initialize the temporary storage lists
        num_subgoals = len(ind_mapping.keys())
        images = [None] * num_subgoals
        goal_loc_xs = [None] * num_subgoals
        goal_loc_ys = [None] * num_subgoals
        subgoal_loc_xs = [None] * num_subgoals
        subgoal_loc_ys = [None] * num_subgoals

        # Populate the storage lists
        try:
            is_numpy = (len(batch['subgoal_data'][ind]['image'].shape) == 3)
        except KeyError:
            is_numpy = False

        if is_numpy:
            for ind, subgoal_datum in batch['subgoal_data'].items():
                counter = ind_mapping[ind]
                images[counter] = torch.Tensor(subgoal_datum['image'])
                goal_loc_xs[counter] = torch.Tensor(
                    subgoal_datum['goal_loc_x'])
                goal_loc_ys[counter] = torch.Tensor(
                    subgoal_datum['goal_loc_y'])
                subgoal_loc_xs[counter] = torch.Tensor(
                    subgoal_datum['subgoal_loc_x'])
                subgoal_loc_ys[counter] = torch.Tensor(
                    subgoal_datum['subgoal_loc_y'])

            processed_batch = {
                'image': torch.stack(images, dim=0).permute(
                    (0, 3, 1, 2)).float(),
                'goal_loc_x': torch.stack(goal_loc_xs, dim=0).float(),
                'goal_loc_y': torch.stack(goal_loc_ys, dim=0).float(),
                'subgoal_loc_x': torch.stack(subgoal_loc_xs, dim=0).float(),
                'subgoal_loc_y': torch.stack(subgoal_loc_ys, dim=0).float(),
            }
        else:
            for ind, subgoal_datum in batch['subgoal_data'].items():
                counter = ind_mapping[ind]
                images[counter] = subgoal_datum['image']
                goal_loc_xs[counter] = subgoal_datum['goal_loc_x']
                goal_loc_ys[counter] = subgoal_datum['goal_loc_y']
                subgoal_loc_xs[counter] = subgoal_datum['subgoal_loc_x']
                subgoal_loc_ys[counter] = subgoal_datum['subgoal_loc_y']

            processed_batch = {
                'image': torch.cat(images, dim=0).permute(
                    (0, 3, 1, 2)).float(),
                'goal_loc_x': torch.cat(goal_loc_xs, dim=0).float(),
                'goal_loc_y': torch.cat(goal_loc_ys, dim=0).float(),
                'subgoal_loc_x': torch.cat(subgoal_loc_xs, dim=0).float(),
                'subgoal_loc_y': torch.cat(subgoal_loc_ys, dim=0).float()
            }

        return self.forward_supervised(processed_batch, device), ind_mapping

    def forward_supervised(self, data, device):
        image = data['image'].to(device)

        # Compute goal info tensor
        if 'goal_loc_x' in data.keys():
            g = torch.stack(
                (data['goal_loc_x'], data['goal_loc_y']), 1).expand(
                    [-1, -1, 32, -1]).float().to(device) / 100.0
            s = torch.stack(
                (data['subgoal_loc_x'], data['subgoal_loc_y']), 1).expand(
                    [-1, -1, 32, -1]).float().to(device) / 100.0
        else:
            raise ValueError("Missing goal location data.")

        # Encoding layers
        x = image
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = torch.cat((x, g, s), 1)  # Add the goal info tensor
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)
        x = self.enc_6(x)
        x = self.conv_1x1(x)
        x = self.fc_outs(x)

        return x

    def compute_subgoal_props(_,
                              is_feasible_mat,
                              delta_success_cost_mat,
                              exploration_cost_mat,
                              subgoal_data,
                              ind_mapping,
                              device='cpu',
                              callback_dict=None,
                              limit_subgoals_num=-1,
                              delta_subgoal_data=None):
        return compute_subgoal_props(is_feasible_mat, delta_success_cost_mat,
                                     exploration_cost_mat, subgoal_data,
                                     ind_mapping, device, callback_dict,
                                     limit_subgoals_num, delta_subgoal_data)

    def get_subgoal_prop_impact(self,
                                datum,
                                device='cpu',
                                delta_cost_limit=None):
        # Initialize the simple SGD optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-6)

        # Put network in eval mode if necessary
        is_in_training_mode = self.training
        if is_in_training_mode:
            self.eval()

        # Compute the base subgoal properties
        out, ind_map = self(datum, device)
        nn_out_processed = out[:, :3]
        is_feasible_mat = torch.nn.Sigmoid()(nn_out_processed[:, 0])
        delta_success_cost_mat = nn_out_processed[:, 1]
        exploration_cost_mat = nn_out_processed[:, 2]
        callback_dict_base = {}
        subgoal_props_base, _, _ = self.compute_subgoal_props(
            is_feasible_mat, delta_success_cost_mat, exploration_cost_mat,
            datum['subgoal_data'], ind_map, device, callback_dict_base)

        target_cost = lsp_xai.utils.data.compute_expected_cost_for_policy(
            subgoal_props_base, datum['target_subgoal_policy'])

        backup_cost = lsp_xai.utils.data.compute_expected_cost_for_policy(
            subgoal_props_base, datum['backup_subgoal_policy'])

        delta_cost = target_cost - backup_cost

        # If the target cost is sufficiently below the backup cost
        # don't compute the subgoal properties.
        if delta_cost_limit is not None and delta_cost < -np.abs(
                delta_cost_limit):
            print(f"Bypassing subgoal prop limiting: {delta_cost}.")
            spd_list = [
                SubgoalPropDat(ind, prop_name, 0, 0, 0, 0, 0, 0) for ind,
                prop_name in itertools.product(subgoal_props_base.keys(
                ), ['prob_feasible', 'delta_success_cost', 'exploration_cost'])
            ]

            return {(spd.ind, spd.prop_name): spd for spd in spd_list}

        # Update the model parameters
        optimizer.zero_grad()
        delta_cost.backward()
        optimizer.step()

        # Now compute the updated subgoal properties
        with torch.no_grad():
            # Compute the base subgoal properties
            out, ind_map = self(datum, device)
            nn_out_processed = out[:, :3]
            is_feasible_mat = torch.nn.Sigmoid()(nn_out_processed[:, 0])
            delta_success_cost_mat = nn_out_processed[:, 1]
            exploration_cost_mat = nn_out_processed[:, 2]
            subgoal_props_upd, _, _ = self.compute_subgoal_props(
                is_feasible_mat, delta_success_cost_mat, exploration_cost_mat,
                datum['subgoal_data'], ind_map, device)

        # Compute the data to be returned
        subgoal_prop_data_list = []
        for ind in subgoal_props_base.keys():
            d = (subgoal_props_upd[ind].prob_feasible -
                 subgoal_props_base[ind].prob_feasible).detach().cpu().numpy()
            w = callback_dict_base[(ind,
                                    'prob_feasible')].detach().cpu().numpy()
            subgoal_prop_data_list.append([ind, 'prob_feasible', d, w, d * w])

            d = (subgoal_props_upd[ind].delta_success_cost -
                 subgoal_props_base[ind].delta_success_cost
                 ).detach().cpu().numpy()
            w = callback_dict_base[(
                ind, 'delta_success_cost')].detach().cpu().numpy()
            subgoal_prop_data_list.append(
                [ind, 'delta_success_cost', d, w, d * w])

            d = (subgoal_props_upd[ind].exploration_cost -
                 subgoal_props_base[ind].exploration_cost
                 ).detach().cpu().numpy()
            w = callback_dict_base[(
                ind, 'exploration_cost')].detach().cpu().numpy()
            subgoal_prop_data_list.append(
                [ind, 'exploration_cost', d, w, d * w])

        # Revert via a backwards gradient step
        optimizer.param_groups[0]['lr'] *= -1
        optimizer.step()
        optimizer.zero_grad()

        # Restore network to training mode if necessary
        if is_in_training_mode:
            self.train()

        # Compute the 'rank' of each term and return
        subgoal_prop_data_list.sort(key=lambda spd: abs(spd[-1]), reverse=True)
        abs_delta_cost = [abs(spd[4]) for spd in subgoal_prop_data_list]
        net_data_cost = list(np.cumsum(abs_delta_cost))
        sc = max(net_data_cost)
        is_rank = 1
        if sc == 0:
            sc = 1.0
            is_rank = 0

        spd_list = [
            SubgoalPropDat(ind=spd[0],
                           prop_name=spd[1],
                           delta=spd[2] / sc,
                           weight=spd[3] / sc,
                           delta_cost=spd[4] / sc,
                           delta_cost_fraction=adc / sc,
                           net_data_cost_fraction=ndc / sc,
                           rank=is_rank * rank)
            for rank, (spd, adc, ndc) in enumerate(
                zip(subgoal_prop_data_list, abs_delta_cost, net_data_cost))
        ]

        return {(spd.ind, spd.prop_name): spd for spd in spd_list}

    def loss(self,
             nn_out,
             batch,
             ind_mapping,
             device='cpu',
             writer=None,
             index=None,
             limit_subgoals_num=-1,
             delta_subgoal_data=None,
             do_include_limit_costs=True,
             do_include_negative_costs=True):

        do_limit = ((limit_subgoals_num >= 0)
                    and delta_subgoal_data is not None
                    and max([dsd.rank
                             for dsd in delta_subgoal_data.values()]) > 0)

        # Separate outputs.
        nn_out_processed = nn_out[:, :3]
        is_feasible_mat = torch.nn.Sigmoid()(nn_out_processed[:, 0])
        delta_success_cost_mat = nn_out_processed[:, 1]
        exploration_cost_mat = nn_out_processed[:, 2]

        limited_subgoal_props, delta_success_neg, exploration_neg = self.compute_subgoal_props(
            is_feasible_mat,
            delta_success_cost_mat,
            exploration_cost_mat,
            batch['subgoal_data'],
            ind_mapping,
            device,
            limit_subgoals_num=limit_subgoals_num,
            delta_subgoal_data=delta_subgoal_data)

        target_cost = lsp_xai.utils.data.compute_expected_cost_for_policy(
            limited_subgoal_props, batch['target_subgoal_policy'])
        backup_cost = lsp_xai.utils.data.compute_expected_cost_for_policy(
            limited_subgoal_props, batch['backup_subgoal_policy'])

        loss_diff = torch.sqrt(1 - torch.nn.LogSigmoid()
                               (-(target_cost - backup_cost))) - 1
        if limit_subgoals_num == 0:
            print("No subgoal properties; no loss_diff.")
            loss = 0 * loss_diff
        else:
            loss = loss_diff

        print(f"Target: {target_cost.cpu().detach().numpy()} < "
              f"Backup: {backup_cost.cpu().detach().numpy()} | "
              f"loss_diff: {loss_diff.cpu().detach().numpy()}")

        # Recompute the subgoal properties with all terms if needed
        if do_limit and do_include_limit_costs:
            subgoal_props, delta_success_neg, exploration_neg = self.compute_subgoal_props(
                is_feasible_mat, delta_success_cost_mat, exploration_cost_mat,
                batch['subgoal_data'], ind_mapping, device)

            target_cost = lsp_xai.utils.data.compute_expected_cost_for_policy(
                subgoal_props, batch['target_subgoal_policy'])
            backup_cost = lsp_xai.utils.data.compute_expected_cost_for_policy(
                subgoal_props, batch['backup_subgoal_policy'])
        else:
            subgoal_props = limited_subgoal_props

        if do_include_limit_costs:
            loss_target_upper = torch.nn.ReLU()(
                (target_cost - batch['net_cost_remaining']))
            loss_target_lower = torch.nn.ReLU()(
                (batch['net_cost_remaining_known'] - target_cost))
            loss = loss + loss_target_upper / 50
            loss = loss + loss_target_lower / 500
            print(f"  Target Upper Bound: {batch['net_cost_remaining_known']}")
        else:
            print("  No limit costs")

        if do_include_negative_costs:
            loss_negative_costs = 0
            loss_negative_costs += sum(delta_success_neg)
            loss_negative_costs += sum(exploration_neg)
            loss = loss + loss_negative_costs / 50
        else:
            print("  No negative costs")

        # Write some things
        if writer is not None and index is not None:
            writer.add_scalar("Loss/compare_total", loss.item(), index)
            writer.add_scalar("Loss/compare_diff", loss_diff.item(), index)

            if delta_subgoal_data is not None:
                for dsd in delta_subgoal_data.values():
                    if dsd.rank == (limit_subgoals_num - 1):
                        writer.add_scalar("Debug/subgoal_limit_fraction",
                                          dsd.net_data_cost_fraction, index)
                    if np.isnan(dsd.net_data_cost_fraction):
                        print(dsd)
                        raise ValueError()

            if do_include_limit_costs:
                writer.add_scalar("Loss/compare_target_upper",
                                  loss_target_upper.item(), index)
                writer.add_scalar("Loss/compare_target_lower",
                                  loss_target_lower.item(), index)

            if do_include_negative_costs:
                writer.add_scalar("Loss/compare_negative_costs",
                                  loss_negative_costs.item(), index)
                writer.add_scalar("Debug/delta_success_neg_avg",
                                  (sum(delta_success_neg) /
                                   len(subgoal_props)).item(), index)
                writer.add_scalar("Debug/exploration_neg_avg",
                                  (sum(exploration_neg) /
                                   len(subgoal_props)).item(), index)

        return loss

    def loss_supervised(self,
                        nn_out,
                        data,
                        device='cpu',
                        writer=None,
                        index=None,
                        positive_weight=None):
        # Separate outputs.
        is_feasible_logits = nn_out[:, 0]
        is_feasible_label = data['is_feasible'].to(device)
        if positive_weight is None:
            positive_weight = 1.0

        # Compute the contribution from the is_feasible_label
        weights = 1.0 + (positive_weight - 1.0) * is_feasible_label
        print("RAND", self._args.learning_seed)
        print("XENT", (1 / 10) * self._args.xent_factor)
        is_feasible_xentropy = (
            1 / 10) * self._args.xent_factor * torch.nn.BCEWithLogitsLoss(
                weight=weights)(is_feasible_logits, is_feasible_label)

        # Sum the contributions
        loss = is_feasible_xentropy

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/supervised_is_feasible_xentropy",
                              is_feasible_xentropy.item(), index)
            writer.add_scalar("Loss/supervised_total", loss.item(), index)

        return loss

    def update_datum(self, datum, device):
        with torch.no_grad():
            out, ind_map = self(datum, device)
            out_det = out.detach()
            is_feasibles = torch.nn.Sigmoid()(out_det[:, 0])
            delta_success_costs = out_det[:, 1]
            exploration_costs = out_det[:, 2]
            subgoal_props, delta_success_neg, exploration_neg = self.compute_subgoal_props(
                is_feasibles, delta_success_costs, exploration_costs,
                datum['subgoal_data'], ind_map, device)

            datum = lsp_xai.utils.data.update_datum_policies(
                subgoal_props, datum)

            if datum is None:
                return None

            relevant_inds = list(
                set(datum['target_subgoal_policy']['policy'])
                | set(datum['backup_subgoal_policy']['policy']))

            datum['subgoal_data'] = {
                ind: sg
                for ind, sg in datum['subgoal_data'].items()
                if ind in relevant_inds
            }

            return datum

    @classmethod
    def compute_expected_cost_for_policy(_, subgoal_props,
                                         subgoal_policy_data):
        return lsp_xai.utils.data.compute_expected_cost_for_policy(
            subgoal_props, subgoal_policy_data)

    @classmethod
    def get_net_eval_fn(_, network_file, device, do_return_model=False):
        model = ExpNavVisLSP()
        if network_file is not None:
            model.load_state_dict(torch.load(network_file,
                                             map_location=torch.device('cpu')),
                                  strict=False)
        model.eval()
        model.to(device)

        def frontier_net(image, goal_loc_x, goal_loc_y, subgoal_loc_x,
                         subgoal_loc_y):
            with torch.no_grad():
                image = np.transpose(image, (2, 0, 1))
                out = model.forward_supervised(
                    {
                        'image':
                        torch.tensor(np.expand_dims(image, axis=0)).float(),
                        'goal_loc_x':
                        torch.tensor(np.expand_dims(goal_loc_x,
                                                    axis=0)).float(),
                        'goal_loc_y':
                        torch.tensor(np.expand_dims(goal_loc_y,
                                                    axis=0)).float(),
                        'subgoal_loc_x':
                        torch.tensor(np.expand_dims(subgoal_loc_x,
                                                    axis=0)).float(),
                        'subgoal_loc_y':
                        torch.tensor(np.expand_dims(subgoal_loc_y,
                                                    axis=0)).float()
                    },
                    device=device)
                out[0, 0] = torch.sigmoid(out[0, 0])
                out = out.detach().cpu().numpy()
                return out[0, 0], out[0, 1], out[0, 2]

        if do_return_model:
            return frontier_net, model
        else:
            return frontier_net

    @classmethod
    def preprocess_data(_, datum, is_training=True):
        datum['image'] = np.transpose(datum['image'],
                                      (2, 0, 1)).astype(np.float32) / 255
        return datum

    @tensorboard_plot_decorator
    def plot_data_supervised(self, fig, out, data):
        image = data['image'][0]
        image = np.transpose(image, (1, 2, 0))
        out_base = out[0, :3]
        is_feasible_base = torch.sigmoid(out_base[0]).cpu().numpy()
        is_feasible_base = is_feasible_base

        axs = fig.subplots(4, 1)
        axs[0].imshow(image, interpolation='none')

    def update_model_counterfactual(self, datum, limit_num,
                                    margin, learning_rate, device, num_steps=5000):
        import copy

        datum = copy.deepcopy(datum)
        delta_subgoal_data = self.get_subgoal_prop_impact(
            datum, device, delta_cost_limit=-1e10)

        # Initialize some terms for the optimization
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # Now we perfrom iterative gradient descent until the expected cost of the
        # new target subgoal is lower than that of the originally selected subgoal.
        for ii in range(5000):
            # Update datum to reflect new neural network state
            datum = self.update_datum(datum, device)

            # Compute the subgoal properties by passing images through the network.
            # (PyTorch implicitly builds a graph of these operations so that we can
            # differentiate them later.)
            nn_out, ind_mapping = self(datum, device)
            is_feasibles = torch.nn.Sigmoid()(nn_out[:, 0])
            delta_success_costs = nn_out[:, 1]
            exploration_costs = nn_out[:, 2]
            limited_subgoal_props, _, _ = self.compute_subgoal_props(
                is_feasibles,
                delta_success_costs,
                exploration_costs,
                datum['subgoal_data'],
                ind_mapping,
                device,
                limit_subgoals_num=limit_num,
                delta_subgoal_data=delta_subgoal_data)

            if ii == 0:
                base_subgoal_props = limited_subgoal_props

            # Compute the expected of the new target subgoal:
            q_target = self.compute_expected_cost_for_policy(
                limited_subgoal_props, datum['target_subgoal_policy'])
            # Cost of the 'backup' (formerly the agent's chosen subgoal):
            q_backup = self.compute_expected_cost_for_policy(
                limited_subgoal_props, datum['backup_subgoal_policy'])
            print(
                f"{ii:5} | Q_dif = {q_target - q_backup:6f} | Q_target = {q_target:6f} | Q_backup = {q_backup:6f}"
            )
            assert q_target > 0
            assert q_backup > 0

            # The zero-crossing of the difference between the two is the decision
            # boundary we are hoping to cross by updating the paramters of the
            # neural network via gradient descent.
            q_diff = q_target - q_backup

            if q_diff <= -margin:
                # When it's less than zero, we're done.
                break

            # Via PyTorch magic, gradient descent is easy:
            optimizer.zero_grad()
            q_diff.backward()
            optimizer.step()
        else:
            # If it never crossed the boundary, we have failed.
            raise ValueError("Decision boundary never crossed.")

        upd_subgoal_props = limited_subgoal_props

        return (datum, delta_subgoal_data, base_subgoal_props,
                upd_subgoal_props)
