import torch
import torch.nn as nn
import torch_geometric.utils
import torch.nn.functional as F


class AbstractLSP(nn.Module):
    name = 'AbstractClassForLSP'

    def __init__(self, args=None):
        super(AbstractLSP, self).__init__()
        torch.manual_seed(8616)
        self._args = args

    def forward(self, data, device):
        raise NotImplementedError

    def forward_helper(self, data, device, cnn=False):
        lf = data['latent_features'].type(torch.float).to(device)
        edge_data = data['edge_data']
        x = torch.cat((edge_data[0], edge_data[1]), 0)
        y = torch.cat((edge_data[1], edge_data[0]), 0)
        edge_data = torch.reshape(torch.cat((x, y), 0), (2, -1))
        edge_index = edge_data.to(device)
        edge_features = data['edge_features'].type(torch.float).to(device)
        edge_features = edge_features.repeat(2, 1).to(device) / 200

        distance = data['goal_distance'].view(-1, 1).to(device) / 200
        history = data['history'].view(-1, 1).to(device)
        is_subgoal = data['is_subgoal'].view(-1, 1).to(device)
        degree = torch_geometric.utils.degree(edge_index[0]).view(-1, 1).to(device)
        degree[history == 1] = 0
        degree = degree / 4
        if cnn:
            h = torch.cat((lf, is_subgoal, history, distance, degree), 1)
        else:
            h = torch.cat((lf, is_subgoal, history, degree), 1)
        return h, edge_index, edge_features

    def loss(self, nn_out, data, device='cpu', writer=None, index=None):
        # Separate outputs.
        is_feasible_logits = nn_out[:, 0]
        delta_cost_pred = nn_out[:, 1]
        exploration_cost_pred = nn_out[:, 2]

        # Convert the data
        is_feasible_label = data.y.to(device)
        delta_cost_label = data.dsc.to(device)
        exploration_cost_label = data.ec.to(device)
        pweight = data.pweight.to(device)  # TODO - Remove?
        nweight = data.nweight.to(device)  # TODO - Remove?
        history = data.is_subgoal.to(device)
        rpw = self._args.relative_positive_weight
        subgoal_weight = history  # * (has_updated + 0.1)

        # Compute the contribution from the is_feasible_label
        is_feasible_xentropy = rpw * is_feasible_label * -F.logsigmoid(
            is_feasible_logits) * pweight / 10 + (1 - is_feasible_label) * \
            -F.logsigmoid(-is_feasible_logits) * nweight / 10
        is_feasible_xentropy = torch.sum(subgoal_weight * is_feasible_xentropy)
        is_feasible_xentropy /= torch.sum(subgoal_weight) + 0.001
        # Set the loss type for Delta Success Cost and Exploration Cost
        if self._args.loss == 'l1':
            cost_loss = torch.abs
        else:
            cost_loss = torch.square

        # Delta Success Cost
        delta_cost_pred_error = cost_loss(
            delta_cost_pred - delta_cost_label) \
            / (10 ** 1) * is_feasible_label
        delta_cost_pred_error = torch.sum(subgoal_weight * delta_cost_pred_error)
        delta_cost_pred_error /= torch.sum(subgoal_weight) + 0.001
        # Exploration Cost
        exploration_cost_pred_error = cost_loss(
            exploration_cost_pred - exploration_cost_label) / \
            (20 ** 1 * 4) * (1 - is_feasible_label)
        exploration_cost_pred_error = torch.sum(subgoal_weight * exploration_cost_pred_error)
        exploration_cost_pred_error /= torch.sum(subgoal_weight) + 0.001

        # Sum the contributions
        loss = is_feasible_xentropy + delta_cost_pred_error + \
            exploration_cost_pred_error

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/is_feasible_xentropy",
                              is_feasible_xentropy.item(),
                              index)
            writer.add_scalar("Loss/delta_success_cost_loss",
                              delta_cost_pred_error.item(),
                              index)
            writer.add_scalar("Loss/exploration_cost_loss",
                              exploration_cost_pred_error.item(),
                              index)
            writer.add_scalar("Loss/total_loss",
                              loss.item(),
                              index)

        return loss
