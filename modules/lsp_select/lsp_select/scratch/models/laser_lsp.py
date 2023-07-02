import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hidden(nn.Module):

    def __init__(self, in_size, out_size):
        super(Hidden, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.batchnorm = nn.BatchNorm1d(out_size)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class LaserLSP(nn.Module):

    def __init__(self):
        super(LaserLSP, self).__init__()
        self.l1 = Hidden(260, 256)
        self.l2 = Hidden(256, 128)
        self.l3 = Hidden(128, 48)
        self.l4 = Hidden(48, 16)
        self.out = nn.Linear(16, 3)

    def forward(self, x, device):
        x = x.to(device)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        out = self.out(x)
        prob_feasible = torch.sigmoid(out[..., 0:1])
        delta_success_cost = out[..., 1:2]
        exploration_cost = out[..., 2:3]
        return prob_feasible, delta_success_cost, exploration_cost

    def loss(self, predicted, target, device='cpu', writer=None, index=None):
        prob_feasible = target[..., 0:1].to(device)
        delta_success_cost = target[..., 1:2].to(device)
        exploration_cost = target[..., 2:3].to(device)
        loss_prob_feasible = 80 * F.binary_cross_entropy(predicted[0], prob_feasible, reduction='mean')
        loss_delta_success_cost = 0.3 * torch.mean(prob_feasible * (predicted[1] - delta_success_cost)**2)
        loss_exploration_cost = 0.01 * torch.mean((1 - prob_feasible) * (predicted[2] - exploration_cost)**2)
        loss_total = loss_prob_feasible + loss_delta_success_cost + loss_exploration_cost

        if writer is not None:
            writer.add_scalar("Loss/total",
                              loss_total.item(),
                              index)
            writer.add_scalar("Loss/prob_feasible",
                              loss_prob_feasible.item(),
                              index)
            writer.add_scalar("Loss/delta_success_cost",
                              loss_delta_success_cost.item(),
                              index)
            writer.add_scalar("Loss/exploration_cost",
                              loss_exploration_cost.item(),
                              index)
        return loss_total

    @classmethod
    def preprocess_data(_, datum, is_training=True):
        datum = np.concatenate([datum['laser_scan'],
                                datum['goal_rel_pos'],
                                datum['frontier_rel_pos']])
        datum = torch.as_tensor(datum[None], dtype=torch.float32)
        return datum

    @classmethod
    def get_net_eval_fn(_, network_file, device):
        model = LaserLSP()
        model.load_state_dict(torch.load(network_file))
        model.eval()
        model.to(device)

        def frontier_net(nn_input_data):
            with torch.no_grad():
                nn_input_data = LaserLSP.preprocess_data(nn_input_data, is_training=False)
                prob_feasible, delta_success_cost, exploration_cost = model(nn_input_data, device=device)
                return prob_feasible.item(), delta_success_cost.item(), exploration_cost.item()

        return frontier_net
