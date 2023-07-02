import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

import lsp_gnn
from lsp_gnn.learning.models.lsp import AbstractLSP


class WallClassGNN(AbstractLSP):
    name = 'GNNforWallClass'

    def __init__(self, args=None):
        super(WallClassGNN, self).__init__(args)

        self.fc1 = nn.Linear(3 + 3, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.conv1 = GATv2Conv(8, 8, edge_dim=1)
        self.conv2 = GATv2Conv(8, 8, edge_dim=1)
        self.conv3 = GATv2Conv(8, 8, edge_dim=1)
        self.conv4 = GATv2Conv(8, 8, edge_dim=1)
        self.classifier = nn.Linear(8, 3)

        self.fc1bn = nn.BatchNorm1d(8)
        self.fc2bn = nn.BatchNorm1d(8)
        self.fc3bn = nn.BatchNorm1d(8)
        self.conv1bn = nn.BatchNorm1d(8)
        self.conv2bn = nn.BatchNorm1d(8)
        self.conv3bn = nn.BatchNorm1d(8)
        self.conv4bn = nn.BatchNorm1d(8)

    def forward(self, data, device):
        h, edge_index, edge_features = self.forward_helper(data, device)
        h = F.leaky_relu(self.fc1bn(self.fc1(h)), 0.1)
        h = F.leaky_relu(self.fc2bn(self.fc2(h)), 0.1)
        h = F.leaky_relu(self.fc3bn(self.fc3(h)), 0.1)
        h = F.leaky_relu(self.conv1bn(self.conv1(h, edge_index, edge_features)), 0.1)
        h = F.leaky_relu(self.conv2bn(self.conv2(h, edge_index, edge_features)), 0.1)
        h = F.leaky_relu(self.conv3bn(self.conv3(h, edge_index, edge_features)), 0.1)
        h = F.leaky_relu(self.conv4bn(self.conv4(h, edge_index, edge_features)), 0.1)
        props = self.classifier(h)
        return props

    @classmethod
    def get_net_eval_fn(_, network_file,
                        device=None):
        model = WallClassGNN()
        model.load_state_dict(torch.load(network_file,
                                         map_location=device))
        model.eval()
        model.to(device)

        def frontier_net(datum, vertex_points, subgoals):
            graph = lsp_gnn.utils.preprocess_gcn_data(datum)
            prob_feasible_dict = {}
            dsc_dict = {}
            ec_dict = {}
            with torch.no_grad():
                out = model.forward(graph, device)
                out = out[:, :3]
                out[:, 0] = torch.sigmoid(out[:, 0])
                out = out.detach().cpu().numpy()
                for subgoal in subgoals:
                    index_pos, possible_node = lsp_gnn.utils. \
                        get_subgoal_node(vertex_points, subgoal)
                    # Extract subgoal properties for a subgoal
                    subgoal_props = out[index_pos]
                    prob_feasible_dict[subgoal] = subgoal_props[0]
                    dsc_dict[subgoal] = subgoal_props[1]
                    ec_dict[subgoal] = subgoal_props[2]
                return prob_feasible_dict, dsc_dict, ec_dict, out[:, 0]

        return frontier_net
