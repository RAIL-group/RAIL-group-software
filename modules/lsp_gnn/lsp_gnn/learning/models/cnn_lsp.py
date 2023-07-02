import torch
import torch.nn as nn
import torch.nn.functional as F

import lsp_gnn
from lsp_gnn.learning.models.lsp import AbstractLSP


class WallClassLSP(AbstractLSP):
    name = 'LSPforWallClass'

    def __init__(self, args=None):
        super(WallClassLSP, self).__init__(args)

        self.fc1 = nn.Linear(3 + 4, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 8)
        self.fc5 = nn.Linear(8, 8)
        self.fc6 = nn.Linear(8, 8)
        self.fc7 = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 3)

        self.fc1bn = nn.BatchNorm1d(8)
        self.fc2bn = nn.BatchNorm1d(8)
        self.fc3bn = nn.BatchNorm1d(8)
        self.fc4bn = nn.BatchNorm1d(8)
        self.fc5bn = nn.BatchNorm1d(8)
        self.fc6bn = nn.BatchNorm1d(8)
        self.fc7bn = nn.BatchNorm1d(8)

    def forward(self, data, device):
        h, _, _ = self.forward_helper(data, device, cnn=True)
        h = F.leaky_relu(self.fc1bn(self.fc1(h)), 0.1)
        h = F.leaky_relu(self.fc2bn(self.fc2(h)), 0.1)
        h = F.leaky_relu(self.fc3bn(self.fc3(h)), 0.1)
        h = F.leaky_relu(self.fc4bn(self.fc4(h)), 0.1)
        h = F.leaky_relu(self.fc5bn(self.fc5(h)), 0.1)
        h = F.leaky_relu(self.fc6bn(self.fc6(h)), 0.1)
        h = F.leaky_relu(self.fc7bn(self.fc7(h)), 0.1)
        h = self.classifier(h)
        return h

    @classmethod
    def get_net_eval_fn(_, network_file, device=None):
        model = WallClassLSP()
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
