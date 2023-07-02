import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import learning

from .shared import EncoderNBlocks


class VisLSPOriented(nn.Module):
    name = "VisLSPOriented"

    def __init__(self, args=None, num_outputs=3):
        super(VisLSPOriented, self).__init__()
        self._args = args

        # Initialize the blocks
        self.enc_1 = EncoderNBlocks(3, 64, num_layers=2)
        self.enc_2 = EncoderNBlocks(64, 64, num_layers=2)
        self.enc_3 = EncoderNBlocks(64 + 4, 128, num_layers=2)
        self.enc_4 = EncoderNBlocks(128, 128, num_layers=2)
        self.enc_5 = EncoderNBlocks(128, 256, num_layers=2)
        self.enc_6 = EncoderNBlocks(256, 128, num_layers=2)
        self.conv_1x1 = nn.Conv2d(128, 16, kernel_size=1)

        self.fc_outs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, num_outputs),
        )

        self.goal_bn = nn.BatchNorm2d(2)
        self.subgoal_bn = nn.BatchNorm2d(2)

    def forward(self, data, device):
        image = data['image'].to(device)

        # Compute goal info tensor
        if 'goal_loc_x' in data.keys():
            g = self.goal_bn(
                torch.stack((data['goal_loc_x'], data['goal_loc_y']),
                            1).expand([-1, -1, 32, -1]).float().to(device))
        else:
            raise ValueError("Missing goal location data.")

        if 'subgoal_loc_x' in data.keys():
            s = self.subgoal_bn(
                torch.stack((data['subgoal_loc_x'], data['subgoal_loc_y']),
                            1).expand([-1, -1, 32, -1]).float().to(device))
        else:
            raise ValueError("Missing subgoal location data.")

        x = image

        # Encoding layers
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = torch.cat((x, s, g), 1)  # Add the goal info tensor
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)
        x = self.enc_6(x)
        x = self.conv_1x1(x)
        x = self.fc_outs(x)

        return x

    def loss(self, nn_out, data, device='cpu', writer=None, index=None):
        # Separate outputs.
        is_feasible_logits = nn_out[:, 0]
        delta_cost_pred = nn_out[:, 1]
        exploration_cost_pred = nn_out[:, 2]

        # Convert the data
        is_feasible_label = data['is_feasible'].to(device)
        delta_cost_label = data['delta_success_cost'].to(device)
        exploration_cost_label = data['exploration_cost'].to(device)
        pweight = data['positive_weighting'].to(device)
        nweight = data['negative_weighting'].to(device)
        rpw = self._args.relative_positive_weight

        # Compute the contribution from the is_feasible_label
        is_feasible_xentropy = torch.mean(rpw * is_feasible_label * -F.logsigmoid(is_feasible_logits) * pweight / 10 +
                                          (1 - is_feasible_label) * -F.logsigmoid(-is_feasible_logits) * nweight / 10)

        # Delta Success Cost
        delta_cost_pred_error = torch.mean(torch.square(
            delta_cost_pred - delta_cost_label) / (100 ** 2) * is_feasible_label)

        # Exploration Cost
        exploration_cost_pred_error = torch.mean(torch.square(
            exploration_cost_pred - exploration_cost_label) / (200 ** 2) * (1 - is_feasible_label))

        # Sum the contributions
        loss = is_feasible_xentropy + delta_cost_pred_error + exploration_cost_pred_error

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

    @learning.logging.tensorboard_plot_decorator
    def plot_images(self, fig, image, out, data):
        image = np.transpose(image, (1, 2, 0))
        pred_feasible = torch.sigmoid(out[0]).cpu().numpy()
        is_feasible = data['is_feasible'][0]

        axs = fig.subplots(1, 1)
        axs.imshow(image, interpolation='none')
        axs.set_title(f"{is_feasible}: {pred_feasible}")

    @classmethod
    def preprocess_data(_, datum, is_training=True):
        datum['image'] = np.transpose(datum['image'], (2, 0, 1)).astype(np.float32) / 255
        return datum

    @classmethod
    def get_net_eval_fn(_, network_file, device):
        model = VisLSPOriented()
        model.load_state_dict(torch.load(network_file))
        model.eval()
        model.to(device)

        def frontier_net(nn_input_data):
            with torch.no_grad():
                nn_input_data = VisLSPOriented.preprocess_data(nn_input_data, is_training=False)
                out = model({
                    'image': torch.tensor(np.expand_dims(nn_input_data['image'], axis=0)).float(),
                    'goal_loc_x': torch.tensor(np.expand_dims(nn_input_data['goal_loc_x'], axis=0)).float(),
                    'goal_loc_y': torch.tensor(np.expand_dims(nn_input_data['goal_loc_y'], axis=0)).float(),
                    'subgoal_loc_x': torch.tensor(np.expand_dims(nn_input_data['subgoal_loc_x'], axis=0)).float(),
                    'subgoal_loc_y': torch.tensor(np.expand_dims(nn_input_data['subgoal_loc_y'], axis=0)).float(),
                }, device=device)
                out = out[:, :3]
                out[:, 0] = torch.sigmoid(out[:, 0])
                out = out.detach().cpu().numpy()
                return out[0, 0], out[0, 1], out[0, 2]

        return frontier_net
