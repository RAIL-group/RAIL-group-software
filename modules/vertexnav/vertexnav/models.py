import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.logging import tensorboard_plot_decorator


class EncoderNBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(EncoderNBlocks, self).__init__()
        nin = in_channels
        nout = out_channels
        modules = []

        # First layer
        modules.append(nn.Conv2d(nin, nout, kernel_size=3, stride=1,
                                 padding=1))
        modules.append(nn.BatchNorm2d(nout))
        modules.append(nn.LeakyReLU(0.1, inplace=True))

        # Add remaining layers
        for ii in range(1, num_layers):
            modules.append(
                nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding=1))
            modules.append(nn.BatchNorm2d(nout))
            modules.append(nn.LeakyReLU(0.1, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.cnn_layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn_layers(x)


class DecoderNBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(DecoderNBlocks, self).__init__()
        nin = in_channels
        nout = out_channels
        modules = []

        # Define the per-layer strides
        strides = [1] * num_layers
        strides[-1] = 2

        # First layer
        modules.append(nn.Conv2d(nin, nout, kernel_size=3, stride=1,
                                 padding=1))
        modules.append(nn.BatchNorm2d(nout))
        modules.append(nn.LeakyReLU(0.1, inplace=True))

        # Add remaining layers
        for ii in range(1, num_layers):
            if strides[ii] > 1:
                modules.append(
                    nn.ConvTranspose2d(nout,
                                       nout,
                                       kernel_size=3,
                                       stride=strides[ii],
                                       output_padding=1))
            else:
                modules.append(
                    nn.ConvTranspose2d(nout,
                                       nout,
                                       kernel_size=3,
                                       stride=strides[ii]))
            modules.append(nn.BatchNorm2d(nout))
            modules.append(nn.LeakyReLU(0.1, inplace=True))

        self.cnn_layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn_layers(x)[:, :, 1:-1, 1:-1]


class VertexNavGrid(nn.Module):
    name = "VertexNavGrid"
    ROLL_VARIABLES = [
        'image', 'is_vertex', 'is_right_gap', 'is_corner', 'is_left_gap',
        'is_point_vertex'
    ]

    def __init__(self, args=None):
        super(VertexNavGrid, self).__init__()
        self._args = args

        # Initialize the blocks
        self.enc_1 = EncoderNBlocks(3, 64, num_layers=2)
        self.enc_2 = EncoderNBlocks(64, 64, num_layers=2)
        self.enc_3 = EncoderNBlocks(64, 128, num_layers=2)
        self.enc_4 = EncoderNBlocks(128, 128, num_layers=2)
        self.enc_5 = EncoderNBlocks(128, 256, num_layers=2)

        self.dec_1 = DecoderNBlocks(256, 128, num_layers=2)
        self.dec_2 = DecoderNBlocks(128, 64, num_layers=2)
        self.dec_3 = DecoderNBlocks(64, 64, num_layers=2)
        self.conv_out = nn.Conv2d(64, 5, kernel_size=1)
        self.goal_bn = nn.BatchNorm2d(2)

    def forward(self, data, device):
        image = data['image'].to(device)
        x = image

        # Encoding layers
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)

        # Decoding layers
        x = self.dec_1(x)
        x = self.dec_2(x)
        x = self.dec_3(x)

        # Output layer
        x = self.conv_out(x)

        return x

    def loss(self, nn_out, data, device='cpu', writer=None, index=None):
        def masked_reduce_mean(data, mask):
            mask = (mask > 0.5).float()
            mask_sum = torch.sum(mask)
            data_sum = torch.sum(data * mask)
            return data_sum / (mask_sum + 1)

        rpw = self._args.relative_positive_weight

        # Compute the contribution from is_vertex
        is_vertex_logits = nn_out[:, 0, :, :]
        is_vertex_label = data['is_vertex'].to(device)
        is_vertex_xentropy = rpw * is_vertex_label * -F.logsigmoid(is_vertex_logits) + \
            (1 - is_vertex_label) * -F.logsigmoid(-is_vertex_logits)
        is_vertex_xentropy = torch.mean(is_vertex_xentropy)

        # Separate outputs.
        is_label_logits = nn_out[:, 1:, :, :]
        is_right_gap = data['is_right_gap'].to(device)
        is_corner = data['is_corner'].to(device)
        is_left_gap = data['is_left_gap'].to(device)
        is_point_vertex = data['is_point_vertex'].to(device)
        is_label_label = torch.stack(
            [is_right_gap, is_corner, is_left_gap, is_point_vertex], axis=1)
        is_label_logsoftmax = torch.nn.LogSoftmax(dim=1)(is_label_logits)
        is_label_xentropy = -torch.sum(is_label_logsoftmax * is_label_label,
                                       dim=1)
        is_label_xentropy = masked_reduce_mean(is_label_xentropy,
                                               is_vertex_label)

        loss = self._args.vertex_pred_weight * is_vertex_xentropy + is_label_xentropy

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/is_vertex_xentropy",
                              is_vertex_xentropy.item(), index)
            writer.add_scalar("Loss/total_loss", loss.item(), index)

        return loss

    @tensorboard_plot_decorator
    def plot_images(self, fig, image, out, data):
        image = np.transpose(image, (1, 2, 0))

        is_vertex_pred = torch.sigmoid(out[0, :, :]).cpu().numpy()
        is_vertex_label = data['is_vertex'][0]

        is_label_pred = torch.softmax(out[1:, :, :], dim=0).cpu().numpy()
        is_label_pred = np.transpose(is_label_pred, axes=[1, 2, 0])

        is_right_gap = data['is_right_gap'][0] * 1.0
        is_corner = data['is_corner'][0] * 1.0
        is_left_gap = data['is_left_gap'][0] * 1.0
        is_point_vertex = data['is_point_vertex'][0] * 1.0
        is_label_label = torch.stack(
            [is_right_gap, is_corner, is_left_gap, is_point_vertex],
            axis=2).numpy()

        is_label_label[:, :, 0][is_vertex_label < 0.5] = float('NaN')
        is_label_label[:, :, 1][is_vertex_label < 0.5] = float('NaN')
        is_label_label[:, :, 2][is_vertex_label < 0.5] = float('NaN')

        axs = fig.subplots(3, 2)
        axs[0, 0].imshow(image, interpolation='none')
        axs[1, 0].imshow(is_vertex_label,
                         interpolation='none',
                         vmin=0.0,
                         vmax=1.0)
        axs[2, 0].imshow(is_vertex_pred,
                         interpolation='none',
                         vmin=0.0,
                         vmax=1.0)

        axs[0, 1].imshow(image, interpolation='none')
        axs[1, 1].imshow(is_label_label[:, :, :3], interpolation='none')
        axs[2, 1].imshow(is_label_pred[:, :, :3], interpolation='none')

    @classmethod
    def get_net_eval_fn(_, network_file, device, do_return_model=False):
        model = VertexNavGrid()
        model.load_state_dict(torch.load(network_file,
                                         map_location=torch.device('cpu')),
                              strict=False)
        model.eval()
        model.to(device)

        def vertex_net(image):
            with torch.no_grad():
                image = np.transpose(image, (2, 0, 1))
                out = model(
                    {
                        'image':
                        torch.tensor(np.expand_dims(image, axis=0)).float(),
                    },
                    device=device)

            out[0, 0] = torch.sigmoid(out[0, 0])
            out[0, 1:] = torch.softmax(out[0, 1:], dim=0)
            out = out.detach().cpu().numpy()
            return {
                'is_vertex': out[0, 0],
                'vertex_label': np.transpose(out[0, 1:], axes=(1, 2, 0))
            }

        if do_return_model:
            return vertex_net, model
        else:
            return vertex_net
