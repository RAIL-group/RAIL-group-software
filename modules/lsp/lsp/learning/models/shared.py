import torch.nn as nn


class EncoderNBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(EncoderNBlocks, self).__init__()
        nin = in_channels
        nout = out_channels
        modules = []

        # First layer
        modules.append(
            nn.Conv2d(nin,
                      nout,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        modules.append(nn.BatchNorm2d(nout, momentum=0.01))
        modules.append(nn.LeakyReLU(0.1, inplace=True))

        # Add remaining layers
        for ii in range(1, num_layers):
            modules.append(
                nn.Conv2d(nout,
                          nout,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            modules.append(nn.BatchNorm2d(nout, momentum=0.01))
            modules.append(nn.LeakyReLU(0.1, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.cnn_layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn_layers(x)
