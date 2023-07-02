import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import lsp_gnn
import learning
from learning.data import CSVPickleDataset
from vertexnav.models import EncoderNBlocks, DecoderNBlocks

MINI_BATCH_SIZE = 40


class AutoEncoder(nn.Module):
    name = "AutoEncoder"

    def __init__(self, args=None):
        super(AutoEncoder, self).__init__()
        torch.manual_seed(8616)
        if args.input_type == 'image':
            start_dim = 3
        elif args.input_type == 'seg_image':
            start_dim = 4
        channel = 32
        hidden = 8
        self._args = args                                # (3, 32, 16)
        self.enc_1 = EncoderNBlocks(start_dim, channel, num_layers=2)  # (8, 16, 8)
        self.enc_2 = EncoderNBlocks(channel, channel, num_layers=2)  # (8, 8, 4)
        self.enc_3 = EncoderNBlocks(channel, hidden, num_layers=2)  # (4, 4, 2)

        self.dec_1 = DecoderNBlocks(hidden, channel, num_layers=2)
        self.dec_2 = DecoderNBlocks(channel, channel, num_layers=2)
        self.dec_3 = DecoderNBlocks(channel, start_dim, num_layers=2)

    # The following method is used only during evaluating
    def encoder(self, data, device):
        x = data['image'].to(device)
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        latent_features = x
        return latent_features

    def decoder(self, x, device):
        x = self.dec_1(x)
        x = self.dec_2(x)
        x = self.dec_3(x)
        output = {
            'image': x
        }
        return output

    # The following method is used only during training
    def forward(self, data, device):
        x = self.encoder(data, device)
        return self.decoder(x, device)  # , self.lsp(x, device)

    def loss(self, nn_out, data, device='cpu',
             writer=None, index=None):
        pred_img = nn_out['image']
        trgt_img = data['image'].to(device)
        if self._args.input_type == 'seg_image':
            image_logsoftmax = torch.nn.LogSoftmax(dim=1)(pred_img)
            loss_img = -torch.sum(image_logsoftmax * trgt_img, dim=1)
            loss_img = torch.mean(loss_img)
        elif self._args.input_type == 'image':
            if self._args.loss == 'l1':
                mae_loss = nn.L1Loss()
                loss_img = mae_loss(pred_img, trgt_img)
            elif self._args.loss == 'l2':
                mse_loss = nn.MSELoss()
                loss_img = mse_loss(pred_img, trgt_img)

        # Logging
        if writer is not None:
            writer.add_scalar("Loss_AE/image_loss",
                              loss_img.item(),
                              index)
        return loss_img

    @classmethod
    def get_net_eval_fn(_, network_file, device, preprocess_for=None, args=None):
        model = AutoEncoder(args)
        model.load_state_dict(torch.load(network_file,
                                         map_location=device))
        model.eval()
        model.to(device)

        def latent_features_net(datum):
            with torch.no_grad():
                if preprocess_for == 'Cond_Eval':
                    latent_features = torch.zeros(0).to(device)
                    for idx, image in enumerate(datum['image']):
                        data = lsp_gnn.utils.preprocess_cnn_data(
                            datum, idx=idx, args=args)
                        lf = model.encoder(data, device)
                        if datum['is_subgoal'][idx] == 0:
                            lf = torch.mean(lf, dim=0).expand(1, -1, -1, -1)
                        latent_features = torch.cat((
                            latent_features, lf), 0)
                elif preprocess_for == 'CNN_Eval':
                    data = lsp_gnn.utils.preprocess_cnn_eval_data(datum, args)
                    latent_features = model.encoder(data, device)
                else:
                    data = datum
                    latent_features = torch.cat([model.encoder({
                        'image': data['image'][idx:idx + MINI_BATCH_SIZE],
                    }, device) for idx in range(0, len(data['image']),
                                                MINI_BATCH_SIZE)])
            return latent_features.flatten(start_dim=1)

        return latent_features_net

    @learning.logging.tensorboard_plot_decorator
    def plot_images(self, fig, out, data):
        index = 0
        pred_img = np.transpose(
            out['image'][index].detach().cpu().numpy(), (1, 2, 0))
        trgt_img = np.transpose(data['image'][index].cpu().numpy(), (1, 2, 0))

        axs = fig.subplots(1, 2, squeeze=False)
        axs[0][0].imshow(trgt_img, interpolation='none')
        axs[0][0].set_title("Input image")
        axs[0][1].imshow(pred_img, interpolation='none')
        axs[0][1].set_title("Recreated image")


def train(args, train_path, test_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.input_type == 'image':
        prep_fn = lsp_gnn.utils.preprocess_ae_img
    elif args.input_type == 'seg_image':
        prep_fn = lsp_gnn.utils.preprocess_ae_seg_img
    else:
        print("No AE training required for wall class label input_type")
        return

    # Create the datasets and loaders
    train_dataset = CSVPickleDataset(train_path, prep_fn)
    print("Number of training images:", len(train_dataset))
    train_loader = DataLoader(train_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=0)

    train_iter = iter(train_loader)
    train_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "train_autoencoder"))

    test_dataset = CSVPickleDataset(test_path, prep_fn)
    print("Number of testing images:", len(test_dataset))
    test_loader = DataLoader(test_dataset,
                             batch_size=16,
                             shuffle=True,
                             num_workers=0)

    test_iter = iter(test_loader)
    test_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "test_autoencoder"))

    # Initialize the network and the optimizer
    model = AutoEncoder(args)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.epoch_size,
        gamma=args.learning_rate_decay_factor)
    index = 0
    while index < args.num_steps:
        # Get the batches
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)
        # out is a dictionary
        out = model.forward(train_batch, device)
        train_loss = model.loss(out,
                                data=train_batch,
                                device=device,
                                writer=train_writer,
                                index=index)
        if index % args.test_log_frequency == 0:
            print(f"[{index}/{args.num_steps}] "
                  f"Train Loss: {train_loss}")

        # Train the system
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if index % args.test_log_frequency == 0:
            try:
                test_batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_batch = next(test_iter)
            with torch.no_grad():
                out = model.forward(test_batch, device)
                test_loss = model.loss(out,
                                       data=test_batch,
                                       device=device,
                                       writer=test_writer,
                                       index=index)
                if args.input_type == 'image':
                    # Plotting
                    model.plot_images(
                        test_writer, 'image', index,
                        out=out, data=test_batch)
                print(f"[{index}/{args.num_steps}] "
                      f"Test Loss: {test_loss.cpu().numpy()}")
            print(f"Learning rate: {scheduler.get_last_lr()[-1]}")

        # Log the learning rate
        test_writer.add_scalar("learning_rate/AE",
                               scheduler.get_last_lr()[-1],
                               index)

        index += 1
        scheduler.step()

    # Saving the model after training
    torch.save(model.state_dict(),
               os.path.join(args.save_dir, "AutoEncoder.pt"))
