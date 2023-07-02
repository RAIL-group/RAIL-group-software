import os
import clip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import lsp_cond
from learning.data import CSVPickleDataset
from vertexnav.models import EncoderNBlocks


class CNNLSP(nn.Module):
    name = 'LSPforRawImage'

    def __init__(self, args=None):
        super(CNNLSP, self).__init__()
        torch.manual_seed(8616)
        self._args = args
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 3)

        self.fc1bn = nn.BatchNorm1d(32)
        self.fc2bn = nn.BatchNorm1d(16)
        self.fc3bn = nn.BatchNorm1d(16)
        self.fc4bn = nn.BatchNorm1d(8)
        self.fc5bn = nn.BatchNorm1d(8)

    def forward(self, data, device):
        h = data['latent_features']
        h = F.leaky_relu(self.fc1bn(self.fc1(h)), 0.1)
        h = F.leaky_relu(self.fc2bn(self.fc2(h)), 0.1)
        h = F.leaky_relu(self.fc3bn(self.fc3(h)), 0.1)
        h = F.leaky_relu(self.fc4bn(self.fc4(h)), 0.1)
        h = F.leaky_relu(self.fc5bn(self.fc5(h)), 0.1)
        h = self.classifier(h)
        return h

    def loss(self, nn_out, data, device='cpu', writer=None, index=None):
        # Separate outputs.
        is_feasible_logits = nn_out[:, 0]
        delta_cost_pred = nn_out[:, 1]
        exploration_cost_pred = nn_out[:, 2]

        # Convert the data
        is_feasible_label = data.y.to(device)
        has_updated = data.has_updated.to(device)
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
        is_feasible_xentropy /= torch.sum(subgoal_weight)+0.001
        # Set the loss type for Delta Success Cost and Exploration Cost
        if self._args.loss == 'l1':
            cost_loss = torch.abs
        else:
            cost_loss = torch.square

        # Delta Success Cost
        delta_cost_pred_error = cost_loss(
            delta_cost_pred - delta_cost_label) \
            / (100 ** 1) * is_feasible_label
        delta_cost_pred_error = torch.sum(subgoal_weight * delta_cost_pred_error)
        delta_cost_pred_error /= torch.sum(subgoal_weight)+0.001
        # Exploration Cost
        exploration_cost_pred_error = cost_loss(
            exploration_cost_pred - exploration_cost_label) / \
            (200 ** 1 * 4) * (1 - is_feasible_label)
        exploration_cost_pred_error = torch.sum(subgoal_weight * exploration_cost_pred_error)
        exploration_cost_pred_error /= torch.sum(subgoal_weight)+0.001
        
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

    # def loss(self, nn_out, data, device='cpu', writer=None, index=None):
    #     # Separate outputs.
    #     is_feasible_logits = nn_out[:, 0]
    #     delta_cost_pred = nn_out[:, 1]
    #     exploration_cost_pred = nn_out[:, 2]

    #     # Convert the data
    #     is_feasible_label = data.y.to(device)
    #     delta_cost_label = data.dsc.to(device)
    #     exploration_cost_label = data.ec.to(device)
    #     pweight = data.pweight.to(device)
    #     nweight = data.nweight.to(device)
    #     rpw = self._args.relative_positive_weight

    #     # Compute the contribution from the is_feasible_label
    #     is_feasible_xentropy = torch.mean(
    #         rpw * is_feasible_label * -F.logsigmoid(is_feasible_logits) 
    #         * pweight / 10 + (1 - is_feasible_label) * 
    #         -F.logsigmoid(-is_feasible_logits) * nweight / 10)

    #     # Delta Success Cost
    #     delta_cost_pred_error = torch.mean(torch.square(
    #         delta_cost_pred - delta_cost_label) / (100 ** 2) * 
    #         is_feasible_label)

    #     # Exploration Cost
    #     exploration_cost_pred_error = torch.mean(torch.square(
    #         exploration_cost_pred - exploration_cost_label) / 
    #         (200 ** 2) * (1 - is_feasible_label))

    #     # Sum the contributions
    #     loss = is_feasible_xentropy + delta_cost_pred_error + \
    #         exploration_cost_pred_error

    #     # Logging
    #     if writer is not None:
    #         writer.add_scalar("Loss/is_feasible_xentropy",
    #                           is_feasible_xentropy.item(),
    #                           index)
    #         writer.add_scalar("Loss/delta_success_cost_loss",
    #                           delta_cost_pred_error.item(),
    #                           index)
    #         writer.add_scalar("Loss/exploration_cost_loss",
    #                           exploration_cost_pred_error.item(),
    #                           index)
    #         writer.add_scalar("Loss/total_loss",
    #                           loss.item(),
    #                           index)

    #     return loss

    @classmethod
    def get_net_eval_fn(_, network_file, device=None):
        model = CNNLSP()
        model.load_state_dict(torch.load(network_file, 
                                         map_location=device)) 
        model.eval()
        model.to(device)

        def frontier_net(datum):
            with torch.no_grad():
                out = model.forward({'latent_features': datum}, device)
                out = out[:, :3]
                out[:, 0] = torch.sigmoid(out[:, 0])
                out = out.detach().cpu().numpy()
                return out[0, 0], out[0, 1], out[0, 2]
        return frontier_net


class CNN_CLIP(CNNLSP):
    name = 'CNNusingCLIP'

    def __init__(self, args=None):
        super(CNN_CLIP, self).__init__()
        torch.manual_seed(8616)
        self._args = args

        self.fc0 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.classifier = nn.Linear(8, 3)

        self.fc0bn = nn.BatchNorm1d(256)
        self.fc1bn = nn.BatchNorm1d(128)
        self.fc2bn = nn.BatchNorm1d(64)
        self.fc3bn = nn.BatchNorm1d(32)
        self.fc4bn = nn.BatchNorm1d(16)
        self.fc5bn = nn.BatchNorm1d(8)

    def forward(self, data, device):
        h = data['latent_features'].type(torch.float).to(device)

        h = F.leaky_relu(self.fc0bn(self.fc0(h)), 0.1)
        h = F.leaky_relu(self.fc1bn(self.fc1(h)), 0.1)
        h = F.leaky_relu(self.fc2bn(self.fc2(h)), 0.1)
        h = F.leaky_relu(self.fc3bn(self.fc3(h)), 0.1)
        h = F.leaky_relu(self.fc4bn(self.fc4(h)), 0.1)
        h = F.leaky_relu(self.fc5bn(self.fc5(h)), 0.1)
        h = self.classifier(h)
        return h

    @classmethod
    def get_net_eval_fn(_, network_file, device=None):
        model = CNN_CLIP()
        model.load_state_dict(torch.load(network_file, 
                                         map_location=device)) 
        model.eval()
        model.to(device)

        def frontier_net(datum):
            with torch.no_grad():
                out = model.forward({'latent_features': datum}, device)
                out = out[:, :3]
                out[:, 0] = torch.sigmoid(out[:, 0])
                out = out.detach().cpu().numpy()
                return out[0, 0], out[0, 1], out[0, 2]
        return frontier_net


def train(args, train_path, test_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Initialize the network and the optimizer
    if args.use_clip:
        print("Using CLIP encoder")
        _, preprocess = clip.load(args.clip_network_file, device=device)
        model = CNN_CLIP(args)
        latent_features_net = lsp_cond.learning.models.auto_encoder. \
            AutoEncoder.get_net_eval_fn(args.clip_network_file, device=device)
    else:
        preprocess = None
        if args.input_type == 'image' or args.input_type == 'seg_image':
            print("Using AE on pano_image")
            model = CNNLSP(args)
            latent_features_net = lsp_cond.learning.models.auto_encoder. \
                AutoEncoder.get_net_eval_fn_old(
                    args.autoencoder_network_file, device=device, args=args)
        elif args.input_type == 'wall_class':
            model = WallClassLSP(args)
            print("Using semantic wall class labels")

    # prep_fn = lsp_cond.utils.preprocess_cnn_training_data(
    #     fn=preprocess, args=args)
    prep_fn = lsp_cond.utils.preprocess_gcn_training_data(
            ['marginal'], preprocess, args)

    # Create the datasets and loaders
    train_dataset = CSVPickleDataset(train_path, prep_fn)
    print("Number of training graphs:", len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=4, 
                              shuffle=True, drop_last=True)
    train_iter = iter(train_loader)
    train_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "train_lsp"))

    test_dataset = CSVPickleDataset(test_path, prep_fn)
    print("Number of testing graphs:", len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=4, 
                             shuffle=True, drop_last=True)
    test_iter = iter(test_loader)
    test_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "test_lsp"))

    # Initialize the network and the optimizer
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
        if args.input_type == 'wall_class':
            train_latent_features = train_batch.x
        elif args.input_type == 'image' or args.input_type == 'seg_image':
            # Get latent features by running the encoder of AutoEncoder/CLIP
            train_latent_features = latent_features_net(
                {'image': train_batch.x}
            )
        out = model.forward({
            'edge_data': train_batch.edge_index,
            'history': train_batch.history,
            'goal_distance': train_batch.goal_distance,
            'is_subgoal': train_batch.is_subgoal,
            'latent_features': train_batch.x
        }, device)
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
            if args.input_type == 'wall_class':
                test_latent_features = test_batch.x
            elif args.input_type == 'image' or args.input_type == 'seg_image':
                test_latent_features = latent_features_net(
                    {'image': test_batch.x}
                )
            with torch.no_grad():
                out = model.forward({
                    'edge_data': test_batch.edge_index,
                    'history': test_batch.history,
                    'goal_distance': test_batch.goal_distance,
                    'is_subgoal': test_batch.is_subgoal,
                    'latent_features': test_batch.x
                }, device)
                test_loss = model.loss(out,
                                       data=test_batch,
                                       device=device,
                                       writer=test_writer,
                                       index=index)
                print(f"[{index}/{args.num_steps}] "
                      f"Test Loss: {test_loss.cpu().numpy()}")

        # Log the learning rate
        test_writer.add_scalar("learning_rate/LSP",
                               scheduler.get_last_lr()[-1],
                               index)
        index += 1
        scheduler.step()

    # Saving the model after training
    torch.save(model.state_dict(),
               os.path.join(args.save_dir, "lsp.pt"))