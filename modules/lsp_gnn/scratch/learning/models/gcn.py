import os
import clip
import torch
import lsp_cond
import learning
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LEConv
from learning.data import CSVPickleDataset
import torch_geometric.utils
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter


class LSPConditionalGNN(nn.Module):
    name = 'GNNforImage'

    def __init__(self, args=None):
        super(LSPConditionalGNN, self).__init__()
        torch.manual_seed(8616)
        self._args = args

        self.fc1 = nn.Linear(64+2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.conv1 = SAGEConv(8, 8)
        self.conv2 = SAGEConv(8, 8)
        self.conv3 = SAGEConv(8, 8)
        self.classifier = nn.Linear(8, 3)

        self.fc1bn = nn.BatchNorm1d(32)
        self.fc2bn = nn.BatchNorm1d(16)
        self.fc3bn = nn.BatchNorm1d(8)
        self.conv1bn = nn.BatchNorm1d(8)
        self.conv2bn = nn.BatchNorm1d(8)
        self.conv3bn = nn.BatchNorm1d(8)

    def forward(self, data, device):
        lf = data['latent_features'].to(device)
        my_tensor = data['edge_data']
        x = torch.cat((my_tensor[0], my_tensor[1]), 0)
        y = torch.cat((my_tensor[1], my_tensor[0]), 0)
        my_tensor = torch.reshape(torch.cat((x, y), 0), (2, -1))
        edge_index = my_tensor.to(device)
        history = data['history'].view(-1, 1).to(device)
        is_subgoal = data['is_subgoal'].view(-1, 1).to(device)
        h = torch.cat((lf, history), 1)
        h = torch.cat((h, is_subgoal), 1)
        h = F.leaky_relu(self.fc1bn(self.fc1(h)), 0.1)
        h = F.leaky_relu(self.fc2bn(self.fc2(h)), 0.1)
        h = F.leaky_relu(self.fc3bn(self.fc3(h)), 0.1)
        h = F.leaky_relu(self.conv1bn(self.conv1(h, edge_index)), 0.1)
        h = F.leaky_relu(self.conv2bn(self.conv2(h, edge_index)), 0.1)
        h = F.leaky_relu(self.conv3bn(self.conv3(h, edge_index)), 0.1)
        props = self.classifier(h)
        return props

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

    @classmethod
    def get_net_eval_fn(_, network_file, 
                        device=None, comb=False):
        model = LSPConditionalGNN()
        model.load_state_dict(torch.load(network_file, 
                                         map_location=device)) 
        model.eval()
        model.to(device)

        def frontier_net(datum, vertex_points, subgoals):
            graph = lsp_cond.utils.preprocess_gcn_data(datum)
            prob_feasible_dict = {}
            dsc_dict = {}
            ec_dict = {}
            with torch.no_grad():    
                out = model.forward(graph, device)
                out = out[:, :3]
                out[:, 0] = torch.sigmoid(out[:, 0])
                out = out.detach().cpu().numpy()
                for subgoal in subgoals:
                    index_pos, possible_node = lsp_cond.utils. \
                        get_subgoal_node(vertex_points, subgoal)
                    # Extract subgoal properties for a subgoal
                    subgoal_props = out[index_pos]
                    prob_feasible_dict[subgoal] = subgoal_props[0]
                    dsc_dict[subgoal] = subgoal_props[1]
                    ec_dict[subgoal] = subgoal_props[2]
                return prob_feasible_dict, dsc_dict, ec_dict, out[:, 0]

        return frontier_net

    @learning.logging.tensorboard_plot_decorator
    def plot_images(self, fig, out, data):
        is_subgoal = data.is_subgoal
        subgoal_idx_pool = [
            idx
            for idx, val in enumerate(is_subgoal)
            if val == 1
        ]
        count = len(subgoal_idx_pool)

        # Separate outputs.
        is_feasible_logits = out[:, 0]

        COL = 2
        if count % COL == 0:
            ROW = count // COL
        else:
            ROW = count // COL + 1
        if ROW < 2:
            ROW = 2
        axs = fig.subplots(ROW, COL)
        trgt_img = np.transpose(data.x.cpu().numpy(), (0, 2, 3, 1))
        row_pointer = 0
        col_pointer = 0
        for idx in subgoal_idx_pool:
            axs[row_pointer][col_pointer].imshow(
                trgt_img[idx], interpolation='none')
            axs[row_pointer][col_pointer].set_title(
                f"[R]: {data.y[idx].cpu().numpy()} "
                f"[P]: {torch.sigmoid(is_feasible_logits[idx]).detach().cpu().numpy():.2f}"
                )
            col_pointer += 1
            if col_pointer == COL:
                col_pointer = 0
                row_pointer += 1


class GNN_CLIP(LSPConditionalGNN):
    name = 'GNNusingCLIP'

    def __init__(self, args=None):
        super(GNN_CLIP, self).__init__()
        torch.manual_seed(8616)
        self._args = args
        self.fc1 = nn.Linear(512+2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.conv1 = SAGEConv(8, 8)
        self.conv2 = SAGEConv(8, 8)
        self.conv3 = SAGEConv(8, 8)
        self.classifier = nn.Linear(8, 3)

        self.fc1bn = nn.BatchNorm1d(256)
        self.fc2bn = nn.BatchNorm1d(128)
        self.fc3bn = nn.BatchNorm1d(64)
        self.fc4bn = nn.BatchNorm1d(32)
        self.fc5bn = nn.BatchNorm1d(16)
        self.fc6bn = nn.BatchNorm1d(8)
        self.conv1bn = nn.BatchNorm1d(8)
        self.conv2bn = nn.BatchNorm1d(8)
        self.conv3bn = nn.BatchNorm1d(8)

    def forward(self, data, device):
        props = self.get_subgoal_props(
            lf=data['latent_features'].type(torch.float).to(device),
            data=data,
            device=device
        )
        return props

    def get_subgoal_props(self, lf, data, device='cpu'):
        edge_index = data['edge_data'].to(device)
        history = data['history'].view(-1, 1).to(device)
        is_subgoal = data['is_subgoal'].view(-1, 1).to(device)
        h = torch.cat((lf, history), 1)
        h = torch.cat((h, is_subgoal), 1)
        h = F.leaky_relu(self.fc1bn(self.fc1(h)), 0.1)
        h = F.leaky_relu(self.fc2bn(self.fc2(h)), 0.1)
        h = F.leaky_relu(self.fc3bn(self.fc3(h)), 0.1)
        h = F.leaky_relu(self.fc4bn(self.fc4(h)), 0.1)
        h = F.leaky_relu(self.fc5bn(self.fc5(h)), 0.1)
        h = F.leaky_relu(self.fc6bn(self.fc6(h)), 0.1)
        h = F.leaky_relu(self.conv1bn(self.conv1(h, edge_index)), 0.1)
        h = F.leaky_relu(self.conv2bn(self.conv2(h, edge_index)), 0.1)
        h = F.leaky_relu(self.conv3bn(self.conv3(h, edge_index)), 0.1)
        h = self.classifier(h)
        return h

    @classmethod
    def get_net_eval_fn(_, network_file, get_lf=False, 
                        device=None, comb=False):
        model = GNN_CLIP()
        model.load_state_dict(torch.load(network_file, 
                                         map_location=device)) 
        model.eval()
        model.to(device)

        def latent_features_gcn_net(latent_features):
            return latent_features

        def frontier_net(datum, vertex_points, subgoals):
            graph = lsp_cond.utils.preprocess_gcn_data(datum)
            prob_feasible_dict = {}
            dsc_dict = {}
            ec_dict = {}
            with torch.no_grad():
                if comb:
                    out = model.forward(graph, device)
                else:    
                    out = model.get_subgoal_props(
                        graph['latent_features'], graph, device)
                out = out[:, :3]
                out[:, 0] = torch.sigmoid(out[:, 0])
                out = out.detach().cpu().numpy()
                for subgoal in subgoals:
                    index_pos, possible_node = lsp_cond.utils. \
                        get_subgoal_node(vertex_points, subgoal)
                    # Extract subgoal properties for a subgoal
                    subgoal_props = out[index_pos]
                    prob_feasible_dict[subgoal] = subgoal_props[0]
                    dsc_dict[subgoal] = subgoal_props[1]
                    ec_dict[subgoal] = subgoal_props[2]
                return prob_feasible_dict, dsc_dict, ec_dict, out[:, 0]

        if get_lf:
            return latent_features_gcn_net
        else:
            return frontier_net


def train(args, train_path, test_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Initialize the network and the optimizer
    preprocess = None
    if args.input_type == 'image' or args.input_type == 'seg_image':
        model = LSPConditionalGNN(args)
        print(f"Using AE on pano_{args.input_type}")
        args.latent_features_net = lsp_cond.learning.models.auto_encoder. \
            AutoEncoder.get_net_eval_fn_old(
                args.autoencoder_network_file, device=device,
                preprocess_for='Cond_Eval', args=args)
    elif args.input_type == 'wall_class':
        model = WallClassGNN(args)
        print("Using semantic wall class labels")
    elif args.use_clip:
        print("Using CLIP encoder")
        _, preprocess = clip.load(args.clip_network_file, device=device)
        model = GNN_CLIP(args)
        latent_features_net = lsp_cond.learning.models.auto_encoder. \
            AutoEncoder.get_net_eval_fn(args.clip_network_file, device=device)
    
    if args.train_marginal_lsp:
        prep_fn = lsp_cond.utils.preprocess_gcn_training_data(
            ['marginal'], preprocess, args)
        train_writer_str = 'train_mlsp'
        test_writer_str = 'test_mlsp'
        lr_writer_str = 'learning_rate/mLSP'
        model_name_str = 'mlsp.pt'
    else:
        prep_fn = lsp_cond.utils.preprocess_gcn_training_data(fn=preprocess)
        train_writer_str = 'train'
        test_writer_str = 'test'
        lr_writer_str = 'learning_rate/cLSP'
        model_name_str = 'model.pt'

    # Create the datasets and loaders
    train_dataset = CSVPickleDataset(train_path, prep_fn)
    print("Number of training graphs:", len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train_iter = iter(train_loader)
    train_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, train_writer_str))

    test_dataset = CSVPickleDataset(test_path, prep_fn)
    print("Number of testing graphs:", len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    test_iter = iter(test_loader)
    test_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, test_writer_str))

    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.epoch_size,
        gamma=args.learning_rate_decay_factor)
    index = 0
    while index < args.num_steps:
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)
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
        test_writer.add_scalar(lr_writer_str,
                               scheduler.get_last_lr()[-1],
                               index)
        index += 1
        scheduler.step()

    # Saving the model after training
    torch.save(model.state_dict(),
               os.path.join(args.save_dir, model_name_str))
