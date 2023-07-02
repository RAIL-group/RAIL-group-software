import os
import torch
import lsp_gnn
import random

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from learning.data import CSVPickleDataset
from lsp_gnn.learning.models.gcn import WallClassGNN
from lsp_gnn.learning.models.cnn_lsp import WallClassLSP


def get_model_prep_fn_and_training_strs(args):
    preprocess = None
    if args.train_marginal_lsp:
        print("Training Marginal LSP ... ...")
        model = WallClassGNN(args)
        prep_fn = lsp_gnn.utils.preprocess_gcn_training_data(
            ['marginal'], preprocess, args)
        train_writer_str = 'train_mlsp'
        test_writer_str = 'test_mlsp'
        lr_writer_str = 'learning_rate/mLSP'
        model_name_str = 'mlsp.pt'
    elif args.train_cnn_lsp:
        print("Training Base LSP ... ...")
        model = WallClassLSP(args)
        prep_fn = lsp_gnn.utils.preprocess_gcn_training_data(
            ['marginal'], preprocess, args)
        train_writer_str = 'train_lsp'
        test_writer_str = 'test_lsp'
        lr_writer_str = 'learning_rate/LSP'
        model_name_str = 'lsp.pt'

    return {
        'model': model,
        'prep_fn': prep_fn,
        'train_writer_str': train_writer_str,
        'test_writer_str': test_writer_str,
        'lr_writer_str': lr_writer_str,
        'model_name_str': model_name_str
    }


def train(args, train_path, test_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Assertion since the current project has working results only for wall class labels
    assert args.input_type == 'wall_class'
    print("Using semantic wall class labels")

    # Get the model and other training info
    model_and_training_info = get_model_prep_fn_and_training_strs(args)
    model = model_and_training_info['model']
    prep_fn = model_and_training_info['prep_fn']
    train_writer_str = model_and_training_info['train_writer_str']
    test_writer_str = model_and_training_info['test_writer_str']
    lr_writer_str = model_and_training_info['lr_writer_str']
    model_name_str = model_and_training_info['model_name_str']

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
            'edge_features': train_batch.edge_features,
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
                    'edge_features': test_batch.edge_features,
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


if __name__ == "__main__":
    args = lsp_gnn.utils.parse_args()
    # Always freeze your random seeds
    torch.manual_seed(8616)
    random.seed(8616)
    train_path, test_path = lsp_gnn.utils.get_data_path_names(args)
    # Train the neural network
    if args.autoencoder_network_file:
        train(args=args, train_path=train_path, test_path=test_path)
    else:
        print("Training AutoEncoder ... ...")
        lsp_gnn.learning.models.auto_encoder.train(
            args=args, train_path=train_path, test_path=test_path)
