import argparse
import glob
import learning
import lsp
import os
import torch
from torch.utils.tensorboard import SummaryWriter


def train_main(args):

    # Get the data files for training

    # Set up the learning
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = lsp.learning.models.VisLSPOriented(args).to(device)
    print(f"Training Device: {device}")

    # Create the datasets and loaders
    training_data_files = glob.glob(os.path.join(args.data_csv_dir, "*training*.csv"))
    testing_data_files = glob.glob(os.path.join(args.data_csv_dir, "*testing*.csv"))
    preprocess_function = lsp.learning.models.VisLSPOriented.preprocess_data
    train_dataset = learning.data.CSVPickleDataset(training_data_files, preprocess_function)
    test_dataset = learning.data.CSVPickleDataset(testing_data_files, preprocess_function)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=4)
    test_loader_iter = iter(test_loader)

    # Set up logging
    train_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "train"))
    test_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "test"))

    # Define the optimizer
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                gamma=args.learning_rate_decay_factor)

    tot_index = 0
    for epoch in range(args.num_epochs):
        for index, batch in enumerate(train_loader):
            out = model(batch, device)
            loss = model.loss(out,
                              batch,
                              device=device,
                              writer=train_writer,
                              index=tot_index)

            if index % 100 == 0:
                with torch.no_grad():
                    try:
                        tbatch = next(test_loader_iter)
                    except StopIteration:
                        test_loader_iter = iter(test_loader)
                        tbatch = next(test_loader_iter)

                    tim = tbatch['image']
                    tout = model(tbatch, device)
                    tloss = model.loss(tout,
                                       tbatch,
                                       device=device,
                                       writer=test_writer,
                                       index=tot_index)

                    print(f"Test Loss({epoch}.{index}, {tot_index}): {tloss.item()}")
                    model.plot_images(test_writer,
                                      'image',
                                      tot_index,
                                      image=tim[0].detach(),
                                      out=tout[0].detach().cpu(),
                                      data=tbatch)

            # Perform update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_index += 1

        # Step the learning rate scheduler
        scheduler.step()

    # Now save the trained model to file
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"{model.name}.pt"))


def get_parser():
    # Add new arguments
    parser = argparse.ArgumentParser(
        description="Train LSP net with PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Logging
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--data_csv_dir', type=str)
    parser.add_argument(
        '--mini_summary_frequency',
        default=100,
        help='Frequency (in steps) mini summary printed to the terminal',
        type=int)
    parser.add_argument('--summary_frequency',
                        default=100,
                        help='Frequency (in steps) summary is logged to file',
                        type=int)

    # Training
    parser.add_argument('--num_epochs',
                        default=8,
                        help='Number of epochs to run training',
                        type=int)
    parser.add_argument('--learning_rate',
                        default=0.002,
                        help='Initial learning rate',
                        type=float)
    parser.add_argument('--learning_rate_decay_factor',
                        default=0.5,
                        help='How much learning rate decreases between epochs.',
                        type=float)
    parser.add_argument('--batch_size',
                        default=32,
                        help='Number of data per training iteration batch',
                        type=int)
    parser.add_argument('--relative_positive_weight',
                        default=1.0,
                        help='Initial learning rate',
                        type=float)

    return parser


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    args = get_parser().parse_args()
    train_main(args)
