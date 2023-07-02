import os
import matplotlib.pyplot as plt
import numpy as np

import lsp
import learning
from lsp_xai.learning.models import ExpNavVisLSP
import environments
import torch

from captum.attr import IntegratedGradients

MODEL_CLASS = ExpNavVisLSP


def get_nn_model_name(model, args, extension):
    """Get the name of the file to which network params will be saved."""
    return os.path.join(args.logdir, f"{model.name}.{extension}.pt")


def explain_visuals_with_lime(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)

    # Load the data
    datum = learning.data.load_compressed_pickle(
        os.path.join(args.save_dir, 'data', args.datum_name))

    # Open the connection to Unity (if desired)
    if args.unity_path is None:
        raise ValueError('Unity Environment Required')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    subgoal_property_net, model = ExpNavVisLSP.get_net_eval_fn(
        args.network_file, device=device, do_return_model=True)

    for ind, input_data in datum['subgoal_data'].items():

        def predict_fn(image):
            out = []
            for img in image:
                [prob_feasible, delta_success_cost, exploration_cost] = \
                    subgoal_property_net(
                        image=img,
                        goal_loc_x=input_data['goal_loc_x'],
                        goal_loc_y=input_data['goal_loc_y'],
                        subgoal_loc_x=input_data['subgoal_loc_x'],
                        subgoal_loc_y=input_data['subgoal_loc_y'])
                out.append([prob_feasible, 1 - prob_feasible])

            return out

        class InterpVis(torch.nn.Module):
            """Helper function. The IntegratedGradients function requires a
            class with a 'forward' function."""

            def __init__(self):
                super(InterpVis, self).__init__()

            def forward(self, image):
                # image = np.transpose(image, (2, 0, 1))
                goal_loc_x = input_data['goal_loc_x']
                goal_loc_y = input_data['goal_loc_y']
                subgoal_loc_x = input_data['subgoal_loc_x']
                subgoal_loc_y = input_data['subgoal_loc_y']
                num = image.shape[0]

                def rep(mat):
                    return torch.tensor(np.repeat(np.expand_dims(mat, axis=0), num, axis=0)).float()

                out = model.forward_supervised(
                    {
                        'image': image,
                        'goal_loc_x': rep(goal_loc_x),
                        'goal_loc_y': rep(goal_loc_y),
                        'subgoal_loc_x': rep(subgoal_loc_x),
                        'subgoal_loc_y': rep(subgoal_loc_y),
                    }, device=device)
                out = out.cpu()
                return torch.sigmoid(out[:, :1])

        interp_model = InterpVis()
        image = np.transpose(input_data['image'], (2, 0, 1))
        input_image = torch.tensor(np.expand_dims(image, axis=0)).float()
        baseline = torch.zeros(1, 3, 128, 512).float()

        ig = IntegratedGradients(interp_model)
        attributions, delta = ig.attribute(
            input_image, baseline,
            target=0, return_convergence_delta=True)
        attr_image = np.transpose(attributions.detach().cpu().numpy()[0], (1, 2, 0))
        attr_image = np.sum(attr_image, axis=2)
        attr_image /= np.abs(attr_image).max()
        likelihood = interp_model(input_image).detach().cpu().numpy()[0]
        print(f"Likelihood: {likelihood}")
        print(f"Total Attribution: {attr_image.sum().sum()}")
        print(f"Min Attribution: {attr_image.min()}")
        plt.subplot(211)
        plt.imshow(input_data['image'])
        plt.subplot(212)
        plt.imshow(attr_image, vmin=-0.5, vmax=0.5, cmap='PiYG')
        plt.title(f"Est. Likelihood: {likelihood}")

        img_name = f'{args.image_base_name}_{ind}.png'
        plt.savefig(os.path.join(args.save_dir, img_name))
        plt.close()


if __name__ == "__main__":
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--current_seed', type=int, default=None)
    parser.add_argument('--sp_limit_num', type=int, default=-1)
    parser.add_argument('--logfile_name', type=str, default='logfile.txt')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--explain_at', type=int, default=0)
    parser.add_argument('--datum_name', type=str)
    parser.add_argument('--network_file', type=str)
    parser.add_argument('--image_base_name', type=str)
    args = parser.parse_args()

    if args.network_file is None:
        args.network_file = get_nn_model_name(MODEL_CLASS, args, 'final')

    explain_visuals_with_lime(args)
