import argparse
import itertools


def expand_dictionary(original_dict, keys_to_expand):
    lists_to_expand = [original_dict[key] for key in keys_to_expand]
    combinations = itertools.product(*lists_to_expand)
    expanded_dicts = []
    for combo in combinations:
        new_dict = original_dict.copy()
        for key, value in zip(keys_to_expand, combo):
            new_dict[key] = value
        expanded_dicts.append(new_dict)
    return expanded_dicts



def common_cmd_parser():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=512)
    parser.add_argument(
        "--num_training_data",
        help="Training data set size. If none, full data is used.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seeds",
        help="Experiments are repeated for each RNG seed given",
        nargs="+",
        type=int,
        default=[0],
    )
    parser.add_argument("--epochs", help="epochs", type=int, default=10)
    parser.add_argument(
        "--num_gradient_step",
        help="Total number of gradient steps taken. If specified, ignore --epoch and calculate epoch number from dataset size and batch size.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--optimizer", help="sgd | entropy-sgd | adam", type=str, default="sgd"
    )
    parser.add_argument(
        "--outputdir",
        help="Path to output directory. Create if not exist.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_result",
        help="save experiment result if specified.",
        action="store_true",
    )
    parser.add_argument(
        "--save_model",
        help="save trained pytorch model if specified.",
        action="store_true",
    )
    parser.add_argument(
        "--save_plot", help="save plots to file if specified", action="store_true"
    )
    parser.add_argument(
        "--show_plot", help="plt.show() if specified.", action="store_true"
    )

    # parser.add_argument('-m', help='mnistfc | mnistconv | allcnn', type=str, default='mnistconv')
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.001)
    parser.add_argument(
        "--sgld_gamma",
        type=float,
        default=None,
        help=" 1/variance of gaussian distribution used for SGLD sampling. If not specified, it is chosen automatically by inspecting the norm of loss gradient.",
    )
    parser.add_argument(
        "--sgld_num_chains",
        type=int,
        default=4,
        help="number of independent SGLD chains for estimating local free energy.",
    )
    parser.add_argument(
        "--sgld_num_iter", type=int, default=100, help="number of SGLD steps."
    )
    parser.add_argument(
        "--sgld_noise_std",
        type=float,
        default=1e-5,
        help="standard deviation of gaussian noise in SGLD.",
    )

    parser.add_argument(
        "--data_rootdir",
        type=str,
        default="./data",
        help="Directory where MNIST data is stored or download to.",
    )
    return parser
