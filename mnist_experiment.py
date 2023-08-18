import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import json
import os
import concurrent

from EntropySGD import EntropySGD
from utils import common_cmd_parser
from local_learning_coeff_estimation import Experiment


# TODO:
# - [ ] Vary `n`. Plot estimated RLCT against `n` and see if it stabilise.
# - [x] Sensitivity analysis on the main hyperparameters: sgld_gamma, sgld_noise_std, sgld_num_iter.
# - [ ] Might need to implement accept-reject step to SGLD sampling. Random Gaussian noise added to very high dimensional space might result in points with very high potential energy. This might be cause of the observed massive variance of lambda hat when `sgld_noise_std` is varied.


class MNISTNet(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes=[1024, 1024],
        input_dim=28 * 28,
        output_dim=10,
        activation=F.relu,
        with_bias=True,
    ):
        super(MNISTNet, self).__init__()
        self.input_dim = input_dim
        # use [0] to specify empty layer.
        hidden_layer_sizes = [val for val in hidden_layer_sizes if val != 0]
        self.layer_sizes = [input_dim] + hidden_layer_sizes + [output_dim]
        self.activation = activation
        self.with_bias = with_bias
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            dim_in, dim_out = self.layer_sizes[i : i + 2]
            self.layers.append(nn.Linear(dim_in, dim_out, bias=self.with_bias).float())

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def parse_commandline():
    parser = common_cmd_parser()
    parser.add_argument(
        "--hidden_layer_sizes",
        help="size of feedforward layers other than the input and output layer",
        nargs="+",
        type=int,
        default=[1024, 1024],
    )
    parser.add_argument(
        "--max_workers",
        help="Maximum number of parallel process running the experiments independently for each given rngseed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--esgd_L",
        help="The L parameter to be used by the EntropySGD optimiser",
        type=int,
        default=5,
    )
    return parser


def main(args, rngseed):
    print(f"Starting experiment for rngseed: {rngseed}")

    def _get_save_filepath(filename):
        filepath = os.path.join(args.outputdir, f"{rngseed}_{filename}")
        print(f"Filepath constructed: {filepath}")
        return filepath

    torch.manual_seed(rngseed)
    np.random.seed(rngseed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ########################################################################
    # Define dataset, network and loss function
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = torchvision.datasets.MNIST(
        root=args.data_rootdir, train=True, download=True, transform=transform
    )
    if args.num_training_data is not None and args.num_training_data < len(trainset):
        random_indices = torch.randperm(len(trainset))[: args.num_training_data]
        trainset = torch.utils.data.Subset(trainset, random_indices)

    testset = torchvision.datasets.MNIST(
        root=args.data_rootdir, train=False, download=True, transform=transform
    )
    net = MNISTNet(
        args.hidden_layer_sizes,
        input_dim=28 * 28,
        output_dim=10,
        activation=F.relu,
        with_bias=True,
    )
    # net = CNN()
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    n_train = len(trainloader.dataset)
    print(f"Number of training data: {n_train}")

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    n_test = len(testloader.dataset)
    print(f"Number of testing data: {n_test}")

    net.to(device)
    print(net)
    network_param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {network_param_count}")

    ########################################################################
    # Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.
    num_step_per_epoch = n_train / trainloader.batch_size
    if args.num_gradient_step is not None:
        num_epoch = int(
            np.round(args.num_gradient_step / num_step_per_epoch, decimals=0)
        )
        num_epoch = max(num_epoch, 1)
    else:
        num_epoch = args.epochs
    print(f"Num epoch: {num_epoch}")
    print(f"Num steps per epoch: {num_step_per_epoch}")

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=0.9, nesterov=True
        )
        experiment = Experiment(
            net,
            trainloader,
            testloader,
            optimizer,
            device,
            loss_fn=criterion,
            sgld_num_chains=args.sgld_num_chains,
            sgld_num_iter=args.sgld_num_iter,
            sgld_gamma=args.sgld_gamma,
            sgld_noise_std=args.sgld_noise_std,
        )
        experiment.run_sgd(num_epoch)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        experiment = Experiment(
            net,
            trainloader,
            testloader,
            optimizer,
            device,
            loss_fn=criterion,
            sgld_num_chains=args.sgld_num_chains,
            sgld_num_iter=args.sgld_num_iter,
            sgld_gamma=args.sgld_gamma,
            sgld_noise_std=args.sgld_noise_std,
        )
        experiment.run_sgd(num_epoch)
    elif args.optimizer.lower() in ["entropy-sgd", "esgd"]:
        optimizer = EntropySGD(
            net.parameters(), eta=args.lr, momentum=0.9, nesterov=False, L=args.esgd_L
        )
        print(device)
        experiment = Experiment(
            net,
            trainloader,
            testloader,
            optimizer,
            device,
            loss_fn=criterion,
            sgld_num_chains=args.sgld_num_chains,
            sgld_num_iter=args.sgld_num_iter,
            sgld_gamma=args.sgld_gamma,
            sgld_noise_std=args.sgld_noise_std,
        )
        experiment.run_entropy_sgd(args.esgd_L, num_epoch)

    print("Finished Training")
    # _map_float = lambda x: list(map(float, x))
    result = {
        "rngseed": rngseed,
        "network_param_count": network_param_count,
        "layers": args.hidden_layer_sizes,
        "optimizer_type": args.optimizer,
        "sgld_gamma": args.sgld_gamma,
        "sgld_num_iter": args.sgld_num_iter,
        "sgld_num_chains": args.sgld_num_chains,
        "sgld_noise_std": args.sgld_noise_std,
        "num_epoch": num_epoch,
        "batch_size": args.batch_size,
        "n_train": n_train,
        "n_test": n_test,
        "lr": args.lr,
    }
    result.update(experiment.records)
    print(json.dumps(result, indent=2))
    if args.save_result:
        outfilepath = _get_save_filepath("result.json")
        print(f"Saving result at: {outfilepath}")
        with open(outfilepath, "w") as outfile:
            json.dump(result, outfile, indent=2)

    ########################################################################
    # Plotting
    # ^^^^^^^^^^^^^^^^^^^^
    l_epochs = list(range(num_epoch))
    lfes = experiment.records["lfe"]
    energies = experiment.records["energy"]
    hatlambdas = experiment.records["hatlambda"]
    test_errors = experiment.records["test_error"]
    train_errors = experiment.records["train_error"]

    print("Generating plots...")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
    ax = axes[0]
    ax.plot(l_epochs, lfes, "x--", label="local free energy")
    ax.plot(l_epochs, energies, "x--", label="$nL_n(w_t)$")
    ax.set_xlabel("epoch")
    ax.legend()

    ax = axes[1]
    ax.plot(
        l_epochs,
        np.array(hatlambdas) * np.log(n_train),
        "x--",
        color="green",
        label="$\lambda(w_t) \log n$",
    )
    ax.set_xlabel("epoch")
    ax.legend()
    plt.suptitle(f"{args.optimizer} final hatlambda {hatlambdas[-1]}")

    ax = axes[2]
    ax.plot(l_epochs, test_errors, "kx--", label="test")
    ax.plot(l_epochs, train_errors, "kx--", label="train")
    ax.set_xlabel("epoch")
    ax.set_ylabel("percent error")
    ax.legend()

    if args.outputdir is not None:
        print("Saving plots....")
        fig.savefig(_get_save_filepath("plots.png"))
    if args.show_plot:
        plt.show()

    ########################################################################
    # Let's quickly save our trained model:
    if args.save_model:
        torch.save(net.state_dict(), _get_save_filepath("model.pth"))

    return result


if __name__ == "__main__":
    args = parse_commandline().parse_args()
    print(f"Commandline arguments:\n{vars(args)}")
    if args.outputdir:
        os.makedirs(args.outputdir, exist_ok=True)

    # use parallel workers on given list of rngseeds
    if args.max_workers is not None and args.max_workers > 1 and len(args.seeds) > 1:
        input1 = [args for _ in range(len(args.seeds))]
        input2 = args.seeds
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            results = list(executor.map(main, input1, input2))
    else:
        for i, rngseed in enumerate(args.seeds):
            print(f"Running seed {i + 1} / {len(args.seeds)} with value: {rngseed}")
            main(args, rngseed)
            print(f"Finished")
