import decimal
import torch
from copy import deepcopy
import torch
import numpy as np
import time
from scipy.special import logsumexp
import functools


class Experiment(object):
    def __init__(
        self,
        net,
        trainloader,
        testloader,
        optimizer,
        device,
        sgld_num_chains=4,
        sgld_num_iter=100,
        sgld_gamma=None,
        sgld_noise_std=1e-5,
    ):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.device = device

        self.sgld_num_chains = sgld_num_chains
        self.sgld_num_iter = sgld_num_iter
        self.sgld_gamma = sgld_gamma
        self.sgld_noise_std = sgld_noise_std

        self.batch_size = trainloader.batch_size
        self.total_train = len(self.trainloader.dataset)

        self.trainloader_iter = iter(self.trainloader)

        self.stateful_loader = None

        self.records = {
            "lfe": [],
            "energy": [],
            "hatlambda": [],
            "test_error": [],
            "train_error": [],
        }

    def eval(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def _generate_next_training_batch(self):
        """
        Generate a batch of data from the trainloader for iterative process that
        doesn't necessarily loop through the entire dataset.
        We are making a stateful iterator here and refresh the iterator when it hits
        StopIteration.
        """
        try:
            data = next(self.trainloader_iter)
        except StopIteration:
            self.trainloader_iter = iter(self.trainloader)
            data = next(self.trainloader_iter)
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        return inputs, labels

    def closure(self):
        inputs, labels = self._generate_next_training_batch()
        self.optimizer.zero_grad()
        outputs = self.net(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        return loss, inputs, labels
        
    @functools.lru_cache(maxsize=128)
    def compute_energy(self):
        # this is nL_n,k, sum of the losses at w^* found so far
        energies = []
        with torch.no_grad():
            for data in self.trainloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs, labels=labels)
                loss = outputs.loss
                energies.append(loss.item() * self.batch_size)
        return sum(energies)

    def _generate_next_batch(self, dataloader):
        if self.stateful_loader is None:
            self.stateful_loader = iter(dataloader)
        try:
            data = next(self.stateful_loader)
        except StopIteration:
            self.stateful_loader = iter(dataloader)
            data = next(self.stateful_loader)
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        return inputs, labels

    def run_sgld_chains(self, num_iter, dataloader, gamma=None, epsilon=1e-5):
        model_copy = deepcopy(self.net)
        gamma_dict = {}
        if gamma is None:
            with torch.no_grad():
                for name, param in model_copy.named_parameters():
                    gamma_val = 100.0 / torch.linalg.norm(param)
                    gamma_dict[name] = gamma_val
        og_params = deepcopy(dict(model_copy.named_parameters()))

        losses = []
        for _ in range(num_iter):
            with torch.enable_grad():
                # call a minibatch loss backward
                # so that we have gradient of average minibatch loss with respect to w'
                inputs, labels = self._generate_next_batch(dataloader)
                outputs = model_copy(inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
            for name, w in model_copy.named_parameters():
                w_og = og_params[name]
                dw = -w.grad.data / np.log(self.total_train) * self.total_train
                if gamma is None:
                    prior_weight = gamma_dict[name]
                else:
                    prior_weight = gamma
                dw.add_(w.data - w_og.data, alpha=-prior_weight)
                w.data.add_(dw, alpha=epsilon / 2)
                gaussian_noise = torch.empty_like(w)
                gaussian_noise.normal_()
                w.data.add_(gaussian_noise, alpha=np.sqrt(epsilon))
                w.grad.zero_()

            yield model_copy

    def compute_local_free_energy(
        self,
        num_iter=100,
        num_chains=1,
        gamma=None,
        epsilon=1e-5,
        verbose=True,
        chain_itemps=None,
    ):
        model_copy = deepcopy(self.net)
        gamma_dict = {}
        if gamma is None:
            with torch.no_grad():
                for name, param in model_copy.named_parameters():
                    gamma_val = 100.0 / torch.linalg.norm(param)
                    gamma_dict[name] = gamma_val
        if chain_itemps is None:
            chain_itemps = []
        og_params = deepcopy(dict(model_copy.named_parameters()))

        chain_Lms = []
        for chain in range(num_chains):
            model_copy = deepcopy(self.net)
            Lms = []
            for _ in range(num_iter):
                with torch.enable_grad():
                    # call a minibatch loss backward
                    # so that we have gradient of average minibatch loss with respect to w'
                    inputs, labels = self._generate_next_training_batch()
                    outputs = model_copy(inputs, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                for name, w in model_copy.named_parameters():
                    w_og = og_params[name]
                    dw = -w.grad.data / np.log(self.total_train) * self.total_train
                    if gamma is None:
                        prior_weight = gamma_dict[name]
                    else:
                        prior_weight = gamma
                    dw.add_(w.data - w_og.data, alpha=-prior_weight)
                    w.data.add_(dw, alpha=epsilon / 2)
                    gaussian_noise = torch.empty_like(w)
                    gaussian_noise.normal_()
                    w.data.add_(gaussian_noise, alpha=np.sqrt(epsilon))
                    w.grad.zero_()
                Lms.append(loss.item())
            chain_Lms.append(Lms)
            if verbose:
                print(f"Chain {chain + 1}: L_m = {np.mean(Lms)}")

        chain_Lms = np.array(chain_Lms)
        local_free_energy = self.total_train * np.mean(chain_Lms)
        if verbose:
            chain_std = np.std(self.total_train * np.mean(chain_Lms, axis=1))
            print(
                f"LFE: {local_free_energy:.2e} (std: {chain_std:.2e}, n_chain={num_chains})"
            )
        return local_free_energy, chain_std

    def _record_epoch(self):
        local_free_energy, energy, hatlambda = self.compute_fenergy_energy_rlct()
        self.records["lfe"].append(local_free_energy)
        self.records["energy"].append(energy)
        self.records["hatlambda"].append(hatlambda)
        test_err = 1 - self.eval(self.testloader)
        train_err = 1 - self.eval(self.trainloader)

        self.records["test_error"].append(test_err)
        self.records["train_error"].append(train_err)
        epoch = len(self.records["test_error"])
        print(
            f"Epoch: {epoch} "
            f"energy: {energy:.4f} "
            f"hatlambda: {hatlambda:.4f} "
            f"test error: {test_err:.4f} "
            f"train error: {train_err:.4f} "
        )
        return

    def compute_fenergy_energy_rlct(self):
        energy = self.compute_energy()
        local_free_energy, local_free_energy_std = self.compute_local_free_energy(
            self.sgld_num_iter,
            self.sgld_num_chains,
            self.sgld_gamma,
            self.sgld_noise_std,
        )
        lfe_standard_error = local_free_energy_std / (self.sgld_num_chains) ** 0.5

        local_free_energy_lower_bound = local_free_energy - lfe_standard_error * 2
        local_free_energy_upper_bound = local_free_energy + lfe_standard_error * 2

        hatlambda = (local_free_energy - energy) / np.log(self.total_train)
        hatlambda_lower = (local_free_energy_lower_bound - energy) / np.log(
            self.total_train
        )
        hatlambda_upper = (local_free_energy_upper_bound - energy) / np.log(
            self.total_train
        )
        return local_free_energy, energy, hatlambda, hatlambda_lower, hatlambda_upper

    def run_entropy_sgd(self, esgd_L, num_epoch):
        print("Running Entropy-SGD optimizer")
        # errors, lfes, energies, lmbdas = [], [], [], []

        for epoch in range(num_epoch):  # loop over the dataset multiple times
            start_time = time.time()
            for _ in range(len(self.trainloader) // esgd_L):
                # len(self.trainloader) is the number of minibatches,
                # division by L is to make the same number of passes as plain SGD below
                self.optimizer.step(self.closure)
            self._record_epoch()
            print(
                f"Finished epoch {epoch + 1} / {num_epoch}, time taken: {time.time() - start_time:.3f}"
            )
        return self.records

    def run_sgd(self, num_epoch):
        print("Running SGD optimizer")
        # SGD should be run L times longer to be fair comparison with entropy-SGD
        # loop over the dataset multiple times
        for epoch in range(num_epoch):
            start_time = time.time()
            for data in self.trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
            self._record_epoch()
            print(
                f"Finished epoch {epoch + 1} / {num_epoch}, time taken: {time.time() - start_time:.3f}"
            )
        return self.records

    def compute_bayes_loss(self, dataloader, num_sgld_iter=50):
        rec = []
        for data in dataloader:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            # using `log(sum(exp(array)) / m) = log(sum(exp(array - max(array))) / m) + max(array)`
            # to pre-emptively avoid overflow errors. Probably this is a non-issue in this application,
            # but why not?
            val_array = []
            for model_copy in self.run_sgld_chains(num_sgld_iter):
                outputs = model_copy(inputs, labels=labels)
                val = outputs.loss
                val_array.append(val)
            
            max_val = max(val_array)
            val_array = torch.tensor(val_array)
            lse = torch.logsumexp(val_array - max_val, dim=0) + max_val - np.log(num_sgld_iter)
            rec.append(lse.item())
        return -np.mean(rec)

    def compute_gibbs_loss(self, dataloader, num_sgld_iter=50):
        avg_losses = []
        for model_copy in self.run_sgld_chains(num_sgld_iter):
            losses = []
            for data in dataloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model_copy(inputs, labels=labels)
                losses.append(outputs.loss)
            avg_losses.append(np.mean(losses))
        return -np.mean(avg_losses)

    def compute_waic(self):
        raise NotImplementedError
