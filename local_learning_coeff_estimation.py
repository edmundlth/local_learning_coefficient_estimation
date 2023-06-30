import torch
from copy import deepcopy
import torch
import numpy as np
import time


class Experiment(object):
    def __init__(
        self,
        net,
        trainloader,
        testloader,
        criterion,
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
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.sgld_num_chains = sgld_num_chains
        self.sgld_num_iter = sgld_num_iter
        self.sgld_gamma = sgld_gamma
        self.sgld_noise_std = sgld_noise_std

        self.batch_size = trainloader.batch_size
        self.total_train = len(self.trainloader.dataset)

        self.trainloader_iter = iter(self.trainloader)

        self.records = {
            "lfe": [], 
            "energy": [], 
            "hatlambda": [], 
            "test_error": [], 
            "train_error": []
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
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        return loss, inputs, labels

    def compute_energy(self):
        # this is nL_n,k, sum of the losses at w^* found so far
        energies = []
        with torch.no_grad():
            for data in self.trainloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                energies.append(loss.item() * self.batch_size)
        return sum(energies)

    def compute_local_free_energy(
        self, num_iter=100, num_chains=1, gamma=None, epsilon=1e-5, verbose=True
    ):
        model_copy = deepcopy(self.net)
        gamma_dict = {}
        if gamma is None:
            with torch.no_grad():
                for name, param in model_copy.named_parameters():
                    gamma_val = 100.0 / np.linalg.norm(param)
                    gamma_dict[name] = gamma_val

        og_params = deepcopy(dict(model_copy.named_parameters()))
        chain_Lms = []
        for chain in range(num_chains):
            Lms = []
            for _ in range(num_iter):
                with torch.enable_grad():
                    # call a minibatch loss backward
                    # so that we have gradient of average minibatch loss with respect to w'
                    inputs, labels = self._generate_next_training_batch()
                    outputs = model_copy(inputs)
                    loss = self.criterion(outputs, labels)
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
                f"LFE: {local_free_energy} (std: {chain_std}, n_chain={num_chains})"
            )
        return local_free_energy
    
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
        local_free_energy = self.compute_local_free_energy(
            self.sgld_num_iter,
            self.sgld_num_chains,
            self.sgld_gamma,
            self.sgld_noise_std,
        )
        hatlambda = (local_free_energy - energy) / np.log(self.total_train)
        return local_free_energy, energy, hatlambda

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
            print(f"Finished epoch {epoch + 1} / {num_epoch}, time taken: {time.time() - start_time:.3f}")
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
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            self._record_epoch()
            print(f"Finished epoch {epoch + 1} / {num_epoch}, time taken: {time.time() - start_time:.3f}")
        return self.records
