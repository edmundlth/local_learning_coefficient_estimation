import decimal
import torch
import torch.nn as nn
from copy import deepcopy
import torch
import numpy as np
import time
from engineering_notation import EngNumber
from scipy.special import logsumexp
import gc


class Experiment(object):
    def __init__(
        self,
        net,
        trainloader,
        testloader,
        optimizer,
        device,
        loss_fn=None,
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
        if loss_fn is None:
            # for BERT model compatibility
            self.loss_fn = lambda outputs, labels: outputs.loss
            self._wrapped_forward = lambda net, inputs, labels: net(
                inputs, labels=labels
            )
        else:
            # this is for MNIST or other classification tasks.
            self.loss_fn = loss_fn
            self._wrapped_forward = lambda net, inputs, labels: net(inputs)

        self.sgld_num_chains = sgld_num_chains
        self.sgld_num_iter = sgld_num_iter
        self.sgld_gamma = sgld_gamma
        self.sgld_noise_std = sgld_noise_std

        self.batch_size = trainloader.batch_size
        self.total_train = len(self.trainloader.dataset)

        self.trainloader_iter = iter(self.trainloader)
        self.stateful_loader = None

        # TODO: THIS IS A HACK. We are just storing these in memory to use in computing functional variance.
        self.all_inputs = []
        self.all_labels = []
        # get all training data
        for batch_data, batch_labels in iter(self.trainloader):
            self.all_inputs.append(batch_data)
            self.all_labels.append(batch_labels)
        print(self.device)
        self.all_inputs = torch.cat(self.all_inputs).to(self.device)
        self.all_labels = torch.cat(self.all_labels).to(self.device)
        self.records = {}

    def save_to_epoch_record(self, key, value):
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(value)
        return

    def eval(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self._wrapped_forward(self.net, inputs, labels)
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

    def closure(self):
        inputs, labels = self._generate_next_training_batch()
        self.optimizer.zero_grad()
        outputs = self._wrapped_forward(self.net, inputs, labels)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        return loss, inputs, labels

    def compute_energy(self):
        # this is nL_n,k, sum of the losses at w^* found so far
        energies = []
        with torch.no_grad():
            for data in self.trainloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self._wrapped_forward(self.net, inputs, labels)
                loss = self.loss_fn(outputs, labels)
                energies.append(loss.item() * inputs.shape[0])
        return sum(energies)

    def run_sgld_chains(
        self, num_iter, dataloader, gamma=None, epsilon=1e-5, itemp=None
    ):
        if itemp is None:
            itemp = 1 / np.log(self.total_train)
        model_copy = deepcopy(self.net)
        gamma_dict = {}
        if gamma is None:
            with torch.no_grad():
                for name, param in model_copy.named_parameters():
                    gamma_val = 100.0 / torch.linalg.norm(param)
                    gamma_dict[name] = gamma_val
        og_params = deepcopy(dict(model_copy.named_parameters()))

        for _ in range(num_iter):
            with torch.enable_grad():
                # call a minibatch loss backward
                # so that we have gradient of average minibatch loss with respect to w'
                inputs, labels = self._generate_next_batch(dataloader)
                outputs = self._wrapped_forward(model_copy, inputs, labels)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
            for name, w in model_copy.named_parameters():
                w_og = og_params[name]
                dw = -w.grad.data * self.total_train * itemp
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

    def compute_functional_variance(
        self, num_iter=100, num_chains=1, gamma=100.0, epsilon=1e-5
    ):
        print("Compute functional variance.")
        with torch.no_grad():
            loss_fn_noreduce = nn.CrossEntropyLoss(reduction="none")
            m = 0
            loss_sum = torch.zeros(len(self.all_inputs))
            loss_sum_sq = torch.zeros(len(self.all_inputs))
            for _ in range(num_chains):
                sgld_generator = self.run_sgld_chains(
                    num_iter=num_iter,
                    dataloader=self.trainloader,
                    gamma=gamma,
                    epsilon=epsilon,
                    itemp=1.0,
                )
                for model_copy in sgld_generator:
                    m += 1
                    outputs = self._wrapped_forward(
                        model_copy, self.all_inputs, self.all_labels
                    )
                    losses = loss_fn_noreduce(outputs, self.all_labels)
                    loss_sum += losses
                    loss_sum_sq += losses * losses
            variance = (loss_sum_sq - loss_sum * loss_sum / m) / (m - 1)
            func_var = torch.sum(variance)
        return func_var

    def compute_local_free_energy(
        self,
        num_iter=100,
        num_chains=1,
        gamma=None,
        epsilon=1e-5,
        itemp=None,
        verbose=True,
    ):
        if itemp is None:
            itemp = 1 / np.log(self.total_train)
        with torch.no_grad():
            chain_Lms = []
            for chain in range(num_chains):
                sgld_generator = self.run_sgld_chains(
                    num_iter=num_iter,
                    dataloader=self.trainloader,
                    gamma=gamma,
                    epsilon=epsilon,
                    itemp=itemp,
                )
                Lms = []
                for model_copy in sgld_generator:
                    inputs, labels = self._generate_next_training_batch()
                    outputs = self._wrapped_forward(model_copy, inputs, labels)
                    loss = self.loss_fn(outputs, labels)
                    Lms.append(loss.item())
                chain_Lms.append(Lms)
                if verbose:
                    print(f"Chain {chain + 1}: L_m = {np.mean(Lms)}")

        chain_Lms = np.array(chain_Lms)
        local_free_energy = self.total_train * np.mean(chain_Lms)
        if verbose:
            chain_std = np.std(self.total_train * np.mean(chain_Lms, axis=1))
            print(
                f"LFE: {EngNumber(local_free_energy)} (std: {EngNumber(chain_std)}, n_chain={num_chains})"
            )
        return local_free_energy, chain_std

    # def compute_local_free_energy2(
    #     self,
    #     num_iter=100,
    #     num_chains=1,
    #     gamma=None,
    #     epsilon=1e-5,
    #     verbose=True,
    # ):
    #     model_copy = deepcopy(self.net)
    #     gamma_dict = {}
    #     if gamma is None:
    #         with torch.no_grad():
    #             for name, param in model_copy.named_parameters():
    #                 gamma_val = 100.0 / torch.linalg.norm(param)
    #                 gamma_dict[name] = gamma_val

    #     og_params = deepcopy(dict(model_copy.named_parameters()))
    #     chain_Lms = []
    #     for chain in range(num_chains):
    #         model_copy = deepcopy(self.net)
    #         Lms = []
    #         for _ in range(num_iter):
    #             with torch.enable_grad():
    #                 # call a minibatch loss backward
    #                 # so that we have gradient of average minibatch loss with respect to w'
    #                 inputs, labels = self._generate_next_training_batch()
    #                 outputs = self._wrapped_forward(model_copy, inputs, labels)
    #                 loss = self.loss_fn(outputs, labels)
    #                 loss.backward()
    #             for name, w in model_copy.named_parameters():
    #                 w_og = og_params[name]
    #                 dw = -w.grad.data / np.log(self.total_train) * self.total_train
    #                 if gamma is None:
    #                     prior_weight = gamma_dict[name]
    #                 else:
    #                     prior_weight = gamma
    #                 dw.add_(w.data - w_og.data, alpha=-prior_weight)
    #                 w.data.add_(dw, alpha=epsilon / 2)
    #                 gaussian_noise = torch.empty_like(w)
    #                 gaussian_noise.normal_()
    #                 w.data.add_(gaussian_noise, alpha=np.sqrt(epsilon))
    #                 w.grad.zero_()
    #             Lms.append(loss.item())
    #         chain_Lms.append(Lms)
    #         if verbose:
    #             print(f"Chain {chain + 1}: L_m = {np.mean(Lms)}")

    #     chain_Lms = np.array(chain_Lms)
    #     local_free_energy = self.total_train * np.mean(chain_Lms)
    #     if verbose:
    #         chain_std = np.std(self.total_train * np.mean(chain_Lms, axis=1))
    #         print(
    #             f"LFE: {EngNumber(local_free_energy)} (std: {EngNumber(chain_std)}, n_chain={num_chains})"
    #         )
    #     return local_free_energy, chain_std

    def _record_epoch(self):
        (
            local_free_energy,
            energy,
            hatlambda,
            hatlambda_lower,
            hatlambda_upper,
        ) = self.compute_fenergy_energy_rlct()
        test_err = 1 - self.eval(self.testloader)
        train_err = 1 - self.eval(self.trainloader)

        # func_var = self.compute_functional_variance(
        #     num_iter=self.sgld_num_iter,
        #     num_chains=self.sgld_num_chains,
        #     gamma=self.sgld_gamma,
        #     epsilon=self.sgld_noise_std,
        # )
        # func_var = float(func_var.item())
        # nu = func_var / 2
        # self.save_to_epoch_record("func_var", func_var)
        # self.save_to_epoch_record("nu", nu)

        self.save_to_epoch_record("lfe", local_free_energy)
        self.save_to_epoch_record("energy", energy)
        self.save_to_epoch_record("hatlambda", hatlambda)
        self.save_to_epoch_record("test_error", test_err)
        self.save_to_epoch_record("train_error", train_err)
        self.save_to_epoch_record("hatlambda_lower", hatlambda_lower)
        self.save_to_epoch_record("hatlambda_upper", hatlambda_upper)

        epoch = len(self.records["test_error"])
        print(
            f"Epoch: {epoch} "
            f"energy: {energy:.4f} "
            f"hatlambda: {hatlambda:.4f} "
            f"test error: {np.format_float_scientific(test_err, precision=3)} "
            f"train error: {np.format_float_scientific(train_err, precision=3)} "
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
                outputs = self._wrapped_forward(self.net, inputs, labels)
                loss = self.loss_fn(outputs, labels)
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
            # we are approximating logsumexp(array, b=1/m) with max(array - log(m))
            max_val = -np.inf
            for model_copy in self.run_sgld_chains(num_sgld_iter):
                outputs = self._wrapped_forward(model_copy, inputs, labels)
                val = self.loss_fn(outputs, labels) - np.log(num_sgld_iter)
                if val > max_val:
                    max_val = val
            rec.append(max_val)
        return -np.mean(rec)

    def compute_gibbs_loss(self, dataloader, num_sgld_iter=50):
        avg_losses = []
        for model_copy in self.run_sgld_chains(num_sgld_iter):
            losses = []
            for data in dataloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self._wrapped_forward(model_copy, inputs, labels)
                losses.append(self.loss_fn(outputs, labels))
            avg_losses.append(np.mean(losses))
        return -np.mean(avg_losses)

    def compute_waic(self):
        raise NotImplementedError
