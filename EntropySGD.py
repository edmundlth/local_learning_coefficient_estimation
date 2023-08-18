from torch.optim.optimizer import Optimizer, required
from copy import deepcopy
import torch
import numpy as np


class EntropySGD(Optimizer):

    def __init__(self, params, eta=0.1, momentum=0, L=20, gamma_0=0.0001, gamma_1=0.001, eta_prime = 0.1, epsilon=0.0001, alpha=0.75, nesterov=False):
        defaults = dict(eta=eta, momentum=momentum, L=L, gamma_0=gamma_0, gamma_1=gamma_1, eta_prime=eta_prime, epsilon=epsilon, alpha=alpha, nesterov=nesterov)
        super(EntropySGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(EntropySGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure):

        if 'step' not in self.state:
            self.state['step'] = 0
        # copy current parameters of network into x_groups
        x_groups = []
        for group in self.param_groups:
            group_copy = dict()
            group_copy['params'] = deepcopy(group['params'])
            x_groups.append(group_copy)
        mu_groups = deepcopy(x_groups)

        # print("x:",x_groups[0]['params'][0])

        L = self.param_groups[0]['L']
        for l in range(L):
            with torch.enable_grad():
                loss, _, _ = closure()
            for group_index, group in enumerate(self.param_groups):
                for param_index, x_prime in enumerate(group['params']):
                    x = x_groups[group_index]['params'][param_index]
                    dx_prime = x_prime.grad.data
                    if group['momentum'] != 0:
                        param_state = self.state[x_prime]
                        if 'inner_loop_velocity' not in param_state:
                            vel = param_state['inner_loop_velocity'] = torch.clone(dx_prime).detach()
                        else:
                            vel = param_state['inner_loop_velocity']
                            if l == 0:
                                vel.fill_(0)
                            vel.mul_(group['momentum']).add_(dx_prime)
                        if group['nesterov']:
                            dx_prime.add_(vel, alpha=group['momentum'])
                        else:
                            dx_prime = vel
                    gamma = group['gamma_0'] * ((1+group['gamma_1']) ** self.state['step'])
                    dx_prime.add_(x.data - x_prime.data, alpha=-gamma)
                    x_prime.data.add_(dx_prime, alpha=-group['eta_prime'])
                    gaussian_noise = torch.empty_like(x_prime)
                    gaussian_noise.normal_()
                    x_prime.data.add_(gaussian_noise, alpha=(np.sqrt(group['eta_prime']) * group['epsilon']))
                    mu = mu_groups[group_index]['params'][param_index]
                    mu.data.mul_(1-group['alpha'])
                    mu.data.add_(x_prime.data, alpha=group['alpha'])

        # print("mu:", mu_groups[0]['params'][0])

        for group_index, group in enumerate(self.param_groups):
            for param_index, p in enumerate(group['params']):
                x = x_groups[group_index]['params'][param_index]
                mu = mu_groups[group_index]['params'][param_index]
                p.data.copy_(x.data)
                gradient = x.data - mu.data
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'outer_loop_velocity' not in param_state:
                        vel = param_state['outer_loop_velocity'] = torch.clone(gradient).detach()
                    else:
                        vel = param_state['outer_loop_velocity']
                        vel.mul_(group['momentum']).add_(gradient)
                    if group['nesterov']:
                        gradient.add_(vel, alpha=group['momentum'])
                    else:
                        gradient = vel
                p.data.add_(gradient, alpha=-group['eta'])

        # print("step:", self.state['step'])
        self.state['step'] += 1

        # print("x after:", x_groups[0]['params'][0])

        return loss
