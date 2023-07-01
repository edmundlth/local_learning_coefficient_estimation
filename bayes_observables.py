import numpy as np
from scipy.special import logsumexp

# dimension of `loglike_array` in all the following functions are 
# (num data samples X, num mcmc sampled parameters w)
def compute_bayesian_loss(loglike_array):
    num_mcmc_samples = loglike_array.shape[1]
    result = -np.mean(logsumexp(loglike_array, b=1 / num_mcmc_samples, axis=1))
    return result


def compute_gibbs_loss(loglike_array):
    gerrs = np.mean(loglike_array, axis=0)
    gg = np.mean(gerrs)
    return -gg


def compute_functional_variance(loglike_array):
    # variance over posterior samples and averaged over dataset.
    # V = 1/n \sum_{i=1}^n Var_w(\log p(X_i | w))
    result = np.mean(np.var(loglike_array, axis=1))
    return result


def compute_waic(loglike_array):
    func_var = compute_functional_variance(loglike_array)
    bayes_train_loss = compute_bayesian_loss(loglike_array)
    return bayes_train_loss + func_var


def compute_wbic(tempered_loglike_array):
    return -np.mean(np.sum(tempered_loglike_array, axis=0))
