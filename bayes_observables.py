import numpy as np
from scipy.special import logsumexp

# dimension of `loglike_array` in all the following functions are 
# (num data samples X, num mcmc sampled parameters w)
def compute_bayesian_loss(loglike_array):
    num_mcmc_samples = loglike_array.shape[1]
    result = -np.mean(logsumexp(loglike_array, b=1 / num_mcmc_samples, axis=1))
    return result

def compute_gibbs_loss_vectorised(loglike_fn, data, parameters):
    gg = []
    for param in parameters:
        gg.append(loglike_fn(param, data))
    return -np.mean(gg)

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


def convert_expectation_temp(obs_array, loglike_array, old_itemp, new_itemp):
    """
    Compute expectation of an observable over a tempered posterior with `new_itemp`
    given loglikelihood and the observable evaluated on samples from posterior at `old_itemp`. 

    Reference: Equation 20 of "A Widely Applicable Bayesian Information Criterion" by Watanabe in JMLR. 
    """
    delta_itemp = new_itemp - old_itemp
    weights = delta_itemp * loglike_array
    result = logsumexp(weights, b=obs_array) - logsumexp(weights)
    return np.exp(result)