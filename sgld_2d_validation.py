import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

from utils import expand_dictionary
from toy_potential import SGLDExperiment

from sacred import Experiment
from sacred.observers import MongoObserver
ex = Experiment('sgld_toy2dpotential_experiment')
ex.observers.append(MongoObserver(url='localhost:27017', db_name='sgld_toy2dpotential_experiment'))


@ex.config
def default_config():
    # example configuration variables
    outputdir = None
    sampler_type = "sgld"
    sigma = 1.0
    prior_sigma = 1.0
    monomial_prior_spec = [((i, j), None) for i in range(1, 4) for j in range(i, 4)]
    prior_exponents = None
    num_training_data = 100
    epsilon = 0.01
    num_steps = 1000
    no_plot = True
    w_init = [0.1, 0.1] # chosen to be nonzero so that we don't run into nan issues with monomial prior. 
    seed = 0
    # Add more configuration variables here

@ex.automain
def run_experiment(
    _run,
    outputdir, 
    sampler_type, 
    sigma, 
    prior_sigma, 
    monomial_prior_spec, 
    num_training_data, 
    epsilon, 
    num_steps, 
    w_init,
    seed, 
    no_plot
):
    # Initialize parameters for the experiment
    key = random.PRNGKey(seed)
    np.random.seed(seed)    
    w_init = jnp.array(w_init)
    itemp = 1 / jnp.log(num_training_data)
    _run.info["result"] = []
    
    num_plots = len(monomial_prior_spec)
    nrow = 3
    ncol = num_plots // nrow + (num_plots % nrow != 0)
    
    if not no_plot:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
        axes = np.ravel(axes)
    else:
        fig, axes = None, None

    for c in range(len(monomial_prior_spec)):
        monomial_exps, prior_exps = monomial_prior_spec[c]
        # Expt set up
        k1, k2 = monomial_exps
        if prior_exps is not None:
            h1, h2 = prior_exps
            true_lambda = min([(h1 + 1) / (2 * k1), (h2 + 1) / (2 * k2)])
        else:
            true_lambda = 1 / (2 * max(k1, k2))

        polynomial = jax.jit(lambda w: (w[0] ** k1) * (w[1] ** k2))
        symhatlambda = "$\hat{\lambda}$"
        sympotential = f"$w_1^{k1}w_2^{k2}$"
        
        
        key, subkey = random.split(key)
        experiment = SGLDExperiment(
            polynomial, sigma, prior_sigma, key, num_training_data, prior_exponents=prior_exps
        )

        # For SGLD
        if sampler_type == "sgld":
            tempered_samples = experiment.run_sgld(w_init=w_init, epsilon=epsilon, num_steps=num_steps, itemp=itemp)
        elif sampler_type == "mcmc":
            tempered_samples = experiment.run_mcmc(num_samples=num_steps, itemp=itemp)
        lambdahat = experiment.compute_lambdahat(tempered_samples, true_w=(0, 0))
        func_var = experiment.compute_functional_variance(tempered_samples)
        sing_fluc_est = func_var * itemp / 2
        # lambdahat = experiment.compute_multitemp_lambdahat(trajectory)
        print(f"{sampler_type}, {monomial_exps}, {prior_exps}, itemp:{itemp:.4f}, lambdahat:{lambdahat:.4f}, lambda:{true_lambda:.4f}, nu:{sing_fluc_est}")
        
        result = {
            "monomial_exponents": [k1, k2],
            "prior_exponents": list(prior_exps) if prior_exps is not None else None,
            "true_lambda": true_lambda,
            "lambdahat": float(lambdahat), 
            "func_var": float(func_var), 
            "nu": float(sing_fluc_est)
        }
        _run.info["result"].append(result)

        if not no_plot:
            ax = axes[c]
            experiment.plot(tempered_samples, ax)
            title_string = f"{sympotential}, {symhatlambda}={lambdahat:.4f}, $\lambda$={true_lambda:.4f}"
            ax.set_title(title_string, fontsize=8)
    
    if not no_plot:
        s = "$w_1^{k_1}w_2^{k_2}$"
        # HACK: disabled suptitle. 
        # fig.suptitle(
        #     f"{s}, n={num_training_data}, $\sigma=${sigma}, $\sigma_p=${prior_sigma}, chain_len={num_steps}, $\epsilon=${epsilon}, seed={seed}"
        # )
    
    if outputdir and fig is not None:
        os.makedirs(outputdir, exist_ok=True)
        if sampler_type == "sgld":
            filename = f"posterior_sgld_n{num_training_data}_eps{epsilon}_sigmap{prior_sigma}_chain{num_steps}_rngseed{seed}.png"
        elif sampler_type == "mcmc":
            filename =  f"posterior_mcmc_n{num_training_data}_sigmap{prior_sigma}_chain{num_steps}_rngseed{seed}.png"
        filepath = os.path.join(outputdir, filename)
        fig.tight_layout()
        fig.savefig(filepath)  
        print(f"Figure saved at: {filepath}")
        _run.add_artifact(filepath, name=filename, content_type="image", metadata={"filepath": filepath})
    return
