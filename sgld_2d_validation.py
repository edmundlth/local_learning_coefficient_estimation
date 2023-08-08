import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import os
import json
import argparse
import functools
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np

from bayes_observables import convert_expectation_temp, compute_functional_variance, compute_waic
from utils import expand_dictionary

# Stochastic Gradient Langevin Dynamics
def SGLD_step(key, grad_log_posterior_fn, w, epsilon):
    eta = random.normal(key, shape=w.shape) * jnp.sqrt(epsilon)
    return w + epsilon * grad_log_posterior_fn(w) / 2 + eta


# Function to run SGLD
def run_SGLD(grad_log_posterior_fn, w_init, epsilon, num_steps, key):
    # grad_log_posterior_fn = lambda w: grad_log_posterior(w, x, y, sigma, prior_sigma)
    w = w_init
    trajectory = [w]
    for _ in range(num_steps):
        key, subkey = random.split(key)
        w = SGLD_step(subkey, grad_log_posterior_fn, w, epsilon)
        trajectory.append(w)
    return jnp.array(trajectory)


class SGLDExperiment:
    def __init__(self, polynomial, sigma, prior_sigma, key, num_training_data, prior_mean=(0.0, 0.0)):
        self.polynomial = polynomial
        self.sigma = sigma
        self.prior_mean = jnp.array(prior_mean)
        self.prior_sigma = prior_sigma
        self.key = key
        self.num_training_data = num_training_data

        # Generate some data
        self.key, subkey = random.split(self.key)
        self.x = random.normal(self.key, shape=(self.num_training_data,))
        self.y = 0 * self.x + self.sigma * random.normal(
            subkey, shape=(self.num_training_data,)
        )

        self.log_likelihood = jax.jit(lambda w: jnp.sum(self._log_likelihood(w)))

        self.trajectories = []
    
    def forward(self, w, x):
        return self.polynomial(w) * x
        # return w[0] * self.x + w[1]

    # Function to compute the log-likelihood of the observations given the parameters
    def _log_likelihood(self, w):
        mu = self.forward(w, self.x)
        return norm.logpdf(self.y, loc=mu, scale=self.sigma)

    # Function to compute the log-posterior
    def _log_posterior(self, w, itemp=1.0):
        loglikehood_val = jnp.sum(self._log_likelihood(w))
        prior = jnp.sum(norm.logpdf(w, loc=self.prior_mean, scale=self.prior_sigma))
        return loglikehood_val * itemp + prior

    def compute_lambdahat(self, samples, true_w=(0, 0)):
        true_w = jnp.array(true_w)
        sample_energy = -np.mean([self.log_likelihood(w) for w in samples])
        true_energy = -self.log_likelihood(true_w)
        hatlambda = (sample_energy - true_energy) / jnp.log(self.num_training_data)
        return hatlambda
    
    def compute_functional_variance(self, samples):
        loglike_array = self.create_loglike_array(samples)
        func_var = np.mean(np.var(loglike_array, axis=1))
        return func_var

    def run_sgld(self, w_init, epsilon, num_steps, itemp=1.0, store_trajectory=False):
        self.key, subkey = random.split(self.key)
        log_posterior = jax.jit(lambda w: self._log_posterior(w, itemp))
        grad_log_posterior_fn = jax.jit(jax.grad(log_posterior))
        trajectory = run_SGLD(grad_log_posterior_fn, w_init, epsilon, num_steps, subkey)

        if store_trajectory:
            self.trajectories.append(
                {
                    "itemp": itemp,
                    "samples": trajectory,
                    "chain_length": num_steps,
                    "w_init": w_init,
                    "epsilon": epsilon,
                    "type": "sgld",
                }
            )
        return trajectory

    def model(self, itemp=1.0):
        # Numpyro Bayesian model based on the polynomial
        w = numpyro.sample("w", dist.MultivariateNormal(loc=self.prior_mean, covariance_matrix=self.prior_sigma * jnp.eye(2)))
        # mu = self.polynomial(w) * self.x
        mu = self.forward(w, self.x)
        with numpyro.plate("N", self.num_training_data):
            numpyro.sample(
                "y", dist.Normal(mu, self.sigma / jnp.sqrt(itemp)), obs=self.y
            )

    def run_mcmc(
        self,
        num_chains=1,
        num_samples=1000,
        thinning=1,
        num_warmup=10,
        itemp=1.0,
        store_samples=False,
    ):
        kernel = NUTS(self.model)
        mcmc = MCMC(
            kernel,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            thinning=thinning,
            progress_bar=False
        )
        mcmc.run(self.key, itemp=itemp)
        samples = mcmc.get_samples()["w"]

        if store_samples:
            self.trajectories.append(
                {
                    "itemp": itemp,
                    "samples": samples,
                    "num_chains": num_chains,
                    "num_warmup": num_warmup,
                    "thinning": thinning,
                    "chain_length": len(samples),
                    "type": "mcmc_nuts",
                }
            )
        return samples
    
    def plot(self, samples, ax, true_w=(0, 0)):
        lambdahat = self.compute_lambdahat(samples, true_w=true_w)
        val = np.max(np.abs(samples))
        val = np.max([1.0, val])
        w1_vals = jnp.linspace(-val, val, 100)
        w2_vals = jnp.linspace(-val, val, 100)
        W1, W2 = jnp.meshgrid(w1_vals, w2_vals)
        log_posterior = jax.jit(lambda w: self._log_posterior(w, itemp=1.0))
        Z = jnp.array(
                [[log_posterior(jnp.array([w1, w2])) for w1 in w1_vals] for w2 in w2_vals]
            )
        Z = Z / self.num_training_data
        ax.contour(W1, W2, Z, levels=100, cmap="magma")
        ax.plot(
            samples[:, 0],
            samples[:, 1],
            "kx",
            linewidth=0.1,
            markersize=0.1,
            alpha=0.8,
        )
        # ax.scatter(trajectory[0, 0], trajectory[0, 1], c='red')
        # ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='blue')
        return lambdahat

    def create_loglike_array(self, samples):
        loglike_array = jnp.vstack(
            [self._log_likelihood(w) for w in samples]
        ).T
        return loglike_array

    def compute_multitemp_lambdahat(self, samples):
        ensemble_energy1 = -np.mean([self.log_likelihood(w) for w in samples])

        loglike_array = self.create_loglike_array(samples)
        energy = jnp.sum(loglike_array, axis=0)
        itemp1 = 1 / jnp.log(self.num_training_data)
        itemp2 = 1.2 * itemp1
        ensemble_energy2 = convert_expectation_temp(energy, energy, itemp1, itemp2)

        hatlambda = (ensemble_energy1 - ensemble_energy2) / (1/itemp1 - 1/itemp2)
        return hatlambda


def process_json_file(file_path):
    # Read the JSON file
    data = pd.read_json(file_path)
    # Calculate lambda values
    data['lambda'] = 1 / (2 * data['monomial'].apply(lambda x: max(x)))
    # Convert monomial index to LaTeX format
    data["monomial"] = [f"$w_1^{k1}w_2^{k2}$" for k1, k2 in data["monomial"]]
    # Concatenate sampler and chain_length for later use
    data['sampler_chain'] = data['sampler'] + ' (' + data['chain_length'].astype(str) + ')'
    return data

def generate_latex_table(df):
    grouped = df.groupby('monomial')
    sampler_chains = df['sampler_chain'].unique()

    latex_table = "\\begin{tabular}{l|" + " c" * (len(sampler_chains) + 1) + "}\n\\toprule\n"
    latex_table += "monomial & " + " & ".join(sampler_chains) + " & $\\lambda$ \\\\ \hline\n"

    for monomial, group in grouped:
        mean_std = group.groupby('sampler_chain')['lambdahat'].agg([np.mean, np.std])
        row_values = [
            f"{mean_std.loc[sc, 'mean']:.4f} ({mean_std.loc[sc, 'std']:.4f})" 
            if sc in mean_std.index else '' 
            for sc in sampler_chains
        ]
        lambda_value = f"{group['lambda'].iloc[0]:.4f}"

        latex_table += monomial + " & " + " & ".join(row_values) + " & " + lambda_value + " \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}"

    return latex_table

def run_experiment(expt_config, no_plot=True):
    # Initialize parameters for the experiment
    w_init = jnp.array([0.0, 0.0])
    sampler_type = expt_config["sampler"]
    sigma = expt_config["sigma"]
    prior_sigma = expt_config["prior_sigma"]
    num_training_data = expt_config["num_training_data"]
    itemp = 1 / jnp.log(num_training_data)
    num_steps = expt_config["num_steps"]
    epsilon = expt_config["epsilon"]
    rngseed = expt_config["rngseed"]
    key = random.PRNGKey(rngseed)

    m = 4
    monomial_exponents = [(0, i) for i in range(1, m)] + [(i, j) for i in range(1, m) for j in range(i, m)]
    # monomial_exponents = [(1, 1)]
    num_plots = len(monomial_exponents)
    nrow = 3
    ncol = num_plots // nrow + (num_plots % nrow != 0)
    
    expt_results = []
    if not no_plot:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
        axes = np.ravel(axes)
    else:
        fig, axes = None, None
    for c, (k1, k2) in enumerate(monomial_exponents):
        # Expt set up
        polynomial = jax.jit(lambda w: (w[0] ** k1) * (w[1] ** k2))
        symhatlambda = "$\hat{\lambda}$"
        sympotential = f"$w_1^{k1}w_2^{k2}$"
        true_lambda = 1 / (2 * max(k1, k2))
        key, subkey = random.split(key)
        experiment = SGLDExperiment(
            polynomial, sigma, prior_sigma, key, num_training_data
        )

        # For SGLD
        if sampler_type == "sgld":
            samples = experiment.run_sgld(w_init=w_init, epsilon=epsilon, num_steps=num_steps, itemp=1.0)
            tempered_samples = experiment.run_sgld(w_init=w_init, epsilon=epsilon, num_steps=num_steps, itemp=itemp)
        elif sampler_type == "mcmc":
            samples = experiment.run_mcmc(num_samples=num_steps, itemp=1.0)
            tempered_samples = experiment.run_mcmc(num_samples=num_steps, itemp=itemp)
        lambdahat = experiment.compute_lambdahat(tempered_samples, true_w=(0, 0))
        func_var = experiment.compute_functional_variance(samples)
        sing_fluc_est = func_var * num_training_data / 2
        # lambdahat = experiment.compute_multitemp_lambdahat(trajectory)
        print(f"{sampler_type}, {(k1, k2)}, itemp:{itemp:.4f}, lambdahat:{lambdahat:.4f}, lambda:{true_lambda:.4f}, nu:{sing_fluc_est}")
        result = {
            "num_training_data": num_training_data, 
            "sigma": sigma, 
            "prior_sigma": prior_sigma, 
            "monomial": (k1, k2),
            "sampler": sampler_type, 
            "epsilon": epsilon if sampler_type == "sgld" else None, 
            "chain_length": num_steps, 
            "rngseed": rngseed, 
            "lambdahat": float(lambdahat),
            "func_var": float(func_var), 
            "nu": float(sing_fluc_est)
        }
        expt_results.append(result)

        if not no_plot:
            ax = axes[c]
            experiment.plot(tempered_samples, ax)
            title_string = f"{sympotential}, {symhatlambda}={lambdahat:.4f}, $\lambda$={true_lambda:.4f}"
            ax.set_title(title_string, fontsize=8)
    
    if not no_plot:
        s = "$w_1^{k_1}w_2^{k_2}$"
        fig.suptitle(
            f"{s}, n={num_training_data}, $\sigma=${sigma}, $\sigma_p=${prior_sigma}, chain_len={num_steps}, $\epsilon=${epsilon}, seed={rngseed}"
        )
    return expt_results, fig

def run_and_save(config, outputdir=None, no_plot=True):
    print(f"Config: {config}")
    results, fig = run_experiment(config, no_plot=no_plot)
    if outputdir and fig is not None:
        sampler = config["sampler"]
        num_steps = config["num_steps"]
        rngseed = config["rngseed"]
        if sampler == "sgld":
            epsilon = config["epsilon"]
            filename = f"posterior_sgld_eps{epsilon}_chain{num_steps}_rngseed{rngseed}.png"
        elif sampler == "mcmc":
            filename =  f"posterior_mcmc_chain{num_steps}_rngseed{rngseed}.png"
        fig.savefig(os.path.join(outputdir, filename), bbox_inches="tight")
    return results


def parse_commandline():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--num_training_data",
        help="Training data set size.",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--prior_sigma",
        help="Standard deviation of the localising gaussian prior.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sigma",
        help="Standard deviation of the observation noise in the model.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--epsilon",
        help="Epsilon parameter for SGLD",
        type=float,
        nargs="+", 
        default=[1e-4, 5e-4, 1e-3],
    )
    parser.add_argument(
        "--num_steps",
        help="Number of SGLD steps",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000],
    )
    parser.add_argument(
        "--seeds",
        help="Experiments are repeated for each RNG seed given",
        nargs="+",
        type=int,
        default=[0, 1],
    )
    parser.add_argument(
        "--outputdir", 
        help="A directory to store outputs.", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--no_plot", 
        help="Don't produce any plots for the experiments",
        action="store_true"
    )
    parser.add_argument(
        "--max_workers",
        help="Maximum number of parallel process running the experiments independently for each given configuration",
        type=int,
        default=None,
    )
    return parser

if __name__ == "__main__":
    args = parse_commandline().parse_args()
    print(f"Commandline arguments:\n{vars(args)}")
    if args.outputdir:
        os.makedirs(args.outputdir, exist_ok=True)
    
    config_dict = {
            "num_training_data": args.num_training_data, 
            "sigma": args.sigma, 
            "prior_sigma": args.prior_sigma, 
            "epsilon": args.epsilon, 
            "num_steps": args.num_steps, 
            "rngseed": args.seeds, 
    }
    config_dict.update({"sampler": "sgld"})
    expt_configs = expand_dictionary(config_dict, ["epsilon", "num_steps", "rngseed"])
    config_dict.update({"sampler": "mcmc"})
    expt_configs += expand_dictionary(config_dict, ["num_steps", "rngseed"])
    print(f"Number of experiments: {len(expt_configs)}.")


    if args.max_workers is not None and args.max_workers > 1: 
        print("Running parallel processes")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            exec_fn = functools.partial(run_and_save, outputdir=args.outputdir, no_plot=args.no_plot)
            outputs = list(executor.map(exec_fn, expt_configs))
            expt_results = []
            for result in outputs:
                expt_results += result
    else:
        expt_results = []
        for config in expt_configs:
            expt_results += run_and_save(config, outputdir=args.outputdir, no_plot=args.no_plot)
        
    if args.outputdir:
        filepath = os.path.join(args.outputdir, "results.json")
        with open(filepath, 'w') as outfile:
            json.dump(expt_results, outfile, indent=2)
