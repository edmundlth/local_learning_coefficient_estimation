import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import Distribution, constraints
from numpyro.infer import MCMC, NUTS
import pandas as pd
import numpy as np

from bayes_observables import convert_expectation_temp

class BiVariatePowerLaw(Distribution):
    # support = constraints.real  # x, y can be any real number, we'll take absolute inside
    support = constraints.independent(constraints.interval(-5, 5), 1)
    
    def __init__(self, h1, h2, maxval=5):
        self.h1 = h1
        self.h2 = h2
        self.maxval = maxval
        
        # Compute normalization constant
        self.Z = np.prod([2 * maxval ** (h1 + 1) / (h1 + 1), 2 * maxval ** (h2 + 1) / (h2 + 1)])
        super(BiVariatePowerLaw, self).__init__(event_shape=(2,))
    
    def log_prob(self, value):
        x, y = value[..., 0], value[..., 1]
        unnormalized_log_prob = self.h1 * jnp.log(jnp.abs(x)) + self.h2 * jnp.log(jnp.abs(y))
        return unnormalized_log_prob - jnp.log(self.Z)
    

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
    def __init__(self, polynomial, sigma, prior_sigma, key, num_training_data, prior_mean=(0.0, 0.0), prior_exponents=None):
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
        self.prior_exponents = prior_exponents
        if prior_exponents is not None:
            self.prior_dist = BiVariatePowerLaw(prior_exponents[0], prior_exponents[1])
            self.localising_prior = dist.MultivariateNormal(loc=self.prior_mean, covariance_matrix=self.prior_sigma * jnp.eye(2))
            h1, h2 = self.prior_exponents
            self._log_prior = jax.jit(
                lambda w: h1 * jnp.log(jnp.abs(w[0])) + h2 * jnp.log(jnp.abs(w[1])) + self.localising_prior.log_prob(w)
            )
        else:
            self.prior_dist = dist.MultivariateNormal(loc=self.prior_mean, covariance_matrix=self.prior_sigma * jnp.eye(2))
            self._log_prior = jax.jit(lambda w: self.prior_dist.log_prob(w))
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
        logprior = self._log_prior(w)
        # prior = jnp.sum(norm.logpdf(w, loc=self.prior_mean, scale=self.prior_sigma))
        return loglikehood_val * itemp + logprior

    def compute_lambdahat(self, samples, true_w=(0, 0)):
        true_w = jnp.array(true_w)
        sample_energy = -np.mean([self.log_likelihood(w) for w in samples])
        true_energy = -self.log_likelihood(true_w)
        hatlambda = (sample_energy - true_energy) / jnp.log(self.num_training_data)
        return hatlambda
    
    def compute_functional_variance(self, samples):
        loglike_array = self.create_loglike_array(samples)
        func_var = np.sum(np.var(loglike_array, axis=1))
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
        w = numpyro.sample("w", self.prior_dist)
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
            markersize=0.2,
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

