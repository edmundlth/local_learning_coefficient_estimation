import jax
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

import os
import json
from concurrent.futures import ProcessPoolExecutor
import functools

from sgld_2d_validation import (
    SGLDExperiment,
    parse_commandline, 
    expand_dictionary, 
)



def run_experiment(expt_config):
    # Initialize parameters for the experiment
    K1, K2 = expt_config["exponents"]
    sampler_type = expt_config["sampler"]
    sigma = expt_config["sigma"]
    prior_sigma = expt_config["prior_sigma"]
    num_training_data = expt_config["num_training_data"]
    itemp = 1 / jnp.log(num_training_data)
    num_steps = expt_config["num_steps"]
    epsilon = expt_config["epsilon"]
    key = expt_config["rngkey"]
    
    polynomial = jax.jit(lambda w: ((w[0] - 1)**K1) * ((w[0]**2 + w[1]**2)**K2))
    
    expt_results = []
    
    for w_init_tuple in [(0.0, 0.0), (1.0, 0.3)]:
        w_init = jnp.array(w_init_tuple)
        # Expt set up
        key, _ = random.split(key)
        experiment = SGLDExperiment(
            polynomial, sigma, prior_sigma, key, num_training_data, prior_mean=w_init
        )
        # For SGLD
        if sampler_type == "sgld":
            trajectory = experiment.run_sgld(w_init=w_init, epsilon=epsilon, num_steps=num_steps, itemp=itemp)
        elif sampler_type == "mcmc":
            trajectory = experiment.run_mcmc(num_samples=num_steps, itemp=itemp)
        lambdahat = experiment.compute_lambdahat(trajectory, true_w=(0, 0))
        
        print(f"w_init:{w_init_tuple}, rngkey:{key.tolist()}, itemp:{itemp:.4f}, lambdahat:{lambdahat:.4f}")
        result = {
            "num_training_data": num_training_data, 
            "sigma": sigma, 
            "w_init": w_init_tuple, 
            "prior_sigma": prior_sigma, 
            "sampler": sampler_type, 
            "epsilon": epsilon if sampler_type == "sgld" else None, 
            "chain_length": num_steps, 
            "rngkey": key.tolist(),
            "lambdahat": float(lambdahat),
        }
        expt_results.append(result)

    return expt_results


if __name__ == "__main__":
    parser = parse_commandline()
    parser.add_argument("--num_repeat", help="number of repeated experiments using successive random.split of the rng key.", type=int, default=1)
    parser.add_argument("--exponents", type=int, nargs="*", default=[1, 2])
    args = parser.parse_args()
    K1, K2 = args.exponents

    print(f"Commandline arguments:\n{vars(args)}")
    if args.outputdir:
        os.makedirs(args.outputdir, exist_ok=True)
    
    keys = random.split(random.PRNGKey(args.seeds[0]), num=args.num_repeat)
    
    config_dict = {
            "exponents": args.exponents,
            "num_training_data": args.num_training_data, 
            "sigma": args.sigma, 
            "prior_sigma": args.prior_sigma, 
            "epsilon": args.epsilon[0], 
            "num_steps": args.num_steps[0], 
            "rngseed": args.seeds[0], 
            "rngkey": keys,
            "sampler": "sgld"
    }
    expt_configs = expand_dictionary(config_dict, ["rngkey"])
    print(f"Number of experiments: {len(expt_configs)}.")

    if args.max_workers is not None and args.max_workers > 1: 
        print("Running parallel processes")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            outputs = list(executor.map(run_experiment, expt_configs))
            expt_results = []
            for result in outputs:
                expt_results += result
    else:
        expt_results = []
        for config in expt_configs:
            expt_results += run_experiment(config)
        
    if args.outputdir:
        datetime_string = datetime.now().strftime('%Y%m%d%H%M')
        filepath = os.path.join(args.outputdir, f"results_{datetime_string}.json")
        with open(filepath, 'w') as outfile:
            json.dump(expt_results, outfile, indent=2)
        
        df = pd.DataFrame(expt_results)
        df['w_init'] = df['w_init'].apply(lambda x: tuple(x))
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.histplot(data=df, x="lambdahat", hue="w_init", bins=20, ax=ax)
        ymax, ymin = ax.get_ylim()
        ax.vlines([1/(2 * K1), 2 / (4 * K2)], ymin=ymin, ymax=ymax, alpha=0.5, linestyles="dashed", color="black")

        sigma = config_dict["sigma"]
        prior_sigma = config_dict["prior_sigma"]
        num_training_data = config_dict["num_training_data"]
        num_steps = config_dict["num_steps"]
        epsilon = config_dict["epsilon"]
        rngseed = config_dict["rngseed"]

        potential_str = f"$(w_1 - 1)^{2 * K1} (w_1^2 + w_2^2)^{2 * K2}$"
        fig.suptitle(
            f"{potential_str}, "
            f"$n={num_training_data}$, "
            f"$\sigma={sigma}$, "
            f"$\sigma_p={prior_sigma}$, "
            f"$\epsilon={epsilon}$, "
            f"chain_len={num_steps}, "
            f"rngseed={rngseed}"
        )

        filepath = os.path.join(args.outputdir, f"hist_{datetime_string}.png")
        fig.savefig(f"")
        fig.savefig(filepath)
        print(f"histogram filepath: {filepath}")
        
