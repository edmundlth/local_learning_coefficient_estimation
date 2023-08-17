from sgld_2d_validation import ex
from utils import expand_dictionary
from multiprocessing import Pool

# Maximum of parallel processes to run
MAX_NUM_WORKERS = 6

# Change the values in each config dictionary for different experiments. 
# All values of type `list` in the dictionary will be expanded with itertools.product
rngseeds = list(range(0, 10))
ALL_CONFIG_VALUES_SGLD = {
    "sampler_type": "sgld",
    "num_training_data": 1000, 
    "sigma": 0.5, 
    "prior_sigma": 1.0, 
    "epsilon": [1e-4, 5e-4, 1e-3], 
    "num_steps": 1000, 
    "seed": rngseeds, 
}

ALL_CONFIG_VALUES_MCMC = {
    "sampler_type": "mcmc",
    "num_training_data": 1000, 
    "sigma": 0.5, 
    "prior_sigma": 1.0, 
    "num_steps": 1000, 
    "seed": rngseeds,    
}

def _create_config_list(all_config_vals): 
    # assume none of the config values are suppose to be list. 
    to_be_expanded = [key for key, val in all_config_vals.items() if isinstance(val, list)]
    return expand_dictionary(all_config_vals, to_be_expanded)


def exec_fn(config, outputdir=None, no_plot=True):
    config.update({"outputdir": outputdir, "no_plot": no_plot})
    run = ex.run(config_updates=config)
    print(run.config["outputdir"])
    return

if __name__ == "__main__":
    config_list = _create_config_list(ALL_CONFIG_VALUES_SGLD) + _create_config_list(ALL_CONFIG_VALUES_MCMC)
    print(f"Total experiments count: {len(config_list)}")
    with Pool(processes=MAX_NUM_WORKERS) as pool:
        pool.map(exec_fn, config_list)