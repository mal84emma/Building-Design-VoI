"""Compute posterior optimal system designs."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
# run using `python -m experiments.{fname}`

import time
import warnings
from tqdm import tqdm
import numpy as np
import gurobipy as gp
import multiprocess as mp
from functools import partial
from utils import data_handling, posterior_design
from prob_models import shape_posterior_model


if __name__ == '__main__':

    # Get run option
    run_option = int(sys.argv[1])

    scenarios_to_do = 160
    offset = 0

    if run_option == 0:
        expt_name = 'unconstr'
        sizing_constraints = {'battery':None,'solar':None}
    elif run_option == 1:
        expt_name = 'solar_constr'
        sizing_constraints = {'battery':None,'solar':150.0}
    else:
        raise ValueError('Invalid run option. Please provide valid CLI argument.')


    np.random.seed(0)

    info_type = 'type'
    n_concurrent_designs = 8
    # need to be careful with this as L1/2 cache size may be exceeded, causing slowdown due to increased misses

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        from experiments.expt_config import *
        from experiments.shape.shape_expts_config import *
        n_post_samples = 1000 # adjust for design

        post_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{info_type}_info')

        try:
            m = gp.Model()
            solver_kwargs = {'solver':'GUROBI','Threads':4}
            # restrict solver threads to prevent slowdown due to thread swapping
        except:
            solver_kwargs = {}

        # Load prior scenario samples.
        scenarios_path = os.path.join(results_dir,'sampled_scenarios.csv')
        scenarios = data_handling.load_scenarios(scenarios_path)
        n_buildings = scenarios.shape[1]

        # Set up output directory.
        out_dir = os.path.join(post_results_dir,'designs')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Set up wrapper function for posterior design.
        design_wrapper = partial(
            posterior_design,
            out_dir=out_dir,
            years=years,
            posterior_model=shape_posterior_model,
            info_type=info_type,
            n_post_samples=n_post_samples,
            data_dir=dataset_dir,
            building_file_pattern=building_fname_pattern,
            cost_dict=cost_dict,
            sizing_constraints=sizing_constraints,
            solver_kwargs=solver_kwargs,
            num_reduced_scenarios=num_reduced_scenarios
        )

        # Compute posterior optimal system designs.
        scenarios_to_design = list(enumerate(scenarios))[offset:offset+scenarios_to_do]
        if n_concurrent_designs > 1:
            with mp.Pool(n_concurrent_designs) as pool:
                design_results = list(tqdm(pool.imap(design_wrapper, scenarios_to_design), total=len(scenarios_to_design)))
        else:
            design_results = [design_wrapper(t) for t in tqdm([(None,s) for s in scenarios_to_design])]