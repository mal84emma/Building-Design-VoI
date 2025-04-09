"""Compute posterior optimal system designs."""

import os
import sys
import warnings
from tqdm import tqdm
import numpy as np
import gurobipy as gp
import multiprocess as mp
from functools import partial
from utils import data_handling, retry_wrapper
from experiments.configs.experiments import parse_experiment_args
from model_wrappers import posterior_design
from prob_models import posterior_model



if __name__ == '__main__':

    # Run params
    scenarios_to_do = 256
    offset = 0

    from experiments.configs.config import *

    # Get run options
    [expt_id,n_buildings,info_id] = [int(sys.argv[i]) for i in range(1,4)]
    expt_no = "".join(str(i) for i in [expt_id,n_buildings,info_id])
    expt_name, sizing_constraints, info_type = parse_experiment_args(expt_id, n_buildings, info_id)

    if len(sys.argv) > 3: scen_no = int(sys.argv[3])
    else: scen_no = None


    np.random.seed(0)
    n_post_samples = 1000 # override config settings to get more samples for scenario reduction
    # need to be careful with this as L1/2 cache size may be exceeded, causing slowdown due to increased misses

    post_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info')
    if not os.path.exists(os.path.join(post_results_dir,'designs')):
        os.makedirs(os.path.join(post_results_dir,'designs'))

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        try:
            m = gp.Model()
            solver_kwargs = {'solver':'GUROBI','Threads':5}
            # restrict solver threads to prevent slowdown due to thread swapping
        except:
            solver_kwargs = {}

        # Load prior scenario samples.
        scenarios_path = os.path.join(results_dir,f'sampled_scenarios_{n_buildings}b.csv')
        scenarios, measurements = data_handling.load_scenarios(scenarios_path)
        n_buildings = scenarios.shape[1]

        # Set up output directory.
        out_dir = os.path.join(post_results_dir,'designs')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Set up wrapper function for posterior design.
        design_wrapper = partial(
            retry_wrapper(posterior_design),
            out_dir=out_dir,
            prob_config=prob_config,
            posterior_model=posterior_model,
            info_type=info_type,
            n_post_samples=n_post_samples,
            data_dir=dataset_dir,
            solar_file_pattern=solar_fname_pattern,
            building_file_pattern=building_fname_pattern,
            cost_dict=cost_dict,
            sizing_constraints=sizing_constraints,
            solver_kwargs=solver_kwargs,
            num_reduced_scenarios=num_reduced_scenarios,
            expt_no=expt_no,
            show_progress=True if scen_no is not None else False,
        )

        # Compute posterior optimal system designs.
        if scen_no is None:
            scenarios_to_design = list(enumerate(measurements))[offset:offset+scenarios_to_do]
            if n_concurrent_designs > 1:
                with mp.Pool(n_concurrent_designs) as pool:
                    design_results = list(tqdm(pool.imap(design_wrapper, scenarios_to_design), total=len(scenarios_to_design)))
            else:
                design_results = [design_wrapper(t) for t in tqdm([(None,s) for s in scenarios_to_design])]
        else:
            design_result = design_wrapper((scen_no, measurements[scen_no]))