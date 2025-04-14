"""Evaluate performance of posterior optimal solutions (system designs)."""

import os
import sys
import warnings
from tqdm import tqdm
import numpy as np
from functools import partial
from utils import data_handling, retry_wrapper
from experiments.configs.experiments import parse_experiment_args
from model_wrappers import posterior_evaluation
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

    if len(sys.argv) > 4: scen_no = int(sys.argv[4])
    else: scen_no = None


    np.random.seed(0)

    post_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info')
    if not os.path.exists(os.path.join(post_results_dir,'evals')):
        os.makedirs(os.path.join(post_results_dir,'evals'), exist_ok=True)

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        # Load prior scenario samples.
        scenarios_path = os.path.join(results_dir,f'sampled_scenarios_{n_buildings}b.csv')
        scenarios, measurements = data_handling.load_scenarios(scenarios_path)
        n_buildings = scenarios.shape[1]

        # Load posterior optimal system designs.
        designs_path_pattern = os.path.join(post_results_dir,'designs','s{j}_posterior_design_results.csv')
        scenario_tuples = list(enumerate(measurements))[offset:offset+scenarios_to_do]

        # Set up output directory.
        out_dir = os.path.join(post_results_dir,'evals')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Set up wrapper function for posterior design.
        eval_wrapper = partial(
            retry_wrapper(posterior_evaluation),
            design_results_path_pattern=designs_path_pattern,
            out_dir=out_dir,
            prob_config=prob_config,
            posterior_model=posterior_model,
            info_type=info_type,
            n_post_samples=n_post_samples,
            data_dir=dataset_dir,
            solar_file_pattern=solar_fname_pattern,
            building_file_pattern=building_fname_pattern,
            cost_dict=cost_dict,
            solver_kwargs={},
            n_processes=n_processes,
            expt_no=expt_no
        )

        # Evaluate posterior optimal system designs.
        if scen_no is None:
            eval_results = [eval_wrapper(t) for t in tqdm(scenario_tuples)]
        else:
            eval_result = eval_wrapper(scenario_tuples[scen_no])