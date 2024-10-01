"""Evaluate performance of posterior optimal solutions (system designs)."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import time
import warnings
from tqdm import tqdm
import numpy as np
import multiprocess as mp
from functools import partial
from utils import data_handling
from experiments.configs.experiments import parse_experiment_args
from model_wrappers import posterior_evaluation
from prob_models import posterior_model


def retry_wrapper(*args, **kwargs):
    """Retry wrapper for posterior design.
    Scenario reduction sometimes fails to initialize, so retry
    5 times to allow it to succeed."""

    for _ in range(5):
        try:
            eval_result = posterior_evaluation(*args, **kwargs)
            break
        except Exception as e:
            print(f'Error: {e}')
            time.sleep(1)

    return eval_result


if __name__ == '__main__':

    # Run params
    scenarios_to_do = 256
    offset = 0

    from experiments.configs.config import *

    # Get run options
    [expt_id,n_buildings,info_id] = [int(sys.argv[i]) for i in range(1,4)]
    expt_name, sizing_constraints, info_type = parse_experiment_args(expt_id, n_buildings, info_id)


    np.random.seed(0)
    n_processes = mp.cpu_count()

    post_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info')
    if not os.path.exists(os.path.join(post_results_dir,'evals')):
        os.makedirs(os.path.join(post_results_dir,'evals'))

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
            os.makedirs(out_dir)

        # Set up wrapper function for posterior design.
        eval_wrapper = partial(
            retry_wrapper, #posterior_evaluation,
            design_results_path_pattern=designs_path_pattern,
            out_dir=out_dir,
            prob_config=prob_config,
            posterior_model=posterior_model,
            info_type=info_type,
            n_post_samples=n_post_samples,
            data_dir=dataset_dir,
            building_file_pattern=building_fname_pattern,
            cost_dict=cost_dict,
            solver_kwargs={},
            n_processes=n_processes
        )

        # Evaluate posterior optimal system designs.
        eval_results = [eval_wrapper(t) for t in tqdm(scenario_tuples)]