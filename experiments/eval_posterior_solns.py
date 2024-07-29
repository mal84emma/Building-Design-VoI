"""Evaluate performance of posterior optimal solutions (system designs)."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import warnings
from tqdm import tqdm
import numpy as np
import multiprocess as mp
from functools import partial
from utils import data_handling
from model_wrappers import posterior_evaluation
from prob_models import shape_posterior_model, level_posterior_model


if __name__ == '__main__':

    # Run params
    scenarios_to_do = 160
    offset = 0

    # Get run options
    expt_type = str(sys.argv[1])
    expt_id = int(sys.argv[2])
    info_id = int(sys.argv[3])

    from experiments.configs.general_config import *
    if expt_type == 'shape':
        from experiments.configs.shape_expts_config import *
        posterior_model = shape_posterior_model
    elif expt_type == 'level':
        from experiments.configs.level_expts_config import *
        posterior_model = level_posterior_model
    else:
        raise ValueError('Invalid run option for `expt_type`. Please provide valid CLI argument.')

    if expt_id == 0:
        expt_name = 'unconstr'
        sizing_constraints = {'battery':None,'solar':None}
    elif expt_id == 1:
        expt_name = 'solar_constr'
        sizing_constraints = {'battery':None,'solar':150.0}
    else:
        raise ValueError('Invalid run option for `expt_id`. Please provide valid CLI argument.')

    if info_id == 0:
        info_type = 'type'
    elif info_id == 1:
        info_type = 'mean'
    elif info_id == 2:
        info_type = 'peak'
    elif info_id == 3:
        info_type = 'mean+peak'
    else:
        raise ValueError('Invalid run option for `info_id`. Please provide valid CLI argument.')


    np.random.seed(0)
    n_processes = mp.cpu_count()

    post_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{info_type}_info')
    if not os.path.exists(os.path.join(post_results_dir,'evals')):
        os.makedirs(os.path.join(post_results_dir,'evals'))

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        # Load prior scenario samples.
        scenarios_path = os.path.join(results_dir,'sampled_scenarios.csv')
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
            posterior_evaluation,
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