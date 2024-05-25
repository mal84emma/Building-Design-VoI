"""Evaluate performance of prior optimal solution (system design)."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import warnings
from utils import data_handling
import multiprocessing as mp
from energy_system import evaluate_multi_system_scenarios

if __name__ == '__main__':

    n_processes = mp.cpu_count()

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        from experiments.expt_config import *

        # Load prior scenario samples.
        scenarios_path = os.path.join('experiments','results','sampled_scenarios.csv')
        scenarios = data_handling.load_scenarios(scenarios_path)
        n_buildings = scenarios.shape[1]

        # Load prior optimal system design.
        design_results_path = os.path.join('experiments','results','prior','prior_design_results.csv')
        system_design = data_handling.load_design_results(design_results_path)

        # Evaluate prior optimal system design.
        mean_cost, eval_results = evaluate_multi_system_scenarios(
            scenarios,
            system_design,
            dataset_dir,
            building_fname_pattern,
            design=True,
            cost_dict=cost_dict,
            n_processes=n_processes,
            show_progress=True
        )
        print(mean_cost)

        # Save results.
        out_path = os.path.join('experiments','results','prior','prior_eval_results.csv')
        data_handling.save_eval_results(eval_results, system_design, scenarios, out_path)