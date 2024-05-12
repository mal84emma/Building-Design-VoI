"""Evaluation perform of prior optimal solution (system design)."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

from utils import data_handling
from energy_system import evaulate_multi_system_scenarios

if __name__ == '__main__':

    from experiments.expt_config import *

    # Load prior scenario samples.
    scenarios_path = os.path.join('experiments','results','sampled_scenarios.csv')
    scenarios = data_handling.load_scenarios(scenarios_path)
    n_buildings = scenarios.shape[1]

    # Load prior optimal system design.
    design_results_path = os.path.join('experiments','results','prior_design_results.csv')
    system_design = data_handling.load_design_results(design_results_path)

    # Evaluate prior optimal system design.
    n_processes = 3

    mean_cost, eval_results = evaulate_multi_system_scenarios(
        scenarios[:6],
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
    out_path = os.path.join('experiments','results','prior_eval_results.csv')
    data_handling.save_eval_results(eval_results, system_design, scenarios[:6], out_path)