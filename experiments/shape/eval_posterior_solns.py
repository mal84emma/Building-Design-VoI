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
from prob_models import posterior_model
from energy_system import evaluate_multi_system_scenarios


def posterior_evaluation(
        scenario_id_tuple,
        design_results_path_pattern,
        out_dir,
        years,
        n_post_samples,
        info_type,
        data_dir,
        building_file_pattern,
        cost_dict,
        solver_kwargs={},
        n_processes=None,
        show_progress=True
    ):
    """Wrapper function for evaluating system based on posterior samples.

    Args:
        scenario_id_tuple (_type_): Tuple containing scenario number
            (index of samples from prior), and scenario vector for 
            sampled scenario. Single argument to unpack for ease of
            multiprocessing.
        design_results_path_pattern (str): Path pattern for loading
            system design results. Formatted with scenario number.
        out_dir (path): Directory to save design results to.
        years (list): List of valid years for posterior distr/sampling.
        n_post_samples (int): No. of samples to draw from posterior.
        info_type (str): 'type' or 'profile'. Type of information provided by
            sampled scenario, determining posterior dist. to use.
        NOTE: all args below are passed to `evaluate_multi_system_scenarios`
            function. See docstring for details.
        data_dir (path):
        building_file_pattern (str):
        cost_dict (dict):
        solver_kwargs (dict, optional): Defaults to {}.
        num_reduced_scenarios (int, optional): Defaults to None.
        n_processes (int, optional): Defaults to None.
        show_progress (bool, optional): Defaults to False.

    Returns:
        dict: Not used.
    """

    scenario_num,measured_scenario = scenario_id_tuple

    # Load system design.
    system_design = data_handling.load_design_results(design_results_path_pattern.format(j=scenario_num))

    # Sample scenarios from posterior based on info type.
    if info_type == 'profile':
        sampled_scenarios = [measured_scenario]
    elif info_type == 'type':
        sampled_scenarios = posterior_model(measured_scenario[:,0], n_post_samples, years)

    # Evaluate system.
    if show_progress: print(f'\nStarting scenario {scenario_num} evaluation @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')
    start = time.time()
    mean_cost, eval_results = evaluate_multi_system_scenarios(
            sampled_scenarios,
            system_design,
            data_dir,
            building_file_pattern,
            design=True,
            cost_dict=cost_dict,
            solver_kwargs=solver_kwargs,
            n_processes=n_processes if info_type == 'type' else None,
            show_progress=show_progress,
        )
    end = time.time()

    # Save results.
    out_path = os.path.join(out_dir,f's{scenario_num}_posterior_eval_results.csv')
    data_handling.save_eval_results(eval_results, system_design, sampled_scenarios, out_path)

    # Report finish.
    print(f'\nScenario {scenario_num} evaluation completed in {end-start:.2f}s @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')

    return eval_results


if __name__ == '__main__':

    np.random.seed(0)

    info_type = 'type'
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

        # Load posterior optimal system designs.
        designs_path_pattern = os.path.join('experiments','results',f'posterior_solar_constr_{info_type}_info','designs','s{j}_posterior_design_results.csv')
        scenario_tuples = [(j,scenario) for j,scenario in enumerate(scenarios)][:100] ##temp

        # Set up output directory.
        out_dir = os.path.join('experiments','results',f'posterior_solar_constr_{info_type}_info','evals')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Set up wrapper function for posterior design.
        eval_wrapper = partial(
            posterior_evaluation,
            design_results_path_pattern=designs_path_pattern,
            out_dir=out_dir,
            years=years,
            n_post_samples=n_post_samples,
            info_type=info_type,
            data_dir=dataset_dir,
            building_file_pattern=building_fname_pattern,
            cost_dict=cost_dict,
            solver_kwargs={},
            n_processes=n_processes
        )

        # Evaluate posterior optimal system designs.
        eval_results = [eval_wrapper(t) for t in tqdm(scenario_tuples)]