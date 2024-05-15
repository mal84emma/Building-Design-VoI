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
from energy_system import evaulate_multi_system_scenarios


def posterior_evaluation(
        design_scen_id_tuple,
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
        design_scen_id_tuple (_type_): Tuple containing scenario number
            (index of samples from prior), system design, and scenario vector
            for sampled scenario. Single argument to unpack for ease of
            multiprocessing.
        out_dir (path): Directory to save design results to.
        years (list): List of valid years for posterior distr/sampling.
        n_post_samples (int): No. of samples to draw from posterior.
        info_type (str): 'type' or 'profile'. Type of information provided by
            sampled scenario, determining posterior dist. to use.
        NOTE: all args below are passed to `evaulate_multi_system_scenarios`
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

    scenario_num,system_design,measured_scenario = design_scen_id_tuple

    # Sample scenarios from posterior based on info type.
    if info_type == 'profile':
        sampled_scenarios = [measured_scenario]
    elif info_type == 'type':
        sampled_scenarios = posterior_model(measured_scenario[:,0], n_post_samples, years)

    # Evaluate system.
    if show_progress: print(f'Starting scenario {scenario_num} evaluation @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')
    start = time.time()
    mean_cost, eval_results = evaulate_multi_system_scenarios(
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
    print(f'Scenario {scenario_num} evaluation completed in {end-start:.2f}s @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')

    return eval_results


if __name__ == '__main__':

    np.random.seed(0)

    info_type = 'type'
    n_processes = 6 # mp.cpu_count()

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        from experiments.expt_config import *
        ##temp
        n_post_samples = 24

        # Load prior scenario samples.
        scenarios_path = os.path.join('experiments','results','sampled_scenarios.csv')
        scenarios = data_handling.load_scenarios(scenarios_path)
        n_buildings = scenarios.shape[1]

        ##temp
        scenarios = scenarios[:5]

        # Load posterior optimal system designs.
        design_results_path = os.path.join('experiments','results',f'posterior_{info_type}_info','designs','s{j}_posterior_design_results.csv')
        design_scen_tuples = [(j,data_handling.load_design_results(design_results_path.format(j=j)),scenario) for j,scenario in enumerate(scenarios)]

        # Set up output directory.
        out_dir = os.path.join('experiments','results',f'posterior_{info_type}_info','evals')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Set up wrapper function for posterior design.
        eval_wrapper = partial(
            posterior_evaluation,
            data_dir=dataset_dir,
            building_file_pattern=building_fname_pattern,
            cost_dict=cost_dict,
            out_dir=out_dir,
            years=years,
            n_post_samples=n_post_samples,
            info_type=info_type,
            solver_kwargs={},
            n_processes=n_processes
        )

        # Evaluate posterior optimal system designs.
        eval_results = [eval_wrapper(t) for t in tqdm(design_scen_tuples)]