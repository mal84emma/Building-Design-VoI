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
from utils import get_Gurobi_WLS_env, data_handling
from prob_models import shape_posterior_model
from energy_system import design_system


def posterior_design(
        scenario_id_tuple,
        out_dir,
        years,
        n_post_samples,
        info_type,
        data_dir,
        building_file_pattern,
        cost_dict,
        sizing_constraints={},
        solver_kwargs={},
        num_reduced_scenarios=None,
        show_progress=False
    ):
    """Wrapper function for designing system based on posterior samples.

    Args:
        scenario_id_tuple (tuple): Tuple containing scenario number (index of
            samples from prior) and scenario vector for sampled scenario.
            Single argument to unpack for ease of multiprocessing.
        out_dir (path): Directory to save design results to.
        years (list): List of valid years for posterior distr/sampling.
        n_post_samples (int): No. of samples to draw from posterior.
        info_type (str): 'type' or 'profile'. Type of information provided by
            sampled scenario, determining posterior dist. to use.
        NOTE: all args below are passed to `design_system` function. See docstring
            for details.
        data_dir (path):
        building_file_pattern (str):
        cost_dict (dict):
        sizing_constraints (dict, optional): Defaults to {}.
        solver_kwargs (dict, optional): Defaults to {}.
        num_reduced_scenarios (int, optional): Defaults to None.
        show_progress (bool, optional): Defaults to False.

    Returns:
        dict: Not used.
    """

    scenario_num,measured_scenario = scenario_id_tuple

    # Set up solver environment.
    if solver_kwargs['solver'] == 'GUROBI':
        solver_kwargs['env'] = get_Gurobi_WLS_env(silence=not show_progress if scenario_num != None else True)

    # Sample scenarios from posterior based on info type.
    if info_type == 'profile':
        sampled_scenarios = [measured_scenario]
        num_reduced_scenarios = None
    elif info_type == 'type':
        sampled_scenarios = shape_posterior_model(measured_scenario[:,0], n_post_samples, years)

    # Design system.
    start = time.time()
    design_results = design_system(
            sampled_scenarios,
            data_dir,
            building_file_pattern,
            cost_dict,
            sizing_constraints=sizing_constraints,
            solver_kwargs=solver_kwargs,
            num_reduced_scenarios=num_reduced_scenarios,
            show_progress=show_progress if scenario_num != None else False,
            process_id=scenario_num
        )
    end = time.time()

    # Save results.
    out_path = os.path.join(out_dir,f's{scenario_num}_posterior_design_results.csv')
    data_handling.save_design_results(design_results, out_path)

    # Report finish.
    print(f'Scenario {scenario_num} design completed in {end-start:.2f}s @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')

    return design_results



if __name__ == '__main__':

    np.random.seed(0)

    info_type = 'type'
    n_concurrent_designs = 8
    # need to be careful with this as L1/2 cache size may be exceeded, causing slowdown due to increased misses

    sizing_constraints = {'battery':None,'solar':150.0}

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        from experiments.shape.expt_config import *
        n_post_samples = 1000 # adjust for design

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
        out_dir = os.path.join(results_dir,f'posterior_constr_solar_{info_type}_info','designs')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Set up wrapper function for posterior design.
        design_wrapper = partial(
            posterior_design,
            out_dir=out_dir,
            years=years,
            n_post_samples=n_post_samples,
            info_type=info_type,
            data_dir=dataset_dir,
            building_file_pattern=building_fname_pattern,
            cost_dict=cost_dict,
            sizing_constraints=sizing_constraints,
            solver_kwargs=solver_kwargs,
            num_reduced_scenarios=num_reduced_scenarios
        )

        # Compute posterior optimal system designs.
        if n_concurrent_designs > 1:
            with mp.Pool(n_concurrent_designs) as pool:
                design_results = list(tqdm(pool.imap(design_wrapper, list(enumerate(scenarios))), total=len(scenarios)))
        else:
            design_results = [design_wrapper(t) for t in tqdm(zip([None]*len(scenarios),scenarios))]