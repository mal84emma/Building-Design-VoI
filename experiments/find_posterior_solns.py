"""Compute posterior optimal system designs."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import warnings
from tqdm import tqdm
import gurobipy as gp
import multiprocess as mp
from functools import partial
from utils import get_Gurobi_WLS_env, data_handling
from prob_models import posterior_model
from energy_system import design_system


def posterior_design(
        scenario_id_tuple,
        data_dir,
        building_file_pattern,
        cost_dict,
        out_dir,
        years,
        n_post_samples,
        info_type,
        solver_kwargs={},
        num_reduced_scenarios=None,
        show_progress=False
    ):
    """ToDo"""

    scenario_num,measured_scenario = scenario_id_tuple

    # Set up solver environment.
    if solver_kwargs['solver'] == 'GUROBI':
        solver_kwargs['env'] = get_Gurobi_WLS_env(silence=not show_progress if scenario_num != None else True)

    # Sample scenarios from posterior based on info type.
    if info_type == 'profile':
        sampled_scenarios = [measured_scenario]
        num_reduced_scenarios = None
    elif info_type == 'type':
        sampled_scenarios = posterior_model(measured_scenario[:,0], n_post_samples, years)

    # Design system.
    design_results = design_system(
            sampled_scenarios,
            data_dir,
            building_file_pattern,
            cost_dict,
            solver_kwargs=solver_kwargs,
            num_reduced_scenarios=num_reduced_scenarios,
            show_progress=show_progress if scenario_num != None else False,
            process_id=scenario_num
        )

    # Save results.
    out_path = os.path.join(out_dir,f's{scenario_num}_posterior_design_results.csv')
    data_handling.save_design_results(design_results, out_path)

    return design_results



if __name__ == '__main__':

    info_type = 'type'
    n_post_samples = 1000
    n_concurrent_designs = 2

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)

        from experiments.expt_config import *

        try:
            m = gp.Model()
            solver_kwargs = {'solver': 'GUROBI'}
        except:
            solver_kwargs = {}

        # Load prior scenario samples.
        scenarios_path = os.path.join('experiments','results','sampled_scenarios.csv')
        scenarios = data_handling.load_scenarios(scenarios_path)
        n_buildings = scenarios.shape[1]

        # Set up output directory.
        out_dir = os.path.join('experiments','results',f'posterior_{info_type}_info','designs')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Set up wrapper function for posterior design.
        design_wrapper = partial(
            posterior_design,
            data_dir=dataset_dir,
            building_file_pattern=building_fname_pattern,
            cost_dict=cost_dict,
            out_dir=out_dir,
            years=years,
            n_post_samples=n_post_samples,
            info_type=info_type,
            solver_kwargs=solver_kwargs,
            num_reduced_scenarios=num_reduced_scenarios
        )

        # Compute posterior optimal system designs.
        if n_concurrent_designs > 1:
            with mp.Pool(n_concurrent_designs) as pool:
                design_results = list(tqdm(pool.imap(design_wrapper, list(enumerate(scenarios))), total=len(scenarios)))
        else:
            design_results = [design_wrapper(t) for t in tqdm(zip([None]*len(scenarios),scenarios))]