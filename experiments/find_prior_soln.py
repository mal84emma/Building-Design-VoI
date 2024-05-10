"""Compute prior optimal system design."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import numpy as np
import gurobipy as gp
from utils import load_scenarios, get_Gurobi_WLS_env
from energy_system import design_system

if __name__ == '__main__':

    from experiments.expt_config import *

    # Load prior scenario samples.
    scenarios_path = os.path.join('experiments','results','sampled_scenarios.csv')
    scenarios = load_scenarios(scenarios_path)
    n_buildings = scenarios.shape[1]

    # Compute prior optimal system design.
    try:
        m = gp.Model()
        e = get_Gurobi_WLS_env()
        solver_kwargs = {'solver': 'GUROBI', 'env': e}
    except:
        solver_kwargs = {}

    design_results = design_system(
        scenarios,
        dataset_dir,
        building_fname_pattern,
        cost_dict,
        solver_kwargs=solver_kwargs,
        num_reduced_scenarios=num_reduced_scenarios,
        show_progress=True
    )

    for key in ['objective','objective_contrs','battery_capacities','solar_capacities','grid_con_capacity']:
        print(design_results[key])

    system_design = {
        'battery_capacities': design_results['battery_capacities'].flatten(),
        'solar_capacities': design_results['solar_capacities'].flatten(),
        'grid_con_capacity': design_results['grid_con_capacity']
    }

    # Save results.
    ... # implement fn to save results from LP design??