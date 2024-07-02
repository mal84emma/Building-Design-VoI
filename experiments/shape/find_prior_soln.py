"""Compute prior optimal system design."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
# run using `python -m experiments.{fname}`

import warnings
import numpy as np
import gurobipy as gp
from utils import get_Gurobi_WLS_env, data_handling
from energy_system import design_system

if __name__ == '__main__':

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)

        from experiments.shape.expt_config import *

        # Load prior scenario samples.
        scenarios_path = os.path.join(results_dir,'sampled_scenarios.csv')
        scenarios = data_handling.load_scenarios(scenarios_path)
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

        # Save results.
        out_path = os.path.join(results_dir,'prior','prior_design_results.csv')
        data_handling.save_design_results(design_results, out_path)