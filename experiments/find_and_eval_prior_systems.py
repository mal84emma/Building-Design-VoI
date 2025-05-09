"""Quantify benefit derived from each part of the system for prior case."""

import os
import sys
import warnings
import gurobipy as gp
from utils import get_Gurobi_WLS_env, data_handling
from energy_system import design_system, evaluate_multi_system_scenarios

if __name__ == '__main__':

    # Get run options
    n_buildings = int(sys.argv[1])

    if len(sys.argv) > 2: expt_no = int(sys.argv[2])
    else: expt_no = None


    from experiments.configs.config import *

    options_dicts = [
        {
            'case_name': 'unconstr',
            'sizing_constraints': {'battery':None,'solar':None},
            'use_battery': True
        },
        {
            'case_name': 'constr_solar',
            'sizing_constraints': {'battery':None,'solar':solar_constraint},
            'use_battery': True
        },
        {
            'case_name': 'battery_only',
            'sizing_constraints': {'battery':None,'solar':1e-6},
            'use_battery': True
        },
        {
            'case_name': 'solar_only',
            'sizing_constraints': {'battery':1e-6,'solar':None},
            'use_battery': False
        },
        {
            'case_name': 'neither',
            'sizing_constraints': {'battery':1e-6,'solar':1e-6},
            'use_battery': False
        }
    ]


    if not os.path.exists(os.path.join(results_dir,'prior')):
        os.makedirs(os.path.join(results_dir,'prior'), exist_ok=True)

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        # Load prior scenario samples.
        scenarios_path = os.path.join(results_dir,f'sampled_scenarios_{n_buildings}b.csv')
        scenarios,_ = data_handling.load_scenarios(scenarios_path)
        n_buildings = scenarios.shape[1]

        try:
            m = gp.Model()
            e = get_Gurobi_WLS_env()
            solver_kwargs = {'solver':'GUROBI','Method':2,'Threads':4,'env':e}
        except:
            solver_kwargs = {}


        for n,d in enumerate(options_dicts): # options specifiying each partial system

            if (expt_no is None) or (expt_no == n):

                case_name = d['case_name']
                expt_code = "".join(str(i) for i in [n_buildings,n])

                # Design system.
                # ==============
                design_results = design_system(
                    scenarios,
                    dataset_dir,
                    solar_fname_pattern,
                    building_fname_pattern,
                    cost_dict,
                    sizing_constraints=d['sizing_constraints'],
                    expt_no=expt_code,
                    solver_kwargs=solver_kwargs,
                    num_reduced_scenarios=num_reduced_scenarios,
                    show_progress=True
                )

                print(case_name)
                for key in ['objective','objective_contrs','battery_capacities','solar_capacities','grid_con_capacity']:
                    print(design_results[key])

                out_path = os.path.join(results_dir,'prior',f'{case_name}_{n_buildings}b_design_results.csv')
                data_handling.save_design_results(design_results, out_path)

                # Evaluate system.
                # ================
                mean_cost, eval_results = evaluate_multi_system_scenarios(
                    scenarios,
                    design_results,
                    dataset_dir,
                    solar_fname_pattern,
                    building_fname_pattern,
                    design=True,
                    cost_dict=cost_dict,
                    use_battery=d['use_battery'],
                    n_processes=n_processes,
                    expt_no=expt_code,
                    show_progress=True
                )
                print(case_name, mean_cost)

                out_path = os.path.join(results_dir,'prior',f'{case_name}_{n_buildings}b_eval_results.csv')
                data_handling.save_eval_results(eval_results, design_results, scenarios, out_path)