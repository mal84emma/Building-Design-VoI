"""Quantify benefit derived from each part of the system for prior case."""

import os
import sys
import numpy as np
import pandas as pd
from utils import data_handling


if __name__ == '__main__':

    n_buildings = int(sys.argv[1])

    from experiments.configs.config import *

    results_dir = os.path.join('experiments','results')

    cases = ['unconstr','constr_solar','battery_only','solar_only','neither']

    # Compare design results for each system case.
    # ============================================
    columns = ['total battery','total solar','grid con. capacity']
    design_results = pd.DataFrame(index=cases,columns=columns)

    for case in cases:
        design_results_path = os.path.join(results_dir,'prior',f'{case}_{n_buildings}b_design_results.csv')
        prior_design_results = data_handling.load_design_results(design_results_path)
        design_results.loc[case] = [
            np.sum(prior_design_results['battery_capacities']),
            np.sum(prior_design_results['solar_capacities']),
            prior_design_results['grid_con_capacity']
        ]
    print('Designs:')
    print(design_results)

    # Compare eval results for each system case.
    # ==========================================
    columns = ['mean total','mean elec','mean carbon','mean grid ex','N ex scens','total std','total min','total max']
    cost_results = pd.DataFrame(index=cases,columns=columns)

    for case in cases:
        eval_results_path = os.path.join(results_dir,'prior',f'{case}_{n_buildings}b_eval_results.csv')
        eval_results = data_handling.load_eval_results(eval_results_path)
        overall_costs = [res['objective'] for res in eval_results]
        cost_results.loc[case] = [
            np.mean(overall_costs),
            np.mean([res['objective_contrs'][0] for res in eval_results]),
            np.mean([res['objective_contrs'][1] for res in eval_results]),
            np.mean([res['objective_contrs'][2] for res in eval_results]),
            np.sum([res['objective_contrs'][2] > 0 for res in eval_results]),
            np.std(overall_costs),
            np.min(overall_costs),
            np.max(overall_costs)
        ]
    print('\nCosts:')
    print(cost_results)

    # Compute relative costs of each system case.
    # ===========================================
    relative_costs = pd.DataFrame(index=cases,columns=columns)
    for case in cases:
        relative_costs.loc[case] = cost_results.loc[case]/cost_results.loc['neither']*100
    print('\nRelative costs:')
    print(relative_costs)