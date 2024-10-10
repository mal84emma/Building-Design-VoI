"""Compute Value of Information from test results."""

import os
import sys
from utils import data_handling
from experiments.configs.experiments import parse_experiment_args

import numpy as np


if __name__ == '__main__':

    from experiments.configs.config import *

    # Get run options
    [expt_id,n_buildings,info_id] = [int(sys.argv[i]) for i in range(1,4)]
    expt_name, sizing_constraints, info_type = parse_experiment_args(expt_id, n_buildings, info_id)


    # Load posterior eval results.
    # ============================
    print(f'\nInfo type: {info_type}')
    posterior_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info')
    posterior_design_results_files = [file for file in os.listdir(os.path.join(posterior_results_dir,'designs')) if file.endswith(".csv")]
    scenario_numbers = [int(file.split('_')[0][1:]) for file in posterior_design_results_files]
    posterior_design_results_files = [dfile for scen_num,dfile in sorted(zip(scenario_numbers, posterior_design_results_files))]

    posterior_eval_results_files = [file for file in os.listdir(os.path.join(posterior_results_dir,'evals')) if file.endswith(".csv")]
    scenario_numbers = [int(file.split('_')[0][1:]) for file in posterior_eval_results_files]
    posterior_eval_results_files = [efile for scen_num,efile in sorted(zip(scenario_numbers, posterior_eval_results_files))]

    n_post_samples = len(posterior_design_results_files)

    post_design_results = [data_handling.load_design_results(os.path.join(posterior_results_dir, 'designs', file)) for file in posterior_design_results_files]
    posterior_mean_grid_cap = np.mean([res['grid_con_capacity'] for res in post_design_results])
    post_eval_results = [data_handling.load_eval_results(os.path.join(posterior_results_dir, 'evals', file)) for file in posterior_eval_results_files]
    post_mean_costs = [np.mean([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results]
    posterior_mean_cost = np.mean(post_mean_costs)
    posterior_mean_cost_std = np.mean([np.std([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results])
    posterior_overall_cost_std = np.std([res['objective'] for scenario_results in post_eval_results for res in scenario_results])
    posterior_mean_cost_range = np.mean([np.max([res['objective'] for res in scenario_results]) - np.min([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results])
    posterior_mean_max_cost = np.mean([np.max([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results])
    posterior_mean_min_cost = np.mean([np.min([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results])
    print(f'Posterior Mean Cost: {posterior_mean_cost}')
    print(f'Posterior Mean Cost Std Error: {np.std(post_mean_costs)/np.sqrt(len(post_mean_costs))}')
    print(f'Posterior Cost Mean Std: {posterior_mean_cost_std}')
    print(f'Posterior Overall Cost Std: {posterior_overall_cost_std}')
    print('Mean min-max: ', np.min(post_mean_costs), np.max(post_mean_costs))
    print('Overall min-max: ', *[f([res['objective'] for scenario_results in post_eval_results for res in scenario_results]) for f in [np.min, np.max]])
    print(f'Posterior Mean Cost Range: {posterior_mean_cost_range}')
    print(f'Posterior Mean Min and Max Cost: {posterior_mean_min_cost}, {posterior_mean_max_cost}')
    print(f'Posterior Mean Grid Cap: {posterior_mean_grid_cap}')

    print('')

    # Load prior eval results.
    # ========================
    print('Prior:')
    prior_design_results = data_handling.load_design_results(os.path.join(results_dir,'prior',f'{expt_name}_{n_buildings}b_design_results.csv'))
    prior_grid_cap = prior_design_results['grid_con_capacity']
    prior_eval_results = data_handling.load_eval_results(os.path.join(results_dir,'prior',f'{expt_name}_{n_buildings}b_eval_results.csv'))
    prior_costs = [res['objective'] for res in prior_eval_results][:n_post_samples]
    # only use prior cost samples corresponding to posterior samples for fair comparison
    # keeps VoI > 0 as prior and posterior are compared on same ground-truth scenarios
    prior_mean_cost = np.mean(prior_costs)
    prior_cost_std = np.std(prior_costs)
    print(f'Prior Mean Cost: {prior_mean_cost}')
    print(f'Prior Cost Std: {prior_cost_std}')
    print(f'Prior Mean Std Error: {prior_cost_std/np.sqrt(len(prior_costs))}')
    print('Overall min-max: ', np.min(prior_costs), np.max(prior_costs))
    print(f'Prior Cost Range: {np.max(prior_costs) - np.min(prior_costs)}')
    print(f'Prior grid cap: {prior_grid_cap}')

    # Compute Value of Information.
    # =============================
    print('')
    voi = prior_mean_cost - posterior_mean_cost
    print(f'Value of Information ({info_type} info): {voi}')
    print(f'VOI as % of Prior Mean Cost: {voi/prior_mean_cost*100:.3f}%')