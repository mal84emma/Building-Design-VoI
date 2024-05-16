"""Compute Value of Information from test results."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import numpy as np
from utils import data_handling


if __name__ == '__main__':

    # Load prior eval results.
    # ========================
    prior_eval_results_path = os.path.join('experiments','results','prior','prior_eval_results.csv')
    prior_eval_results = data_handling.load_eval_results(prior_eval_results_path)
    prior_costs = [res['objective'] for res in prior_eval_results]
    prior_mean_cost = np.mean(prior_costs)
    prior_cost_std = np.std(prior_costs)
    print(f'Prior Mean Cost: {prior_mean_cost}')
    print(f'Prior Cost Std: {prior_cost_std}')
    print('Overall min-max: ', np.min(prior_costs), np.max(prior_costs))

    # Load posterior eval results.
    # ============================
    print('')
    for info_type in ['type']: # 'profile'
        posterior_eval_results_dir = os.path.join('experiments','results',f'posterior_{info_type}_info','evals')
        posterior_eval_results_files = [file for file in os.listdir(posterior_eval_results_dir) if file.endswith(".csv")]
        post_eval_results = [data_handling.load_eval_results(os.path.join(posterior_eval_results_dir, file)) for file in posterior_eval_results_files]
        post_mean_costs = [np.mean([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results]
        posterior_mean_cost = np.mean(post_mean_costs)
        posterior_mean_cost_std = np.mean([np.std([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results])
        print(f'Posterior Mean Cost ({info_type} info): {posterior_mean_cost}')
        print(f'Posterior Cost Std ({info_type} info): {posterior_mean_cost_std}')
        print('Mean min-max: ', np.min(post_mean_costs), np.max(post_mean_costs))
        print('Overall min-max: ', *[f([res['objective'] for scenario_results in post_eval_results for res in scenario_results]) for f in [np.min, np.max]])

        # Compute Value of Information.
        # =============================
        voi = prior_mean_cost - posterior_mean_cost
        print(f'Value of Information ({info_type} info): {voi}')
        print(f'VOI as % of Prior Mean Cost: {voi/prior_mean_cost*100:.3f}%')