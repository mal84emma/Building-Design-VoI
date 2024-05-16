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
    prior_mean_cost = np.mean([res['objective'] for res in prior_eval_results])
    prior_cost_std = np.std([res['objective'] for res in prior_eval_results])
    print(f'Prior Mean Cost: {prior_mean_cost}')
    print(f'Prior Cost Std: {prior_cost_std}')

    # Load posterior eval results.
    # ============================
    print('')
    for info_type in ['type']: # 'profile'
        posterior_eval_results_dir = os.path.join('experiments','results',f'posterior_{info_type}_info','evals')
        posterior_eval_results_files = [file for file in os.listdir(posterior_eval_results_dir) if file.endswith(".csv")]
        post_eval_results = [data_handling.load_eval_results(os.path.join(posterior_eval_results_dir, file)) for file in posterior_eval_results_files]
        posterior_mean_cost = np.mean([np.mean([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results])
        posterior_mean_cost_std = np.mean([np.std([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results])
        print(f'Posterior Mean Cost ({info_type} info): {posterior_mean_cost}')
        print(f'Posterior Cost Std ({info_type} info): {posterior_mean_cost_std}')

        # Compute Value of Information.
        # =============================
        voi = prior_mean_cost - posterior_mean_cost
        print(f'Value of Information ({info_type} info): {voi}')
        print(f'VOI as % of Prior Mean Cost: {voi/prior_mean_cost*100:.2f}%')