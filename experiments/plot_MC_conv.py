"""Visualise convergence of MC estimate of prior solution mean cost."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import numpy as np
from utils import data_handling
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load prior eval results.
    # ========================
    prior_eval_results_path = os.path.join('experiments','results','prior','prior_eval_results.csv')
    prior_eval_results = data_handling.load_eval_results(prior_eval_results_path)
    prior_costs = [res['objective'] for res in prior_eval_results]
    prior_MC_estimates = [np.mean(prior_costs[:i+1]) for i in range(len(prior_costs))]

    # Load posterior eval results.
    # ============================
    post_MC_estimates = {}
    for info_type in ['type']: # 'profile'
        posterior_eval_results_dir = os.path.join('experiments','results',f'posterior_{info_type}_info','evals')
        posterior_eval_results_files = [file for file in os.listdir(posterior_eval_results_dir) if file.endswith(".csv")]
        post_eval_results = [data_handling.load_eval_results(os.path.join(posterior_eval_results_dir, file)) for file in posterior_eval_results_files]
        post_mean_costs = [np.mean([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results]
        post_MC_estimates[info_type] = [np.mean(post_mean_costs[:i+1]) for i in range(len(post_mean_costs))]

    # Plot convergence of MC estimate of mean cost.
    lss = ['--',':']
    fig = plt.figure()
    plt.plot(range(1,len(prior_MC_estimates)+1), np.array(prior_MC_estimates)/1e6, 'k-', label='Prior')
    for i,info_type in enumerate(['type']): # 'profile'
        plt.plot(range(1,len(post_MC_estimates[info_type])+1), np.array(post_MC_estimates[info_type])/1e6, c='k', ls=lss[i], label=f'Posterior ({info_type} info)')
    plt.xlabel('Number of scenarios')
    plt.ylabel('Mean cost ($m)')
    plt.xlim(0,320)#len(prior_MC_estimates)+1)
    plt.legend()
    plt.show()