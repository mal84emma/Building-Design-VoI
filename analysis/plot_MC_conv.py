"""Visualise convergence of MC estimate of prior solution mean cost."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

from utils import data_handling
from experiments.configs.experiments import parse_experiment_args

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    from experiments.configs.config import *

    # Get run options
    [expt_id,n_buildings,info_id] = [int(sys.argv[i]) for i in range(1,4)]
    expt_name, sizing_constraints, info_type = parse_experiment_args(expt_id, n_buildings, info_id)

    # Load posterior eval results.
    # ============================
    post_MC_estimates = {}
    posterior_eval_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info','evals')
    posterior_eval_results_files = sorted([file for file in os.listdir(posterior_eval_results_dir) if file.endswith(".csv")])
    scenario_numbers = [int(file.split('_')[0][1:]) for file in posterior_eval_results_files]
    posterior_eval_results_files = [efile for scen_num,efile in sorted(zip(scenario_numbers, posterior_eval_results_files))]
    n_post_samples = len(posterior_eval_results_files)
    post_eval_results = [data_handling.load_eval_results(os.path.join(posterior_eval_results_dir, file)) for file in posterior_eval_results_files]
    post_mean_costs = [np.mean([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results]
    post_MC_estimates[info_type] = [np.mean(post_mean_costs[:i+1]) for i in range(len(post_mean_costs))]

    # Load prior eval results.
    # ========================
    prior_eval_results_path = os.path.join(results_dir,'prior',f'{expt_name}_{n_buildings}b_eval_results.csv')
    prior_eval_results = data_handling.load_eval_results(prior_eval_results_path)
    prior_costs = [res['objective'] for res in prior_eval_results][:n_post_samples]
    prior_MC_estimates = [np.mean(prior_costs[:i+1]) for i in range(len(prior_costs))]

    # Plot convergence of MC estimate of mean cost.
    fig = plt.figure()
    plt.plot(range(1,len(prior_MC_estimates)+1), np.array(prior_MC_estimates)/1e6, 'k-', label='Prior')
    plt.hlines(prior_MC_estimates[-1]/1e6, 0, len(post_MC_estimates[info_type]), colors='k', linestyles='-', alpha=0.5)
    plt.plot(range(1,len(post_MC_estimates[info_type])+1), np.array(post_MC_estimates[info_type])/1e6, c='k', ls='--', label=f'Posterior ({info_type} info)')
    plt.hlines(post_MC_estimates[info_type][-1]/1e6, 0, len(post_MC_estimates[info_type]), colors='k', linestyles='--', alpha=0.5)
    plt.xlabel('Number of scenarios')
    plt.ylabel('Mean cost ($m)')
    plt.xlim(0,n_post_samples)
    plt.legend()
    plt.show()