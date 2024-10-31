"""Visualise convergence of MC estimate of prior solution mean cost."""

import os
import sys
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
    posterior_eval_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info','evals')
    posterior_eval_results_files = sorted([file for file in os.listdir(posterior_eval_results_dir) if file.endswith(".csv")])
    scenario_numbers = [int(file.split('_')[0][1:]) for file in posterior_eval_results_files]
    posterior_eval_results_files = [efile for scen_num,efile in sorted(zip(scenario_numbers, posterior_eval_results_files))]
    n_post_samples = len(posterior_eval_results_files)
    post_eval_results = [data_handling.load_eval_results(os.path.join(posterior_eval_results_dir, file)) for file in posterior_eval_results_files]
    post_mean_costs = [np.mean([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results]
    post_MC_estimates = np.array([np.mean(post_mean_costs[:i+1]) for i in range(len(post_mean_costs))])
    post_MC_SEs = [np.std(post_mean_costs[:i+1])/np.sqrt(i+1) for i in range(len(post_mean_costs))]

    # Load prior eval results.
    # ========================
    prior_eval_results_path = os.path.join(results_dir,'prior',f'{expt_name}_{n_buildings}b_eval_results.csv')
    prior_eval_results = data_handling.load_eval_results(prior_eval_results_path)
    prior_costs = [res['objective'] for res in prior_eval_results][:n_post_samples]
    prior_MC_estimates = np.array([np.mean(prior_costs[:i+1]) for i in range(len(prior_costs))])
    prior_MC_SEs = [np.std(prior_costs[:i+1])/np.sqrt(i+1) for i in range(len(prior_costs))]

    # Compute VoI estimates.
    # ======================
    vois = prior_MC_estimates - post_MC_estimates
    voi_SEs = [np.std(vois[:i+1])/np.sqrt(i+1) for i in range(len(vois))]

    # Report standard errors of MC estimates.
    # =======================================
    prior_MC_SE = np.std(prior_costs)/np.sqrt(n_post_samples)
    post_MC_SE = np.std(post_mean_costs)/np.sqrt(n_post_samples)
    print(f'Prior MC SE: {prior_MC_SE}')
    print(f'Posterior MC SE: {post_MC_SE}')
    print(f'VoI SE: {voi_SEs[-1]}')

    # Plot convergence of MC estimate of mean cost.
    # =============================================
    xs = np.arange(1,n_post_samples+1)

    fig, ax = plt.subplots()
    ax2 = plt.twinx()
    ax2.plot(xs, vois/1e6, c='hotpink', alpha=0.8, label='Difference (VoI)', zorder=0)
    ax2.fill_between(xs,
                     vois/1e6 - 1.96*np.array(voi_SEs)/1e6,
                     vois/1e6 + 1.96*np.array(voi_SEs)/1e6,
                     color='hotpink', alpha=0.25, lw=0)
    ax2.set_ylabel('VoI estimate (£m)')
    ax2.set_ylim(0)
    ax.set_zorder(ax2.get_zorder() + 1) # put ax in front of ax2
    ax.patch.set_visible(False) # hide the 'canvas'

    ax.plot(xs, prior_MC_estimates/1e6, c='#135F89', label='Prior')
    ax.fill_between(xs,
                     prior_MC_estimates/1e6 - 1.96*np.array(prior_MC_SEs)/1e6,
                     prior_MC_estimates/1e6 + 1.96*np.array(prior_MC_SEs)/1e6,
                     color='#135F89', alpha=0.25, lw=0)
    ax.hlines(prior_MC_estimates[-1]/1e6, 0, n_post_samples, colors='#135F89', linestyles='-', alpha=0.5)
    ax.plot(xs, np.array(post_MC_estimates)/1e6, c='k', label=f'Posterior')
    ax.fill_between(xs,
                     post_MC_estimates/1e6 - 1.96*np.array(post_MC_SEs)/1e6,
                     post_MC_estimates/1e6 + 1.96*np.array(post_MC_SEs)/1e6,
                     color='k', alpha=0.15, lw=0)
    ax.hlines(post_MC_estimates[-1]/1e6, 0, n_post_samples, colors='k', alpha=0.5)

    ax.set_ylim(None,25)
    ax.set_xlabel('Number of scenarios')
    ax.set_ylabel('Mean cost (£m)')
    plt.xlim(0,n_post_samples)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.tight_layout()
    plt.savefig(os.path.join('plots','MC_conv.pdf'))
    plt.show()