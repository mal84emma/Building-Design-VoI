"""Compute Value of Information from test results."""

import os
import sys
from utils import data_handling
from experiments.configs.experiments import parse_experiment_args

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    plt.style.use('./resources/plots.mplstyle')

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
    post_cost_stds = [np.std([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results]
    post_cost_ranges = [np.ptp([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results]
    print(np.mean(post_mean_costs),np.mean(post_cost_stds),np.mean(post_cost_ranges))

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
    prior_cost_range = np.ptp(prior_costs)
    print(f'Prior Cost Std: {prior_cost_std}')
    print('Overall min-max: ', np.min(prior_costs), np.max(prior_costs))
    print('Overall range: ', np.ptp(prior_costs))

    # pick out operating costs
    prior_op_costs = [np.sum([res['objective_contrs'][i] for i in range(3)]) for res in prior_eval_results][:n_post_samples]
    print('operating costs;')
    print(np.mean(prior_op_costs), np.std(prior_op_costs))
    print(np.min(prior_op_costs), np.max(prior_op_costs))
    print(np.ptp(prior_op_costs))

    # Plotting
    # ========
    fig, ax = plt.subplots()
    sns.kdeplot(post_cost_stds/prior_cost_std, label='Std dev', ax=ax, c='k', lw=2, cut=0)
    sns.kdeplot(post_cost_ranges/prior_cost_range, label='Range', ax=ax, c='k', alpha=0.5, lw=2, cut=0)
    ymax = ax.get_ylim()[1]
    plt.scatter(post_cost_stds/prior_cost_std, np.ones(n_post_samples)*1.08*ymax, marker='|', c='k', alpha=0.25, lw=0.5)
    plt.scatter(post_cost_ranges/prior_cost_range, np.ones(n_post_samples)*1.02*ymax, marker='|', c='k', alpha=0.075, lw=0.5)
    xmin = ax.get_xlim()[0]
    plt.text(xmin-0.0075, 1.05*ymax, 'Samples', rotation=90, va='center', ha='right', fontsize=10)
    plt.xlabel('Proportion of prior value')
    #plt.xlim(0.1,0.6)
    plt.ylim(0, ymax*1.2)
    ax.get_yaxis().set_ticks([])
    plt.legend(title='Metric', loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join('plots','posterior_cost_variability.pdf'))
    plt.show()