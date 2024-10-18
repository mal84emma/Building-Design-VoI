"""Plot posterior costs and relation to prior costs."""

import os
import sys
from utils import data_handling
from experiments.configs.experiments import parse_experiment_args

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    from experiments.configs.config import *

    # Get run options
    [expt_id,n_buildings,info_id] = [int(sys.argv[i]) for i in range(1,4)]
    expt_name, sizing_constraints, info_type = parse_experiment_args(expt_id, n_buildings, info_id)

    # Load scenarios & prior results.
    # ===============================
    scenarios_path = os.path.join(results_dir,f'sampled_scenarios_{n_buildings}b.csv')
    scenarios,_ = data_handling.load_scenarios(scenarios_path)

    prior_design_results = data_handling.load_design_results(os.path.join(results_dir,'prior',f'{expt_name}_{n_buildings}b_design_results.csv'))
    prior_grid_cap = prior_design_results['grid_con_capacity']
    prior_eval_results = data_handling.load_eval_results(os.path.join(results_dir,'prior',f'{expt_name}_{n_buildings}b_eval_results.csv'))
    prior_costs = [res['objective'] for res in prior_eval_results]
    prior_lcoes = prior_costs/np.array([np.sum(scen[:,2])*365*24*cost_dict['opex_factor'] for scen in scenarios])

    posterior_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info')
    posterior_eval_results_files = [file for file in os.listdir(os.path.join(posterior_results_dir,'evals')) if file.endswith(".csv")]
    scenario_numbers = [int(file.split('_')[0][1:]) for file in posterior_eval_results_files]
    posterior_eval_results_files = [efile for scen_num,efile in sorted(zip(scenario_numbers, posterior_eval_results_files))]
    post_eval_results = [data_handling.load_eval_results(os.path.join(posterior_results_dir, 'evals', file)) for file in posterior_eval_results_files]
    posterior_costs = [res['objective'] for scenario_results in post_eval_results for res in scenario_results]
    posterior_lcoes = [res['objective']/np.sum([bs[2]*365*24*cost_dict['opex_factor'] for bs in res['scenario']]) for scenario_results in post_eval_results for res in scenario_results]

    # Plot distributions of total scenario costs.
    # ===========================================
    fig, ax = plt.subplots()
    sns.kdeplot(np.array(prior_costs)/1e6, label='Prior', ax=ax, c='k', alpha=0.5, lw=2)
    sns.kdeplot(np.array(posterior_costs)/1e6, label='Posterior', ax=ax, c='k', lw=2)
    ymax = ax.get_ylim()[1]
    plt.vlines(np.mean(prior_costs)/1e6, 0, ymax, colors='k', alpha=0.5, linestyles='dashed', lw=1.5)
    plt.vlines(np.mean(posterior_costs)/1e6, 0, ymax, colors='k', linestyles='dashed', lw=1.5)
    plt.annotate('', xy=(np.mean(prior_costs)/1e6, ymax*0.4), xytext=(np.mean(prior_costs)/1e6 + 1.25, ymax*0.4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, headlength=8))
    plt.annotate('', xy=(np.mean(posterior_costs)/1e6, ymax*0.4), xytext=(np.mean(posterior_costs)/1e6 - 1.25, ymax*0.4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, headlength=8))
    plt.text(np.mean([*prior_costs,*posterior_costs])/1e6, ymax*0.445, 'VoI', va='center', ha='center', fontsize=12, fontfamily='serif',
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='square,pad=0.125'))
    plt.xlabel('Total scenario cost (£m)')
    plt.ylim(0, ymax)
    ax.get_yaxis().set_ticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots',f'{expt_name}_{n_buildings}b_posterior_total_costs.pdf'))
    #plt.show()

    fig, ax = plt.subplots()
    sns.kdeplot(prior_lcoes, label='Prior', ax=ax, c='k', alpha=0.5, lw=2)
    sns.kdeplot(posterior_lcoes, label='Posterior', ax=ax, c='k', lw=2)
    ymax = ax.get_ylim()[1]
    plt.vlines(np.mean(prior_lcoes), 0, ymax, colors='k', alpha=0.5, linestyles='dashed', lw=2)
    plt.text(np.mean(prior_lcoes)*1.005, ymax*0.3, f'Mean: £{np.mean(prior_lcoes):.3f}/kWh', alpha=0.5, rotation=90, va='center', ha='left')
    plt.vlines(np.mean(posterior_lcoes), 0, ymax, colors='k', linestyles='dashed', lw=2)
    plt.text(np.mean(posterior_lcoes)*0.995, ymax*0.3, f'Mean: £{np.mean(posterior_lcoes):.3f}/kWh', rotation=90, va='center', ha='right')
    plt.xlabel('Scenario LCOE (£/kWh)')
    plt.ylim(0, ymax)
    ax.get_yaxis().set_ticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots',f'{expt_name}_{n_buildings}b_posterior_lcoes.pdf'))
    #plt.show()