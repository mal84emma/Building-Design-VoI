"""Plot prior costs and correlations with scenarios."""

import os
import sys
from utils import data_handling
from experiments.configs.experiments import parse_experiment_args

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    plt.style.use('./resources/plots.mplstyle')

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

    # Analyse prior costs.
    # ====================
    max_grid_ex_cost = np.max([res['objective_contrs'][2] for res in prior_eval_results])
    max_grid_ex_power = max_grid_ex_cost/(cost_dict['grid_excess']*365*cost_dict['opex_factor'])
    max_grid_ex_frac = max_grid_ex_power/prior_grid_cap*100
    n_grid_ex_scenarios = np.sum([res['objective_contrs'][2] > 0 for res in prior_eval_results])

    print(np.mean(prior_costs),np.std(prior_costs),np.std(prior_costs)/np.mean(prior_costs))
    print(np.mean(prior_lcoes),np.std(prior_lcoes),np.std(prior_lcoes)/np.mean(prior_lcoes))
    print('')
    print(f'Max grid exceedance cost (£): {max_grid_ex_cost:.0f}')
    print(f'Max grid exceedance power (kW): {max_grid_ex_power:.0f} ({max_grid_ex_frac:.1f}%)')
    print(f'No. grid exceedance scenarios: {n_grid_ex_scenarios}/{len(prior_costs)}')

    # Plot correlations between scenario cost and mean & peak load.
    # =============================================================
    fig, ax = plt.subplots()
    total_mean_loads = [np.sum(scen[:,2]) for scen in scenarios]
    plt.scatter(
        x=total_mean_loads,
        y=np.array(prior_costs)/1e6,
        marker='o',
        alpha=0.5,
        lw=0,
        zorder=0
    )
    plt.xlabel('Total mean load (kW)')
    plt.ylabel('Total scenario cost (£m)')
    #plt.show()

    fig, ax = plt.subplots()
    total_peak_loads = [np.sum(scen[:,3]) for scen in scenarios]
    plt.scatter(
        x=total_peak_loads,
        y=np.array(prior_costs)/1e6,
        marker='o',
        alpha=0.5,
        lw=0,
        zorder=0
    )
    plt.xlabel('Total peak load (kW)')
    plt.ylabel('Total scenario cost (£m)')
    #plt.show()

    # Plot distributions of total scenario costs.
    # ===========================================
    fig, ax = plt.subplots()
    sns.kdeplot(np.array(prior_costs)/1e6, label='Prior', ax=ax, c='k', lw=2)
    ymax = ax.get_ylim()[1]
    plt.vlines(np.mean(prior_costs)/1e6, 0, ymax, colors='k', linestyles='dashed', lw=2)
    plt.text(np.mean(prior_costs)/1e6*0.995, ymax*0.4, f'Mean cost: £{np.mean(prior_costs)/1e6:.1f}m', rotation=90, verticalalignment='center', horizontalalignment='right')
    plt.xlabel('Total scenario cost (£m)')
    plt.ylim(0, ymax)
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.savefig(os.path.join('plots','prior_total_costs.pdf'))
    #plt.show()

    fig, ax = plt.subplots()
    sns.kdeplot(prior_lcoes, label='Prior', ax=ax, c='k', lw=2)
    ymax = ax.get_ylim()[1]
    plt.vlines(np.mean(prior_lcoes), 0, ymax, colors='k', linestyles='dashed', lw=2)
    plt.text(np.mean(prior_lcoes)*0.9975, ymax*0.4, f'Mean LCOE: £{np.mean(prior_lcoes):.3f}/kWh', rotation=90, verticalalignment='center', horizontalalignment='right')
    plt.xlabel('Scenario LCOE (£/kWh)')
    plt.ylim(0, ymax)
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.savefig(os.path.join('plots','prior_lcoes.pdf'))
    #plt.show()