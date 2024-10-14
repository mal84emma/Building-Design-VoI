"""Plot prior costs and correlations with scenarios."""

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
    plt.show()

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
    plt.show()

    # Plot distributions of total scenario costs.
    # ===========================================
    fig, ax = plt.subplots()
    sns.kdeplot(np.array(prior_costs)/1e6, label='Prior', ax=ax, c='k')
    plt.xlabel('Total scenario cost (£m)')
    plt.show()

    fig, ax = plt.subplots()
    sns.kdeplot(prior_costs/np.array([np.sum(scen[:,2])*365*24*cost_dict['opex_factor'] for scen in scenarios]), label='Prior', ax=ax, c='k')
    plt.xlabel('Scenario LCOE (£/kWh)')
    plt.show()