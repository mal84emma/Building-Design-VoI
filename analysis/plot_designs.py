"""Plot prior and posterior designs."""

import os
import sys
from utils import data_handling
from experiments.configs.experiments import parse_experiment_args

import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colormaps as cmaps


if __name__ == '__main__':

    from experiments.configs.config import *

    # Get run options
    [expt_id,n_buildings,info_id] = [int(sys.argv[i]) for i in range(1,4)]
    expt_name, sizing_constraints, info_type = parse_experiment_args(expt_id, n_buildings, info_id)

    # Load posterior & prior designs.
    # ===============================
    prior_design_results = data_handling.load_design_results(os.path.join(results_dir,'prior',f'{expt_name}_{n_buildings}b_design_results.csv'))

    posterior_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info')
    posterior_design_results_files = [file for file in os.listdir(os.path.join(posterior_results_dir,'designs')) if file.endswith(".csv")]
    scenario_numbers = [int(file.split('_')[0][1:]) for file in posterior_design_results_files]
    posterior_design_results_files = [dfile for scen_num,dfile in sorted(zip(scenario_numbers, posterior_design_results_files))]
    post_design_results = [data_handling.load_design_results(os.path.join(posterior_results_dir, 'designs', file)) for file in posterior_design_results_files]

    # Plot designs.
    # =============
    fig, ax = plt.subplots()

    post_grid_caps = [res['grid_con_capacity'] for res in post_design_results]
    post_solar_caps = [np.sum(res['solar_capacities']) for res in post_design_results]
    post_battery_caps = [np.sum(res['battery_capacities']) for res in post_design_results]
    norm = plt.Normalize(min(post_grid_caps), max(post_grid_caps))
    cmap = cmaps['viridis']

    # compute linear regression for storage vs solar capacity
    reg = linregress(post_solar_caps,post_battery_caps)
    print('storage vs solar', reg.slope, reg.intercept)

    # posterior
    plt.scatter(
        x=post_solar_caps,
        y=post_battery_caps,
        c=post_grid_caps,
        norm=norm,
        cmap=cmap,
        label='Posterior',
        marker='o',
        alpha=0.5,
        lw=0,
        zorder=0
    )
    # linear regression line
    reg_xs = np.linspace(min(post_solar_caps), max(post_solar_caps), 100)
    plt.plot(reg_xs, reg.slope*reg_xs + reg.intercept, 'k-', alpha=0.5, label='Regression')
    # prior
    plt.scatter(
        x=np.sum(prior_design_results['solar_capacities']),
        y=np.sum(prior_design_results['battery_capacities']),
        c=prior_design_results['grid_con_capacity'],
        norm=norm,
        cmap=cmap,
        label='Prior',
        marker='D',
        edgecolors='k',
        zorder=1,
        lw=0.5
    )

    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm), ax=ax, label='Grid Connection Capacity (kW)')

    plt.xlabel('Total Solar Capacity (kWp)')
    plt.ylabel('Total Battery Capacity (kWh)')

    plt.show()


    # Plot correlations between designs and scenarios.
    # ================================================
    fig, ax = plt.subplots()

    prior_av_load = np.sum(np.mean(prior_design_results['reduced_scenarios'][:,:,2],axis=0))
    post_av_loads = [np.sum(np.mean(res['reduced_scenarios'][:,:,2],axis=0)) for res in post_design_results]

    for label,var in zip(['Solar capacity (kWp)','Battery capacity (kWh)','Grid con. capacity (kW)'],['solar_capacities','battery_capacities','grid_con_capacity']):
        xs = np.array(post_av_loads)
        ys = np.array([np.sum(res[var]) for res in post_design_results])
        reg = linregress(post_av_loads,ys)
        print(var, reg.slope, reg.intercept)
        plt.plot(xs, reg.slope*xs + reg.intercept,'k-', alpha=0.5, label='_nolegend_',zorder=0)
        plt.scatter(
            x=xs,
            y=ys,
            label=label,
            marker='o',
            alpha=0.5,
            lw=0,
            zorder=1
        )

    plt.xlabel('Scenario Mean Total Load (kW)')
    plt.ylabel('Optimised district capacity for posterior scenario')

    plt.legend()
    plt.show()