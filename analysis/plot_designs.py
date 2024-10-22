"""Plot prior and posterior designs."""

import os
import sys
from utils import data_handling
from experiments.configs.experiments import parse_experiment_args

import numpy as np
from scipy.stats import linregress
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colormaps as cmaps


if __name__ == '__main__':

    plt.style.use('./resources/plots.mplstyle')

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

    # prior
    ax.scatter(
        x=np.sum(prior_design_results['solar_capacities']),
        y=np.sum(prior_design_results['battery_capacities']),
        c=prior_design_results['grid_con_capacity'],
        norm=norm,
        cmap=cmap,
        label='Prior',
        marker='D',
        edgecolors='k',
        zorder=10,
        lw=0.5
    )
        # linear regression line
    reg_xs = np.linspace(min(post_solar_caps), max(post_solar_caps), 100)
    ax.plot(reg_xs, reg.slope*reg_xs + reg.intercept, 'k-', alpha=0.5, label='__nolegend__', zorder=0)
    # posterior
    ax.scatter(
        x=post_solar_caps,
        y=post_battery_caps,
        c=post_grid_caps,
        norm=norm,
        cmap=cmap,
        label='Posterior',
        marker='o',
        alpha=0.5,
        lw=0,
        zorder=5
    )

    # spacing variables
    vmin_reduce = 0.95
    dist_squash = 5

    # guide lines
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    plt.vlines(np.sum(prior_design_results['solar_capacities']), ymin*vmin_reduce, np.sum(prior_design_results['battery_capacities']),
               colors='k', alpha=0.5, linestyles='dashed', lw=1.5, zorder=0)
    plt.hlines(np.sum(prior_design_results['battery_capacities']), xmin*vmin_reduce, np.sum(prior_design_results['solar_capacities']),
               colors='k', alpha=0.5, linestyles='dashed', lw=1.5, zorder=0)

    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm), ax=ax, pad=0.02)
    cbar.set_label('Grid Connection Capacity (kW)')
    cbar.ax.tick_params(labelsize=8)

    # axis distribution plots
    print('solar', np.std(post_solar_caps), np.std(post_solar_caps)/np.mean(post_solar_caps))
    print('battery', np.std(post_battery_caps), np.std(post_battery_caps)/np.mean(post_battery_caps))
    print('grid', np.std(post_grid_caps), np.std(post_grid_caps)/np.mean(post_grid_caps))

    ax2 = ax.twinx()
    ax2.yaxis.set_visible(False)
    sns.kdeplot(post_solar_caps, ax=ax2, fill=True, color='k', alpha=0.2, lw=0, zorder=0, label='__nolegend__')
    ax2.set_ylim(ax2.get_ylim()[0], dist_squash*ax2.get_ylim()[1])

    ax3 = ax.twiny()
    ax3.xaxis.set_visible(False)
    sns.kdeplot(y=post_battery_caps, ax=ax3, fill=True, color='k', alpha=0.2, lw=0, zorder=0, label='__nolegend__')
    ax3.set_xlim(ax3.get_xlim()[0], dist_squash*ax3.get_xlim()[1])

    # plot formatting
    ax.set_xlim(xmin*vmin_reduce,xmax)
    ax.set_ylim(ymin*vmin_reduce,ymax)
    ax.set_xlabel('Total Solar Capacity (kWp)')
    ax.set_ylabel('Total Battery Capacity (kWh)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('plots',f'{expt_name}_{n_buildings}b_posterior_designs.pdf'))
    #plt.show()


    # Plot correlations between designs and scenarios.
    # ================================================
    fig, ax = plt.subplots()

    prior_av_load = np.sum(np.mean(prior_design_results['reduced_scenarios'][:,:,2],axis=0))
    post_av_loads = [np.sum(np.mean(res['reduced_scenarios'][:,:,2],axis=0)) for res in post_design_results]

    for i,(label,var) in enumerate(zip(['Battery capacity (kWh)','Solar capacity (kWp)','Grid con. capacity (kW)']
                                       ,['battery_capacities','solar_capacities','grid_con_capacity'])):
        xs = np.array(post_av_loads)
        ys = np.array([np.sum(res[var]) for res in post_design_results])
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        reg = linregress(post_av_loads,ys)
        print(var, reg.slope, reg.intercept)
        plt.plot(xs, reg.slope*xs + reg.intercept,'k-', alpha=0.5, label='_nolegend_',zorder=0)
        plt.scatter(
            x=xs,
            y=ys,
            color=c,
            label=label,
            marker='o',
            alpha=0.5,
            lw=0,
            zorder=1
        )
        sns.kdeplot(x=xs, y=ys, ax=ax, fill=True, color=c, alpha=0.25, zorder=0,
                    levels=5, thresh=0.05, label='__nolegend__')

    plt.xlabel('Scenario District Mean Load (kW)')
    plt.ylabel('Optimised district capacity for posterior scenario')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots',f'{expt_name}_{n_buildings}b_posterior_designs_correlation.pdf'))
    #plt.show()


    # plot normalised version to investigate spread about regression line
    fig, ax = plt.subplots()

    for i,(label,var) in enumerate(zip(['Battery capacity (kWh)','Solar capacity (kWp)','Grid con. capacity (kW)']
                                       ,['battery_capacities','solar_capacities','grid_con_capacity'])):
        xs = np.array(post_av_loads)
        ys = np.array([np.sum(res[var]) for res in post_design_results])
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        reg = linregress(post_av_loads,ys)
        plt.plot(xs, np.ones(len(xs)),'k-', alpha=0.5, label='_nolegend_',zorder=0)
        plt.scatter(
            x=xs,
            y=ys/(reg.slope*xs + reg.intercept),
            color=c,
            label=label,
            marker='o',
            alpha=0.5,
            lw=0,
            zorder=1
        )
        sns.kdeplot(x=xs, y=ys/(reg.slope*xs + reg.intercept), ax=ax, fill=True, color=c, alpha=0.25, zorder=0,
                    levels=5, thresh=0.05, label='__nolegend__')

    plt.xlabel('Scenario District Mean Load (kW)')
    plt.ylabel('Normalised district capacity for posterior scenario')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join('plots',f'{expt_name}_{n_buildings}b_posterior_designs_norm_correlation.pdf'))
    plt.show()