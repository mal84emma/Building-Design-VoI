"""Quantify accuracy of Stochastic Program objective for a range of cases."""

import os
import numpy as np
import pandas as pd
from utils import data_handling

from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    from experiments.configs.config import *

    results_dir = os.path.join('experiments','results')

    # Compare SP objective and simulated cost for prior cases.
    # ========================================================
    for n_buildings,case in zip([5,5,1],['unconstr','constr_solar','unconstr']):
        design_results_path = os.path.join(results_dir,'prior',f'{case}_{n_buildings}b_design_results.csv')
        prior_design_results = data_handling.load_design_results(design_results_path)

        eval_results_path = os.path.join(results_dir,'prior',f'{case}_{n_buildings}b_eval_results.csv')
        eval_results = data_handling.load_eval_results(eval_results_path)
        eval_results = eval_results[:n_post_samples]

        SP_obj = prior_design_results['objective']
        sim_cost = np.mean([res['objective'] for res in eval_results])

        obj_error = sim_cost - SP_obj

        print(n_buildings,case)
        print(obj_error)
        print(obj_error/sim_cost*100)

    print('')

    # Perform comparison for posterior base case.
    # ===========================================
    n_buildings = 5
    expt_name = 'unconstr'
    info_type = 'type+mean+peak'

    for info_type in ['type','mean','peak','type+mean+peak']:
        posterior_results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info')

        posterior_design_results_files = [file for file in os.listdir(os.path.join(posterior_results_dir,'designs')) if file.endswith(".csv")]
        scenario_numbers = [int(file.split('_')[0][1:]) for file in posterior_design_results_files]
        posterior_design_results_files = [efile for scen_num,efile in sorted(zip(scenario_numbers, posterior_design_results_files))]
        post_design_results = [data_handling.load_design_results(os.path.join(posterior_results_dir, 'designs', file)) for file in posterior_design_results_files]
        posterior_SP_objs = np.array([res['objective'] for res in post_design_results])

        posterior_eval_results_files = [file for file in os.listdir(os.path.join(posterior_results_dir,'evals')) if file.endswith(".csv")]
        scenario_numbers = [int(file.split('_')[0][1:]) for file in posterior_eval_results_files]
        posterior_eval_results_files = [efile for scen_num,efile in sorted(zip(scenario_numbers, posterior_eval_results_files))]
        post_eval_results = [data_handling.load_eval_results(os.path.join(posterior_results_dir, 'evals', file)) for file in posterior_eval_results_files]
        posterior_sim_costs = np.array([np.mean([res['objective'] for res in scenario_results]) for scenario_results in post_eval_results])

        obj_errors = posterior_sim_costs - posterior_SP_objs

        print(info_type)
        print(np.mean(obj_errors))
        print(np.mean(obj_errors/posterior_sim_costs*100))
        print(np.mean(np.abs(obj_errors)))
        print(np.mean(np.abs(obj_errors)/posterior_sim_costs*100))

    # Plot distribution of errors for base case.
    # ==========================================

    fig, ax = plt.subplots()
    sns.kdeplot(obj_errors, label='Prior', ax=ax, c='k', lw=2, cut=0)
    plt.xlabel('SP objective underestimate (£)')
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    #plt.savefig(os.path.join('plots','prior_lcoes.pdf'))
    plt.show()

    fig, ax = plt.subplots()
    sns.kdeplot(obj_errors/posterior_sim_costs*100, label='Prior', ax=ax, c='k', lw=2, cut=0)
    plt.xlabel('Stochastic Program objective underestimate (%)')
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.savefig(os.path.join('plots','post_SP_obj_error.pdf'))
    plt.show()