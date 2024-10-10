"""Analyse aggregate load profiles from scenarios."""

import os
import numpy as np
import pandas as pd
from utils import data_handling
from load_scenario_reduction import get_scenario_stats

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    from experiments.configs.config import *

    n_buildings = 6

    dataset_dir = os.path.join('data','processed')
    building_fname_pattern = 'ly_{id}-{year}.csv'

    # Load prior scenario samples.
    scenarios_path = os.path.join('experiments','results',f'sampled_scenarios_{n_buildings}b.csv')
    scenarios = data_handling.load_scenarios(scenarios_path)
    n_buildings = scenarios.shape[1]

    # Load profiles data.
    building_year_pairs = np.unique(np.concatenate(scenarios,axis=0),axis=0)
    load_profiles = {
        f'{building_id}-{year}': pd.read_csv(
            os.path.join(dataset_dir, building_fname_pattern.format(id=building_id, year=year)),
            usecols=['Equipment Electric Power [kWh]']
            )['Equipment Electric Power [kWh]'].to_numpy()\
                for (building_id, year) in building_year_pairs
    }

    mean_load = 100

    per_building_peaks = {b_id: [np.max(load_profiles[f'{b_id}-{year}']) for year in years] for b_id in ids}
    all_building_peaks = np.array([np.max(v) for k,v in load_profiles.items()])

    # Compute stats of aggregate load profiles for scenarios.
    stats = np.array([get_scenario_stats(s, load_profiles) for s in scenarios])

    print(f'Max peak load: {np.max(stats[:,2])}')
    print(f'Min peak load: {np.min(stats[:,2])}')
    print(f'Mean peak load: {np.mean(stats[:,2])}')
    print(f'Peak load std: {np.std(stats[:,2])}')

    # Plot distribution of peak load factors.
    ## Use this to demonstrate peak smoothing effect in district systems.
    fig, ax = plt.subplots()
    sns.kdeplot(all_building_peaks/mean_load, ax=ax, c='b', cut=0, levels=200, label='Individual')
    sns.kdeplot(stats[:,2]/(n_buildings*mean_load), ax=ax, c='k', cut=0, levels=200, label='Aggregate')
    plt.xlabel("Annual peak load factor")
    plt.ylabel("Density")
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.legend()
    #plt.savefig(os.path.join('plots',"..."), format="pdf", bbox_inches="tight")
    plt.show()

    # Plot distribution of peak variability for each building
    fig, ax = plt.subplots()
    for b_id in ids:
        sns.kdeplot(per_building_peaks[b_id], ax=ax, cut=0, levels=200, label=b_id)
    plt.xlabel("Annual peak load factor")
    plt.ylabel("Density")
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.legend()
    #plt.savefig(os.path.join('plots',"..."), format="pdf", bbox_inches="tight")
    plt.show()

    # Plot distribution of relative peaks compared to building mean
    ## Use this to derive posterior dist. for peak measurement.
    fig, ax = plt.subplots()
    rel_peaks = np.array([per_building_peaks[b_id]/np.mean(per_building_peaks[b_id]) for b_id in ids]).flatten()
    sns.kdeplot(rel_peaks, ax=ax, c='k', cut=0, levels=200)
    plt.xlabel("Relative annual peak")
    plt.ylabel("Density")
    ax.set_yticks([])
    ax.set_yticklabels([])
    #plt.savefig(os.path.join('plots',"..."), format="pdf", bbox_inches="tight")
    plt.show()