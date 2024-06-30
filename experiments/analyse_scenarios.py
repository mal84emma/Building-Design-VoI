"""Analyse aggregate load profiles from scenarios."""

import os
import numpy as np
import pandas as pd
from utils import data_handling
from load_scenario_reduction import get_scenario_stats

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    dataset_dir = os.path.join('data','processed')
    building_fname_pattern = 'ly_{id}-{year}.csv'

    # Load prior scenario samples.
    scenarios_path = os.path.join('experiments','results','sampled_scenarios.csv')
    scenarios = data_handling.load_scenarios(scenarios_path)

    # Load profiles data.
    building_year_pairs = np.unique(np.concatenate(scenarios,axis=0),axis=0)
    load_profiles = {
        f'{building_id}-{year}': pd.read_csv(
            os.path.join(dataset_dir, building_fname_pattern.format(id=building_id, year=year)),
            usecols=['Equipment Electric Power [kWh]']
            )['Equipment Electric Power [kWh]'].to_numpy()\
                for (building_id, year) in building_year_pairs
    }

    # Compute stats of aggregate load profiles for scenarios.
    stats = np.array([get_scenario_stats(s, load_profiles) for s in scenarios])

    print(f'Max peak load: {np.max(stats[:,1])}')
    print(f'Min peak load: {np.min(stats[:,1])}')
    print(f'Mean peak load: {np.mean(stats[:,1])}')
    print(f'Peak load std: {np.std(stats[:,1])}')

    # Plot results.
    fig, ax = plt.subplots()
    sns.kdeplot(stats[:,1], ax=ax, c='k', cut=0, levels=200)
    plt.xlabel("Max aggregate load in scenario (kW)")
    plt.ylabel("Density")
    ax.set_yticks([])
    ax.set_yticklabels([])
    #plt.savefig(os.path.join('plots',"..."), format="pdf", bbox_inches="tight")
    plt.show()