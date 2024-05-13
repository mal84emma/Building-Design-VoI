"""Plot normalised load-duration curves for each building-year with and without heat electrification.

Data is split into years due to changes in usage trend clashing with normalisation.
"""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_load_duration_curves(building_df, prop_eleced, COP, levels, split):
    """Compute points for normalised load-duration curves with and without heat electrification."""

    if split == 'COP':
        # compute total electrical load using gas usage data and assumed COP of heat pumps
        elec_without = building_df['Equipment Electric Power [kWh]'].to_numpy()
        elec_with = elec_without + building_df['Heating Load [kWh]'].to_numpy()*prop_eleced/COP

        elec_without /= elec_without.mean()
        elec_with /= elec_with.mean()

    if split == 'equal_energy':
        # comppute total electrical load assuming equal plug load and heat electricity usage
        elec = building_df['Equipment Electric Power [kWh]'].to_numpy()
        heat = building_df['Heating Load [kWh]'].to_numpy()

        elec_without = elec/elec.mean()
        elec_with = 0.5*elec/elec.mean() + 0.5*heat/heat.mean()

    without_durations = [np.count_nonzero(elec_without >= level) for level in levels]
    with_durations = [np.count_nonzero(elec_with >= level) for level in levels]

    without_durations = [d/len(elec_without) for d in without_durations]
    with_durations = [d/len(elec_with) for d in with_durations]

    return without_durations, with_durations


if __name__ == '__main__':

    data_dir = os.path.join('data','from-database')
    bname_pattern = 'UCam_Building_b%s.csv'
    years = list(range(2012, 2018))
    building_ids = [0, 4, 8, 19, 25, 40, 58, 102, 104, 118] # available buildings

    prop_heat_electrification = 1.0 # proportion of heating load electrified
    COP = 3 # assumed coefficient of performance for heat pumps

    # Compute normalised load-duration curves
    levels = np.arange(0, 11.01, 0.01)

    hours_per_year = 24*365
    timestamps = pd.read_csv(os.path.join(data_dir, 'timestamps.csv'))
    timestamps['Timestamp (UTC)'] = pd.to_datetime(timestamps['Timestamp (UTC)'])

    for split in ['COP','equal_energy']:
        without_curves = []
        with_curves = []
        for b_id in building_ids[:-1]: # [:-1]
            building_without_curves = []
            building_with_curves = []

            load_data = pd.read_csv(os.path.join(data_dir, bname_pattern % b_id), header=0)
            for year in years:
                year_first_idx = timestamps.index[timestamps['Timestamp (UTC)'].dt.year == year].min()
                year_df = load_data.loc[year_first_idx:year_first_idx+hours_per_year-1].copy()

                without_durations, with_durations = compute_load_duration_curves(year_df, prop_heat_electrification, COP, levels, split)

                building_without_curves.append(without_durations)
                building_with_curves.append(with_durations)
            without_curves.extend(building_without_curves)
            with_curves.extend(building_with_curves)

            # Plot normalised load-duration curves for all building-years
            plt.figure()
            for wo,w in zip(building_without_curves, building_with_curves):
                plt.plot([v if v > 0.0 else None for v in wo], levels, 'b', alpha=0.2)
                plt.plot([v if v > 0.0 else None for v in w], levels, 'r', alpha=0.2)
            plt.hlines(1, 0, 1, 'k', alpha=0.5)
            plt.ylim(0)
            plt.xlim(0, 1)
            plt.ylabel('Normalised load')
            plt.xlabel('Proportion of hours exceeding load')
            plt.show()
            plt.clf()
            plt.close()

        # Plot mean normalised load-duration curves with uncertainty bounds
        plt.figure()
        without_curves = np.array(without_curves)
        with_curves = np.array(with_curves)
        plt.plot([v if v > 0.0 else None for v in without_curves.mean(axis=0)], levels, 'b')
        plt.fill_betweenx(levels, without_curves.mean(axis=0) - without_curves.std(axis=0), without_curves.max(axis=0) + without_curves.std(axis=0), where=without_curves.mean(axis=0) > 0.0, color='b', alpha=0.2)
        plt.plot([v if v > 0.0 else None for v in with_curves.mean(axis=0)], levels, 'r')
        plt.fill_betweenx(levels, with_curves.mean(axis=0) - with_curves.std(axis=0), with_curves.max(axis=0) + with_curves.std(axis=0), where=with_curves.mean(axis=0) > 0.0, color='r', alpha=0.2)
        plt.hlines(1, 0, 1, 'k', alpha=0.5)
        plt.ylim(0)
        plt.xlim(0, 1)
        plt.ylabel('Normalised load')
        plt.xlabel('Proportion of hours exceeding load')
        plt.show()
        plt.clf()
        plt.close()