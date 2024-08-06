"""Compute capacity factors for PV in each year of data."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import numpy as np
import pandas as pd


if __name__ == '__main__':

    data_dir = os.path.join('data','processed')
    bname_pattern = 'ly_{id}-{year}.csv'
    years = list(range(2012, 2018))
    building_id = 0

    capacity_factors = []

    for year in years:
        # read PV generation data
        pv_df = pd.read_csv(os.path.join(data_dir, bname_pattern.format(id=building_id, year=year)))
        capacity_factors.append(pv_df['Solar Generation [W/kW]'].mean()/1e3)

    for year, cf in zip(years, capacity_factors):
        print(f'Year: {year}, Capacity factor: {cf:.3f}')