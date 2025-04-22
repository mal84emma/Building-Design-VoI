"""Compute capacity factors for PV in each year of data."""

import os
import numpy as np
import pandas as pd


if __name__ == '__main__':

    data_dir = os.path.join('data','processed')
    sname_pattern = 'sy_{year}.csv'
    years = list(range(2010, 2020))

    capacity_factors = []

    for year in years:
        # read PV generation data
        pv_df = pd.read_csv(os.path.join(data_dir, sname_pattern.format(year=year)))
        capacity_factors.append(pv_df['Solar Generation [W/kW]'].mean()/1e3)

    for year, cf in zip(years, capacity_factors):
        print(f'Year: {year}, Capacity factor: {cf:.3f}')