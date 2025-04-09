"""Process data from Estates database dataset to format compatible with Stochastic Program requirements.

Three key steps:
- Combine heat and electrical load data, i.e. generating load profiles with electrified heating
    Two options for combining plug load and gas usage data:
    1. Computing electric heating load using gas usage data and assumed COP of heat pumps
    2. Computing electric heating load assuming equal plug load and heat electricity usage
- Separate load_year of load data, saving as files with `ly_{building_id}-{year}.csv` name convention
- Rescale load data to provide selected mean load

This second step is helpful to improve efficiency of scenario generation, as buildings/load_year can be
selected by constructing schemas (only).
"""

import os
import json
import numpy as np
import pandas as pd


if __name__ == '__main__':

    # Set parameters
    # ========================
    data_dir = 'from-database'
    out_dir = 'processed'
    sname_pattern = 'solar_%s.csv'
    soutname_pattern = 'sy_{year}.csv'
    bname_pattern = 'UCam_Building_b%s.csv'
    boutname_pattern = 'ly_{building_id}-{year}.csv'

    solar_years = list(range(2010, 2020)) # valid solar_year of data
    load_year = list(range(2012, 2018)) # valid load_year of data
    building_ids = [0, 4, 8, 19, 25, 40, 58, 102, 104, 118] # available buildings

    prop_heat_electrification = 1.0 # proportion of heating load electrified
    COP = 3 # assumed coefficient of performance for heat pumps
    mean_load = 100 # kWh/timestep, mean load to rescale to

    split_type = 'COP' # 'COP' or 'equal_energy'

    # Process data
    # ========================
    hours_per_year = 24*365
    timestamps = pd.read_csv(os.path.join(data_dir, 'timestamps.csv'))
    timestamps['Timestamp (UTC)'] = pd.to_datetime(timestamps['Timestamp (UTC)'])

    # Copy over auxiliary data (take only first year)
    price_data = pd.read_csv(os.path.join(data_dir, 'pricing.csv'), header=0, nrows=hours_per_year)
    carbon_data = pd.read_csv(os.path.join(data_dir, 'carbon_intensity.csv'), header=0, nrows=hours_per_year)
    weather_data = pd.read_csv(os.path.join(data_dir, 'weather.csv'), header=0, nrows=hours_per_year)

    price_data.to_csv(os.path.join(out_dir, 'pricing.csv'), index=False)
    carbon_data.to_csv(os.path.join(out_dir, 'carbon_intensity.csv'), index=False)
    weather_data.to_csv(os.path.join(out_dir, 'weather.csv'), index=False) # note, weather data is not used

    ## Simplify and save solar data
    for syear in solar_years:
        solar_data = pd.read_csv(os.path.join(data_dir, sname_pattern%syear), header=0)['solar generation [W/kW]']
        solar_data.rename({'solar generation [W/kW]':'Solar Generation [W/kW]'}, inplace=True)
        solar_data = np.around(solar_data[:hours_per_year], 1)
        solar_data.to_csv(os.path.join(out_dir, soutname_pattern.format(year=syear)), index=False)

    ## Process load data
    # Combine heat and electrical load data then save first 8760 hours
    # of data from each year for each building to separate file
    # Use same solar generation data for all building-year pairs
    for b_id in building_ids:
        load_data = pd.read_csv(os.path.join(data_dir, bname_pattern % b_id), header=0)

        for year in load_year:
            year_first_idx = timestamps.index[timestamps['Timestamp (UTC)'].dt.year == year].min()

            elec_load = load_data.loc[year_first_idx:year_first_idx+hours_per_year-1, 'Equipment Electric Power [kWh]'].to_numpy()
            heat_load = load_data.loc[year_first_idx:year_first_idx+hours_per_year-1, 'Heating Load [kWh]'].to_numpy()

            if split_type == 'COP': # compute total electrical load using gas usage data and assumed COP of heat pumps
                norm_load = elec_load + (heat_load*prop_heat_electrification)/COP
                norm_load = norm_load/norm_load.mean()
            elif split_type == 'equal_energy': # compute total electrical load assuming equal plug load and heat electricity usage
                norm_load = 0.5*elec_load/elec_load.mean() + 0.5*heat_load/heat_load.mean()

            out_df = load_data.loc[year_first_idx:year_first_idx+hours_per_year-1].copy()
            out_df['Heating Load [kWh]'] = 0
            out_df['Equipment Electric Power [kWh]'] = np.around(norm_load*mean_load, 1)
            out_df['Solar Generation [W/kW]'] = -1 # indicate empty data
            out_df.to_csv(os.path.join(out_dir, boutname_pattern.format(building_id=b_id, year=year)), index=False)

    ## Construct and save metadata
    mdata = {
        'time_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'building_ids': building_ids,
        'load_year': load_year,
        'mean_load (kWh/step)': mean_load,
        'split_type': split_type,
        'prop_heat_electrification': prop_heat_electrification if split_type == 'COP' else None,
        'COP': COP if split_type == 'COP' else None
    }

    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(mdata, f, indent=4)
