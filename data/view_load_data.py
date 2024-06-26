"""Visualise electricity and gas data for a selected building file."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append('..')
from utils.plotting import init_profile_fig, add_profile

def visualise_building_data(building_file_path, fpath='temp.html', show=False):

    # load data
    building_data = pd.read_csv(building_file_path)
    elec_data = building_data['Equipment Electric Power [kWh]'].to_numpy()
    heat_data = building_data['Heating Load [kWh]'].to_numpy()

    # create plot
    fig = init_profile_fig(
        title=os.path.split(building_file_path)[-1].replace('.csv',''),
        y_titles={'primary': 'Equipment load [kWh]', 'secondary': 'Heating load [kWh]'}
    )

    fig = add_profile(fig, elec_data, name='Electricity load')
    if np.any(heat_data > 0.0): fig = add_profile(fig, heat_data, name='Heating load', yaxis='y2')

    fig.write_html(fpath)
    if show: fig.show()


if __name__ == '__main__':

    # iterable of building ids to visualise data
    ids = [0, 4, 8, 19, 25, 40, 58, 102, 104, 118]
    year = 2012

    data_dir = 'processed' # 'from-database'
    bname_pattern = 'UCam_Building_b%s.csv'
    bname_pattern = f'ly_%s-{year}.csv'

    # for id in ids:
    #     visualise_building_data(os.path.join('from-database', bname_pattern % ids[0]), 'temp.html', True)

    for id in [104]: # ids:
        for year in range(2012, 2018):
            bname_pattern = f'ly_%s-{year}.csv'
            building_file_path = os.path.join(data_dir, bname_pattern % id)
            fpath = 'temp.html'
            show = True

            visualise_building_data(building_file_path,fpath,show)
