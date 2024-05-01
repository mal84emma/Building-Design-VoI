"""Visualise electricity and gas data for a selected building file."""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualise_building_data(building_file_path, fpath='temp.html', show=False):

    # load data
    building_data = pd.read_csv(building_file_path)
    elec_data = building_data['Equipment Electric Power [kWh]'].to_numpy()
    heat_data = building_data['Heating Load [kWh]'].to_numpy()

    # create plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=np.arange(len(elec_data)),
        y=elec_data,
        name='Electricity load',
        connectgaps=False
        ),
        secondary_y=False
    )

    if np.any(heat_data > 0.0):
        fig.add_trace(go.Scatter(
            x=np.arange(len(heat_data)),
            y=heat_data,
            name='Heating load',
            connectgaps=False
            ),
            secondary_y=True
        )

    fig.update_layout(
        xaxis_title='Hour',
        xaxis=dict(rangeslider=dict(visible=True)),
        title=os.path.split(building_file_path)[-1].replace('.csv',''),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            )
        )

    fig.update_yaxes(title_text="Equipment load [kWh]", secondary_y=False)
    if np.any(heat_data > 0.0):
        fig.update_yaxes(title_text="Heating load [kWh]", secondary_y=True)

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    fig.update_yaxes(fixedrange=False)

    fig.update_layout(title_x=0.5) # center title

    fig.write_html(fpath)

    if show:
        fig.show()


if __name__ == '__main__':

    # iterable of building ids to visualise data
    ids = [0, 4, 8, 19, 25, 40, 58, 102, 104, 118]
    year = 2012

    data_dir = 'processed' # 'from-database'
    bname_pattern = 'UCam_Building_b%s.csv'
    bname_pattern = f'ly_%s-{year}.csv'

    for id in ids:
        for year in range(2012, 2018):
            bname_pattern = f'ly_%s-{year}.csv'
            building_file_path = os.path.join(data_dir, bname_pattern % id)
            fpath = 'temp.html'
            show = True

            visualise_building_data(building_file_path,fpath,show)
