"""Plotting utilities."""

import numpy as np

import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots


def init_profile_fig(title=None, y_titles=None) -> Figure:

    # create plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.update_layout(
        xaxis_title='Hour',
        xaxis=dict(rangeslider=dict(visible=True)),
        title=title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            )
        )

    if 'primary' in y_titles:
        fig.update_yaxes(title_text=y_titles['primary'], secondary_y=False)
    if 'secondary' in y_titles:
        fig.update_yaxes(title_text=y_titles['secondary'], secondary_y=True)

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

    return fig

def add_profile(fig, profile, name=None, secondary_y=False) -> Figure:

    fig.add_trace(go.Scatter(
        x=np.arange(len(profile)),
        y=profile,
        name=name,
        connectgaps=False
        ),
        secondary_y=False
    )

    return fig