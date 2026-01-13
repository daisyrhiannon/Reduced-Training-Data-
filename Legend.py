# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 16:49:35 2026

@author: mep24db
"""
import plotly.graph_objects as go
import numpy as np

# Tiny dummy surface (2x2 is enough)
z_dummy = np.array([[0, 0],
                    [0, 0]])

legend_fig = go.Figure()

legend_fig.add_trace(
    go.Surface(
        z=z_dummy,
        colorscale="Jet",
        showscale=False,
        name="Prediction",
        showlegend=True
    )
)

legend_fig.add_trace(
    go.Surface(
        z=z_dummy,
        colorscale="greys",
        showscale=False,
        name="Original",
        showlegend=True
    )
)

legend_fig.add_trace(
    go.Scatter(
        x=[None], y = [None],
        mode="markers",
        marker=dict(size=16, color="black", symbol="circle"),
        name="Training Data"
    )
)

legend_fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    legend=dict(
        font=dict(size=30),
        itemsizing="constant",
        x=0,
        y=1,
        xanchor="left",
        yanchor="top"
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    width=300,
    height=200
)

legend_fig.show()
