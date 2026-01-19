# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 10:08:14 2026

@author: mep24db
"""

import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.io as pio

# Sort out plotting 
pio.renderers.default = 'browser'

data = pd.read_csv('Sine10.25_0.1_5_data.csv')
locs = pd.read_csv('Accelerometer_Locations_shaker1.csv')

LE_3_loc = locs.iloc[:2,1].to_numpy()
LE_8_loc = locs.iloc[:2,2].to_numpy()
LE_14_loc = locs.iloc[:2,3].to_numpy()
LE_18_loc = locs.iloc[:2,4].to_numpy()
LE_21_loc = locs.iloc[:2,5].to_numpy()
C_3_loc = locs.iloc[:2,6].to_numpy()
C_8_loc = locs.iloc[:2,7].to_numpy()
C_14_loc = locs.iloc[:2,8].to_numpy()
C_18_loc = locs.iloc[:2,9].to_numpy()
C_21_loc = locs.iloc[:2,10].to_numpy()
TR_3_loc = locs.iloc[:2,11].to_numpy()
TR_8_loc = locs.iloc[:2,12].to_numpy()
TR_14_loc = locs.iloc[:2,13].to_numpy()
TR_18_loc = locs.iloc[:2,14].to_numpy()
TR_21_loc = locs.iloc[:2,15].to_numpy()


disp_21_TR = data.iloc[:,2].to_numpy().reshape(-1,1)
disp_21_C = data.iloc[:,3].to_numpy().reshape(-1,1)
disp_21_LE = data.iloc[:,4].to_numpy().reshape(-1,1)
disp_18_TR = data.iloc[:,5].to_numpy().reshape(-1,1)
disp_18_C = data.iloc[:,6].to_numpy().reshape(-1,1)
disp_18_LE = data.iloc[:,7].to_numpy().reshape(-1,1)
disp_14_TR = data.iloc[:,8].to_numpy().reshape(-1,1)
disp_14_C = data.iloc[:,9].to_numpy().reshape(-1,1)
disp_14_LE = data.iloc[:,10].to_numpy().reshape(-1,1)
disp_8_TR = data.iloc[:,11].to_numpy().reshape(-1,1)
disp_8_C = data.iloc[:,12].to_numpy().reshape(-1,1)
disp_8_LE = data.iloc[:,13].to_numpy().reshape(-1,1)
disp_3_TR = data.iloc[:,14].to_numpy().reshape(-1,1)
disp_3_C = data.iloc[:,15].to_numpy().reshape(-1,1)
disp_3_LE = data.iloc[:,16].to_numpy().reshape(-1,1)


x_coords = locs.iloc[0,1:].to_numpy(dtype=np.float64)
y_coords = locs.iloc[1,1:].to_numpy(dtype=np.float64)
z_coords = locs.iloc[2,1:].to_numpy(dtype=np.float64)

data_rearr = np.hstack((disp_3_LE,disp_8_LE,disp_14_LE,disp_18_LE,disp_21_LE,disp_3_C,disp_8_C,disp_14_C,disp_18_C,disp_21_C,disp_3_TR,disp_8_TR,disp_14_TR,disp_18_TR,disp_21_TR))

# # Starting surface 
# original = go.Scatter3d(
#     x = x_coords,
#     y = y_coords,
#     z = z_coords, 
#     mode = "markers"
#     )

# fig = go.Figure([original])

# fig.show()


nt, N = data_rearr.shape 

# Start point 
fig = go.Figure(
    data=[go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=data_rearr[0],
        mode="markers",
        marker=dict(
            size=3,
            color=data_rearr[0],
            colorscale="Viridis",
            cmin=data_rearr.min(),
            cmax=data_rearr.max(),
            opacity=0.8
        )
    )]
)

# Build frames directly from z rows
fig.frames = [
    go.Frame(
        data=[go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=data_rearr[t],
            marker=dict(color=data_rearr[t])
        )],
        name=str(t)
    )
    for t in range(nt)
]

# Play / pause + fixed axes
fig.update_layout(
    scene=dict(
        zaxis=dict(range=[data_rearr.min(), data_rearr.max()])
    ),
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None,
                         {"frame": {"duration": 50, "redraw": True},
                          "fromcurrent": True}]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [[None],
                         {"mode": "immediate"}]
            }
        ]
    }]
)

fig.show()