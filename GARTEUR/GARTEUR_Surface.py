# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:11:03 2026

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

# LE_3_loc = locs.iloc[:2,1].to_numpy()
# LE_8_loc = locs.iloc[:2,2].to_numpy()
# LE_14_loc = locs.iloc[:2,3].to_numpy()
# LE_18_loc = locs.iloc[:2,4].to_numpy()
LE_21_loc = locs.iloc[:2,5].to_numpy()
# C_3_loc = locs.iloc[:2,6].to_numpy()
# C_8_loc = locs.iloc[:2,7].to_numpy()
# C_14_loc = locs.iloc[:2,8].to_numpy()
# C_18_loc = locs.iloc[:2,9].to_numpy()
C_21_loc = locs.iloc[:2,10].to_numpy()
# TR_3_loc = locs.iloc[:2,11].to_numpy()
# TR_8_loc = locs.iloc[:2,12].to_numpy()
# TR_14_loc = locs.iloc[:2,13].to_numpy()
# TR_18_loc = locs.iloc[:2,14].to_numpy()
TR_21_loc = locs.iloc[:2,15].to_numpy()

start = 110
end = 400
disp_21_TR = data.iloc[start:end,2].to_numpy().reshape(-1,1)
disp_21_C = data.iloc[start:end,3].to_numpy().reshape(-1,1)
disp_21_LE = data.iloc[start:end,4].to_numpy().reshape(-1,1)
# disp_18_TR = data.iloc[:,5].to_numpy().reshape(-1,1)
# disp_18_C = data.iloc[:,6].to_numpy().reshape(-1,1)
# disp_18_LE = data.iloc[:,7].to_numpy().reshape(-1,1)
# disp_14_TR = data.iloc[:,8].to_numpy().reshape(-1,1)
# disp_14_C = data.iloc[:,9].to_numpy().reshape(-1,1)
# disp_14_LE = data.iloc[:,10].to_numpy().reshape(-1,1)
# disp_8_TR = data.iloc[:,11].to_numpy().reshape(-1,1)
# disp_8_C = data.iloc[:,12].to_numpy().reshape(-1,1)
# disp_8_LE = data.iloc[:,13].to_numpy().reshape(-1,1)
# disp_3_TR = data.iloc[:,14].to_numpy().reshape(-1,1)
# disp_3_C = data.iloc[:,15].to_numpy().reshape(-1,1)
# disp_3_LE = data.iloc[:,16].to_numpy().reshape(-1,1)


x_coords = locs.iloc[0,[5,10,15]].to_numpy(dtype=np.float64)
y_coords = locs.iloc[1,[5,10,15]].to_numpy(dtype=np.float64)
z_coords = locs.iloc[2,[5,10,15]].to_numpy(dtype=np.float64)

data_rearr = np.hstack((disp_21_LE,disp_21_C,disp_21_TR))

time = data.iloc[start:end,0].to_numpy()
nt = time.shape[0]

# point_ids = [0,1,2]

X, T = np.meshgrid(x_coords, time)

Z = data_rearr


fig = go.Figure(
    data=[go.Surface(
        x=T,
        y=X,
        z=Z,
        colorscale="Greys", 
        opacity = 0.7,
        showscale=False,
        showlegend=False
    )]
)

fig.update_layout(
    scene=dict(
        xaxis_title="Time",
        yaxis_title="Position",
        zaxis_title="Acceleration",  
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25),
            center=dict(x=0, y=0.2, z=0),
            up=dict(x=0, y=0, z=1)), 
        xaxis=dict(title=dict(font=dict(size=40)), showticklabels=False),
        yaxis=dict(title=dict(font=dict(size=40)), showticklabels=False),
        zaxis=dict(title=dict(font=dict(size=40)), showticklabels=False)
        )
)

fig.show()

# fig = go.Figure()

# for pid in point_ids:
#     fig.add_trace(
#         go.Scatter3d(
#             x = np.full(nt, x_coords[pid]), 
#             y = time, 
#             z = data_rearr[:, pid], 
#             mode='lines+markers'))

# fig.show()


# fig = go.Figure()

# fig.add_trace(
#     go.Scatter(
#         x = time,
#         y = data_rearr[:,1]
#     ))
# fig.show()