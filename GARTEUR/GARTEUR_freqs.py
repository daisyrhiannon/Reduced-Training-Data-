# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:40:20 2026

@author: mep24db
"""

import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.io as pio
import gpytorch
import torch 
from numpy.fft import rfft, rfftfreq


# Sort out plotting 
pio.renderers.default = 'browser'


# Import data 
data = pd.read_csv('Sine10.25_0.1_5_data.csv')
locs = pd.read_csv('Accelerometer_Locations_shaker1.csv')

start = 30
end = 1000
disp_21_C = data.iloc[start:end,3].to_numpy().ravel()
time = data.iloc[start:end,0].to_numpy().ravel()

z = disp_21_C - np.mean(disp_21_C)

dx = time[1] - time[0]
fft_vals = np.abs(rfft(z))
freqs = rfftfreq(len(z), dx)
# freq = freqs[np.argmax(fft_vals[1:]) + 1]  # skip zero freq

freq = 10.25355
# print("Frequency:", freq)

fft = rfft(z)
k = np.argmax(np.abs(fft[1:])) + 1

# freq = freqs[k]
phase = 3.5
amplitude = 2 * np.abs(fft[k]) / len(z)

y_fit = amplitude * np.cos(2 * np.pi * freq * time + phase)







fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x = time, 
        y = disp_21_C, 
        mode = "lines"
    )
)


fig.add_trace(
    go.Scatter( 
        x = time, 
        y = y_fit, 
        mode = "lines"
        )
    )


fig.show()
