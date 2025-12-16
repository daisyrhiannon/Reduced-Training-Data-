# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:56:25 2025

@author: mep24db
"""

# Import necessary packages 
import numpy as np
import math
import torch
import gpytorch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from gpytorch.constraints import Interval
import time 
from numpy import random
import plotly.graph_objects as go
import plotly.io as pio


# Sort out plotting 
pio.renderers.default = 'browser'


# Create data 
x1o, x2o = np.meshgrid(np.arange(-1, 1, 0.02), np.arange(-1, 1, 0.02))
f = 4
yo = np.sqrt(np.abs(x2o)) * np.sin(f * x1o)


# Flatten data and compile x columns 
x1o_flat = x1o.ravel().reshape(-1,1)
x2o_flat = x2o.ravel().reshape(-1,1)
y = yo.ravel()
x = np.hstack([x1o_flat,x2o_flat])


# Make training data 
x1_strip = x1o_flat[99::100]
x2_strip = x2o_flat[99::100]
y_strip = y[99::100]


# train_x = np.hstack([x1_sub_removed, x2_sub_removed])
# train_y = y_sub_removed.ravel() + 0.01*np.random.randn(y_sub_removed.size)



# Visualise to check 
original = go.Surface(
    x = x1o, 
    y = x2o, 
    z = yo,
    colorscale = 'greys'
    )

training = go.Scatter3d(
    x = x1_strip.ravel(),
    y = x2_strip.ravel(),
    z = y_strip.ravel(),
    mode = 'markers',
    )

fig = go.Figure(data=[original, training])
fig.show()
