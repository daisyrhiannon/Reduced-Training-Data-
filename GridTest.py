# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 13:43:14 2025

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
step=10
x1_sub = x1o[::step, ::step].ravel().reshape(-1,1)
x2_sub = x2o[::step, ::step].ravel().reshape(-1,1)
y_sub = yo[::step, ::step].ravel().reshape(-1,1)


# pull from grid 
Num_out = 40 
rng = np.random.default_rng(21)
random_indices = rng.choice(100, 100, replace = False)
indices_to_remove = random_indices[:Num_out]

x1_sub_removed = np.delete(x1_sub, indices_to_remove, axis =0)
x2_sub_removed = np.delete(x2_sub, indices_to_remove, axis = 0)
y_sub_removed = np.delete(y_sub, indices_to_remove, axis = 0)

train_x = np.hstack([x1_sub_removed, x2_sub_removed])
train_y = y_sub_removed.ravel() + 0.01*np.random.randn(y_sub_removed.size)



# Visualise to check 
original = go.Surface(
    x = x1o, 
    y = x2o, 
    z = yo,
    colorscale = 'greys'
    )

training = go.Scatter3d(
    x = train_x[:,0],
    y = train_x[:,1],
    z = train_y,
    mode = 'markers',
    )

fig = go.Figure(data=[training])
fig.show()
