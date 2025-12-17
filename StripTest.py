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


# # Make training data 
x1_strip = x1o[:,90:100].ravel().reshape(-1,1)
x2_strip = x2o[:,90:100].ravel().reshape(-1,1)
y_strip = yo[:,90:100].ravel().reshape(-1,1)

n_train = 10
rng = np.random.default_rng(12)
random_indices = rng.choice(1000, n_train, replace = False)

x1_train = x1_strip[random_indices]
x2_train = x2_strip[random_indices]
y_train = y_strip[random_indices]


 
train_x = np.hstack([x1_train, x2_train])
train_y = y_train.ravel() + 0.01*np.random.randn(y_train.size)



# Visualise to check 
original = go.Surface(
    x = x1o, 
    y = x2o, 
    z = yo,
    colorscale = 'greys'
    )

training = go.Scatter3d(
    x = train_x[:,0].ravel(),
    y = train_x[:,1].ravel(),
    z = train_y.ravel(),
    mode = 'markers',
    )

fig = go.Figure(data=[original, training])
fig.show()
