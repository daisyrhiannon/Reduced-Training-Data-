# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:00:36 2025

@author: mep24db
"""


import numpy as np
import math
import torch
import gpytorch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'


# Create data
xo = np.linspace(0, 1, 1000)
f = 10 
yo = np.sin(f*2*xo*math.pi)+0.01*np.random.randn(1000) # number at front of noise is standard deviation 

# data admin 
x = torch.from_numpy(xo)
y = torch.from_numpy(yo)

# # Plot to check data looks good 
# fig = px.line(x = xo, y = yo) 
# fig.show() 

# Create training data 
idx_tr = np.random.choice(1000,300, replace = False) 
xtr = (x[idx_tr]) 
ytr = (y[idx_tr]) 


# Scale input data

train_x = ((xtr)) # not scaled because it will change frequency + period 
train_y = (ytr - ytr.mean()) / ytr.std()

x_all = ((x)) 
y_all = (y-ytr.mean())/ytr.std()

# Model set up 
class ExactGPModel(gpytorch.models.ExactGP): 
    def __init__(self, train_x, train_y, likelihood): 
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood) 
        self.mean_module = gpytorch.means.ZeroMean() 
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel()) 
        self.covar_module.base_kernel.initialize(period_length=1.11/(2*f))
        self.covar_module.initialize(outputscale=0.5) # check me! 
        
    def forward(self, x): 
        mean_x = self.mean_module(x) 
        covar_x = self.covar_module(x) 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 
    
# initialize likelihood and model 
likelihood = gpytorch.likelihoods.GaussianLikelihood() 
model = ExactGPModel(train_x, train_y, likelihood) 





# Find optimal model hyperparameters 
model.train() 
likelihood.train() 

# Use gradient descent 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

# Find marginal log likelihood 
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) 

training_iter = 1000 
model.train() 
likelihood.train() 

for i in range(training_iter): 
    # Zero gradients from previous iteration 
    optimizer.zero_grad() 
    # Output from model 
    output = model(train_x) 
    # Calc loss and backprop gradients 
    loss = -mll(output, train_y) 
    loss.backward() 
    print('Iter %d/%d - Loss: %.3f  - Period: %.3f - Noise: %.3f - Signal Variance: %.3f' % ( i + 1, training_iter, loss.item(), model.covar_module.base_kernel.period_length.item(), model.likelihood.noise.item(), model.covar_module.outputscale
 )) 
    optimizer.step()
    
# Get into evaluation (predictive posterior) mode 
model.eval() 
likelihood.eval() 

# Make predictions on the test data 
with torch.no_grad(), gpytorch.settings.fast_pred_var(): 
    #trained_pred_dist = likelihood(model(train_x)) 
    observed_pred = likelihood(model(x_all))

  
# Unscale data 
y_unscaled = observed_pred.mean*ytr.std()+ytr.mean()


# # Get upper and lower confidence bounds
lower, upper = observed_pred.confidence_region()

lower_unscaled = (lower*ytr.std()+ytr.mean()).detach().numpy()
upper_unscaled = (upper*ytr.std()+ytr.mean()).detach().numpy()

# Plot figure    
fig = go.Figure()

fig.add_trace(go.Scatter(x=xo, y=yo, mode='lines', name='Original'))
fig.add_trace(go.Scatter(x=xo, y=y_unscaled, mode='lines', name='Prediction'))
fig.add_trace(go.Scatter(x=xtr.ravel(), y=ytr, mode='markers', name='Training'))
fig.add_trace(go.Scatter(x=np.concatenate([xo,xo[::-1]]), y=np.concatenate([upper_unscaled, lower_unscaled[::-1]]), fill = 'toself', name="Confidence Region"))


fig.show()


MSE = (gpytorch.metrics.mean_squared_error(observed_pred,y_all,squared=True))
MSE2 =(gpytorch.metrics.mean_squared_error(y_unscaled,y,squared=True)) 
NMSE = 100* MSE * (train_y.std())**2
print(f'NMSE: {NMSE:.3f}')