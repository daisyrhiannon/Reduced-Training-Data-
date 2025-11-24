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
yo = np.sin(10*xo*math.pi)+0.01*np.random.rand(1000) 

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
train_x = ((xtr - xtr.mean()) / xtr.std()).unsqueeze(-1)
train_y = (ytr - ytr.mean()) / ytr.std()

x_all = ((x - x.mean()) / x.std()).unsqueeze(-1)

# Model set up 
class ExactGPModel(gpytorch.models.ExactGP): 
    def __init__(self, train_x, train_y, likelihood): 
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood) 
        self.mean_module = gpytorch.means.ConstantMean() 
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) 
        
    def forward(self, x): 
        mean_x = self.mean_module(x) 
        covar_x = self.covar_module(x) 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 
    
# initialize likelihood and model 
likelihood = gpytorch.likelihoods.GaussianLikelihood() 
model = ExactGPModel(train_x, train_y, likelihood) 
# Set the noise value (optional)
# likelihood.initialize(noise=0.1)
# Freeze it
#likelihood.raw_noise.requires_grad_(False)

# model.covar_module.base_kernel.initialize(period_length=0.2)




# Find optimal model hyperparameters 
model.train() 
likelihood.train() 

# Use gradient descent 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

# Find marginal log likelihood 
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) 

training_iter = 300 
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
    print('Iter %d/%d - Loss: %.3f - Period: %.3f - Noise: %.3f' % ( i + 1, training_iter, loss.item(), model.covar_module.base_kernel.period_length.item(), model.likelihood.noise.item() )) 
    optimizer.step()
    
# Get into evaluation (predictive posterior) mode 
model.eval() 
likelihood.eval() 

# Make predictions on the test data 
with torch.no_grad(), gpytorch.settings.fast_pred_var(): 
    trained_pred_dist = likelihood(model(x_all)) 
    observed_pred = likelihood(model(x))

# Plot figure    
# fig = go.Figure()

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
# Get upper and lower confidence bounds
lower, upper = observed_pred.confidence_region()

ax.plot(xtr,ytr,".")
ax.plot(xo,observed_pred.mean)
# Shade between the lower and upper confidence bounds
ax.fill_between(x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
ax.set_ylim([-3, 3])
ax.legend(['Observed Data', 'Mean', 'Confidence'])



# fig.add_trace(go.Scatter(x=xo, y=y, mode='lines', name='Original'))
# fig.add_trace(go.Scatter(x=xo, y=observed_pred.mean, mode='lines', name='Prediction'))
# fig.add_trace(go.Scatter(x=xtr.ravel(), y=ytr, mode='markers', name='Training'))

ax.plot(xo, y)
# ax.plot(train_x,train_y,".")
fig.show()