# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 10:57:50 2025

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
x1_sub = x1o[::step, ::step]
x2_sub = x2o[::step, ::step]
y_sub = yo[::step, ::step]

train_x = np.hstack([x1_sub.ravel().reshape(-1,1),x2_sub.ravel().reshape(-1,1)])
train_y = y_sub.ravel() + 0.01*np.random.randn(y_sub.size)


# Scale input data
x_mean, x_std = train_x.mean(), train_x.std()
x_train_scaled = (train_x - x_mean) / x_std

y_mean, y_std = train_y.mean(), train_y.std()
y_train_scaled = (train_y - y_mean) / y_std

x_test_scaled = (x - x_mean) / x_std


# # Visualise to check 
# original = go.Surface(
#     x = x1o, 
#     y = x2o, 
#     z = yo,
#     colorscale = 'greys'
#     )

# training_scaled = go.Scatter3d(
#     x = x_train_scaled[:,0],
#     y = x_train_scaled[:,1],
#     z = y_train_scaled,
#     mode = 'markers',
#     )

# fig = go.Figure(data=[original,training_scaled])
# fig.show()

# Make tensors 
x_train_scaled_tensor = torch.from_numpy(x_train_scaled).double()
y_train_scaled_tensor = torch.from_numpy(y_train_scaled).double()
x_test_scaled_tensor = torch.from_numpy(x_test_scaled).double()


# Define model 
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train_scaled_tensor, y_train_scaled_tensor, likelihood):
        super(ExactGPModel, self).__init__(x_train_scaled_tensor, y_train_scaled_tensor, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        cos = gpytorch.kernels.CosineKernel().initialize(period_length=1.0)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()*cos)
       #self.covar_module.base_kernel.kernels[1].initialize(period_length=1/f)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# initialize likelihood and model 
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.initialize(noise=0.05) 
model = ExactGPModel(x_train_scaled_tensor, y_train_scaled_tensor, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer for gradient descent 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Train the model
training_iter = 300

model.train()
likelihood.train()

with gpytorch.settings.cholesky_jitter(1e-1): 
    for i in range(training_iter): 
        # Zero gradients from previous iteration 
        optimizer.zero_grad() 
        # Output from model 
        output = model(x_train_scaled_tensor) 
        # Calc loss and backprop gradients 
        loss = -mll(output, y_train_scaled_tensor) 
        loss.backward() 
        print('Iter %d/%d - Loss: %.3f  -  Noise: %.3f - Signal Variance: %.3f' % ( i + 1, training_iter, loss.item(),  model.likelihood.noise.item(), model.covar_module.outputscale.item())
         )
        optimizer.step()
            
    
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    trained_pred_dist = likelihood(model(x_test_scaled_tensor))
    observed_pred = likelihood(model(x_test_scaled_tensor))


    