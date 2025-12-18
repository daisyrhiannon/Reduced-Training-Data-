# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 14:53:48 2025

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
from codecarbon import EmissionsTracker
import time

import plotly.graph_objects as go
import plotly.io as pio


# Sort out plotting 
pio.renderers.default = 'browser'

# Start timer 
start = time.perf_counter()

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
Num_out = 55
rng = np.random.default_rng(21)
random_indices = rng.choice(100, 100, replace = False)
indices_to_remove = random_indices[:Num_out]

x1_sub_removed = np.delete(x1_sub, indices_to_remove, axis =0)
x2_sub_removed = np.delete(x2_sub, indices_to_remove, axis = 0)
y_sub_removed = np.delete(y_sub, indices_to_remove, axis = 0)

x_train = np.hstack([x1_sub_removed, x2_sub_removed])
y_train = y_sub_removed.ravel() + 0.01*np.random.randn(y_sub_removed.size)


# # Scale input data
# x_mean = x_train.mean(axis = 0, keepdims=True)
# x_std = x_train.std(axis = 0, keepdims=True)
# x_train_scaled = (x_train - x_mean) / x_std

# y_mean, y_std = y_train.mean(), y_train.std()
# y_train_scaled = (y_train - y_mean) / y_std

# x_test_scaled = (x - x_mean) / x_std


# # Visualise to check 
# original = go.Surface(
#     x = x1o, 
#     y = x2o, 
#     z = yo,
#     colorscale = 'greys'
#     )

# training_scaled = go.Scatter3d(
#     x = x_train[:,0],
#     y = x_train[:,1],
#     z = y_train,
#     mode = 'markers',
#     )

# fig = go.Figure(data=[original,training_scaled])
# fig.show()

# Make tensors 
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
x_test_tensor = torch.from_numpy(x).float()


# Define model 
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train_scaled_tensor, y_train_scaled_tensor, likelihood):
        super(ExactGPModel, self).__init__(x_train_scaled_tensor, y_train_scaled_tensor, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
     

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# initialize likelihood and model 
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# likelihood.noise = torch.tensor(1e-4)
model = ExactGPModel(x_train_tensor, y_train_tensor, likelihood)


# # Set hyperparameter bounds 
# # model.covar_module.outputscale = torch.tensor(np.var(y)).float() # For fixed variance
# vary = np.var(y)
# model.covar_module.raw_outputscale_constraint = Interval(vary*0.9, vary*1.1)
# startpoint_var = (vary*0.9) + ((vary*1.1)-(vary*0.9)) * torch.rand_like(torch.tensor(vary*0.9))
# model.covar_module.initialize(outputscale = torch.tensor(startpoint_var))

# # model.covar_module.base_kernel.lengthscale =0.72 # For fixed lengthscale 
# fixl = 0.72
# model.covar_module.base_kernel.raw_lengthscale_constraint = Interval(fixl*0.9, fixl*1.1)
# startpoint_ls = (fixl*0.9) + ((fixl*1.1)-(fixl*0.9)) * torch.rand_like(torch.tensor(fixl*0.9))
# model.covar_module.base_kernel.initialize(lengthscale = torch.tensor(startpoint_ls))


# model.covar_module.raw_outputscale.requires_grad_(False)
# model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer for gradient descent 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Train the model
training_iter = 1000
model.train()
likelihood.train()


for i in range(training_iter): 
    # Zero gradients from previous iteration 
    optimizer.zero_grad() 
    # Output from model 
    output = model(x_train_tensor) 
    # Calc loss and backprop gradients 
    loss = -mll(output, y_train_tensor) 
    loss.backward() 
    print('Iter %d/%d - Loss: %.3f  -  Noise: %.3f - Signal Variance: %.3f - Lengthscale: %.3f' % ( i + 1, training_iter, loss.item(),  model.likelihood.noise.item(), model.covar_module.outputscale.item(), model.covar_module.base_kernel.lengthscale.item()))
    optimizer.step()
        
    
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #trained_pred_dist = likelihood(model(x_train_scaled_tensor))
    observed_pred = likelihood(model(x_test_tensor))
    
    
# # Unsacle data 
# results_unscaled = observed_pred.mean*y_std+y_mean

def MSE(ypred,ytest):
    MSE = np.mean(((ypred-ytest)**2))
    return MSE

def nMSE(ypred,ytest):
    nMSE = 100*(np.mean(((ypred-ytest)**2))/np.std(ytest))
    return nMSE
   
error = MSE(observed_pred.mean.numpy(),y) 
errorN = nMSE(observed_pred.mean.numpy(),y)

print( f" NMSE = {errorN}" )

# Plot results 
prediction = go.Surface(
    z=observed_pred.mean.numpy().reshape(x1o.shape),
    x=x1o, 
    y=x2o, 
    colorscale="jet", 
    name='Prediction',
    opacity=0.9,
    showscale=False,
    showlegend=True
)

original = go.Surface(
    z=yo, 
    x=x1o, 
    y=x2o, 
    colorscale='greys', 
    name='Original', 
    opacity=0.7,
    showscale=False,
    showlegend=True
)

training = go.Scatter3d(
    x=x_train[:,0],
    y=x_train[:,1], 
    z=y_train, 
    mode='markers', 
    marker=dict(size=5, color='black', symbol='circle'),
    name='Training Data'
)

fig = go.Figure(data=[prediction, original, training])
fig.update_layout(
    title = f"NMSE = {errorN}",
    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.7)',
        bordercolor='black',
        borderwidth=1))
# fig.show()

# Stop timer 
end = time.perf_counter()
print(f"Runtime: {end - start:.6f} seconds")
