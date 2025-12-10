# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:26:51 2025

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
x_train_scaled_tensor = torch.from_numpy(x_train_scaled).float()
y_train_scaled_tensor = torch.from_numpy(y_train_scaled).float()
x_test_scaled_tensor = torch.from_numpy(x_test_scaled).float()


# Define model 
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train_scaled_tensor, y_train_scaled_tensor, likelihood):
        super(ExactGPModel, self).__init__(x_train_scaled_tensor, y_train_scaled_tensor, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        cos = gpytorch.kernels.CosineKernel()
        rbf = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        product = rbf*cos
        self.covar_module = gpytorch.kernels.ScaleKernel(product)
        # self.covar_module.base_kernel.kernels[1].initialize(period_length=1/f)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

# initialize likelihood and model 
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = torch.tensor(1e-4)
model = ExactGPModel(x_train_scaled_tensor, y_train_scaled_tensor, likelihood)


# initialise hyperparameters 
product = model.covar_module.base_kernel
rbf = product.kernels[0]   
cos = product.kernels[1] 
cos.period_length = torch.tensor(2*np.pi / f).float()
rbf.lengthscale = torch.tensor([0.3, 0.5]).float()
model.covar_module.outputscale = torch.tensor(np.var(y)).float()
model.mean_module.constant = torch.tensor(np.mean(y)).float()


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

breakpoint()

with gpytorch.settings.cholesky_jitter(1e-1): 
    for i in range(training_iter): 
        # Zero gradients from previous iteration 
        optimizer.zero_grad() 
        # Output from model 
        output = model(x_train_scaled_tensor) 
        # Calc loss and backprop gradients 
        loss = -mll(output, y_train_scaled_tensor) 
        loss.backward() 
        ls = rbf.lengthscale.detach().cpu().numpy()
        print('Iter %d/%d - Loss: %.3f  -  Noise: %.3f - Signal Variance: %.3f - Cos Period: %.3f - Lengthscales: %s'  % ( i + 1, training_iter, loss.item(),  model.likelihood.noise.item(), model.covar_module.outputscale.item(), cos.period_length.item(), np.array2string(ls, precision=3)))
        optimizer.step()
            
    
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    trained_pred_dist = likelihood(model(x_test_scaled_tensor))
    observed_pred = likelihood(model(x_test_scaled_tensor))

    
# Unsacle data 
results_unscaled = observed_pred.mean*y_std+y_mean


# Plot results 
prediction = go.Surface(
    z=results_unscaled.numpy().reshape(x1o.shape),
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
    x=train_x[:,0],
    y=train_x[:,1], 
    z=train_y, 
    mode='markers', 
    marker=dict(size=5, color='black', symbol='circle'),
    name='Training Data'
)

fig = go.Figure(data=[prediction, original, training])
fig.update_layout(legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.7)',
        bordercolor='black',
        borderwidth=1))
fig.show()
    