# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 14:17:32 2025

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

# Make data + data admin
x1o, x2o = np.meshgrid(np.arange(-1, 1, 0.02), np.arange(-1, 1, 0.02))
x1o_flat = x1o.ravel()   # or x1o.reshape(-1)
x2o_flat = x2o.ravel()
yo = np.sqrt(np.abs(x2o)) * np.sin(4 * x1o)
x = torch.from_numpy(np.hstack([x1o_flat.reshape(-1,1), x2o_flat.reshape(-1,1)])).float()
y = torch.from_numpy(yo.ravel()).float()

# Make training data 
step=10
x1_sub = x1o[::step, ::step]
x2_sub = x2o[::step, ::step]
y_sub = yo[::step, ::step]

x_train = torch.from_numpy(np.hstack([x1_sub.ravel().reshape(-1,1),x2_sub.ravel().reshape(-1,1)])).float()
# y_train = torch.from_numpy(y_sub.ravel()).float()

y_train_silent = torch.from_numpy(y_sub.ravel()).float() 

with torch.random.fork_rng(): 
    torch.manual_seed(25) 
    y_train = y_train_silent + 0.01*torch.randn_like(y_train_silent)




# Make test data
x_test = x
x_test=x_test.float()


# Compute scalers (save these to unscale later)
x_mean, x_std = x_train.mean(0), x_train.std(0)
x_std[x_std == 0] = 1.0
x_train_scaled = (x_train - x_mean) / x_std

# y_mean, y_std = y_train.mean(), y_train.std()
# if y_std == 0: y_std = 1.0
# y_train_scaled = (y_train - y_mean) / y_std

# x_mean_2, x_std_2 = x_test.mean(), x_test.std()
# if x_std_2 == 0: x_std_2 = 1.0
# x_test_scaled = (x_test - x_mean_2) / x_std_2

# # replace train tensors (and ensure same transform applied to x_test later)
# x_train_2 = x_train_scaled
# y_train_2 = y_train_scaled

# x_test_2= x_test_scaled


# # visualise training data in 3d
# x_np = x_train.cpu().numpy()
# y_np = y_train.cpu().numpy()

# original = go.Surface(
#     z=yo, 
#     x=x1o, 
#     y=x2o, 
#     colorscale='greys', 
#     name='Original', 
#     opacity=0.7,
#     showscale=False,
#     showlegend=True
# )

# tester = go.Scatter3d(
#     x=x_np[:,0],
#     y=x_np[:,1],
#     z=y_np,
#     mode='markers', 
#     marker=dict(size=5, color='black', symbol='circle'),
#     name='Training Data'
#     )

# fig=go.Figure(data=[tester, original])
# fig.show()




# Model class 
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()*gpytorch.kernels.CosineKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_train, y_train, likelihood)


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# with gpytorch.settings.cholesky_jitter(1e-3):
#     with torch.no_grad():
#         observed_pred = likelihood(model(x_test))


training_iter = 100

model.train()
likelihood.train()

with gpytorch.settings.cholesky_jitter(1e-1):
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        rbf=model.covar_module.base_kernel.kernels[0]
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            rbf.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # x_test = torch.from_numpy(np.hstack([x1o_flat.reshape(-1,1), x2o_flat.reshape(-1,1)])).float()
    trained_pred_dist = likelihood(model(x_test))
    observed_pred = likelihood(model(x_test))

    
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf_pred = ax.plot_surface(x1o, x2o, observed_pred.mean.numpy().reshape(x1o.shape), alpha=1)
# surf_origin = ax.plot_surface(x1o,x2o,yo, alpha=0.5)
# ax.legend(['Predicted','Original'])
# plt.show() 

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
fig.update_layout(legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.7)',
        bordercolor='black',
        borderwidth=1))
fig.show()

MSE = (gpytorch.metrics.mean_squared_error(trained_pred_dist,y_test,squared=True))
print(f'MSE: {MSE:.3f}')