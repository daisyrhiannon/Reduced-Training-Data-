# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 15:45:06 2026

@author: mep24db
"""


import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.io as pio
import gpytorch
import torch 
from gpytorch.constraints import Interval


# Sort out plotting 
pio.renderers.default = 'browser'


# Import data 
data = pd.read_csv('Sine10.25_0.1_5_data.csv')
locs = pd.read_csv('Accelerometer_Locations_shaker1.csv')


# Extract data
start = 30
end = 1000
z = data.iloc[start:end,3].to_numpy().reshape(-1,1)

time = data.iloc[start:end,0].to_numpy()


# # Make training data 
# y_train = Y[0:500:10,:].ravel().reshape(-1,1)
# time_train = T[0:500:10,:].ravel().reshape(-1,1)
# z_train = Z[0:500:10,:].ravel()

# x_train = np.hstack((y_train,time_train))

# # Make training data 
# # For random data within the strip 

strip_width = 20
time_strip = time[:10*strip_width].ravel().reshape(-1, 1)
z_strip  = z[:10*strip_width].ravel().reshape(-1, 1)

# For random data within the strip 
n_train = strip_width*5
rng = np.random.default_rng(222)
random_indices = rng.choice(strip_width*10, n_train, replace = False)


time_train = time_strip[random_indices]
z_train_1 = z_strip[random_indices]

 
x_train = np.hstack([time_train])
z_train = z_train_1.ravel() # + 0.01*np.random.randn(y_train.size)

# Plot to check 

# fig = go.Figure()
    
# fig.add_trace(
#     go.Scatter(
#         x=time,
#         y=z.ravel(),
#         mode = "lines"
#     )
# )

# fig.add_trace(
#     go.Scatter(
#         x = time_train.ravel(), 
#         y = z_train, 
#         mode = 'markers',
#         marker=dict(size=5)
#     )
# )

# fig.update_layout(
#     scene=dict(
#         xaxis_title="x1",
#         yaxis_title="x2",
#         zaxis_title="y",  
#         camera=dict(
#             eye=dict(x=1.25, y=1.25, z=1.25),
#             center=dict(x=0, y=0.2, z=0),
#             up=dict(x=0, y=0, z=1)), 
#         xaxis=dict(title=dict(font=dict(size=40)), showticklabels=False, autorange = "reversed"),
#         yaxis=dict(title=dict(font=dict(size=40)), showticklabels=False),
#         zaxis=dict(title=dict(font=dict(size=40)), showticklabels=False)
#         )
#     )


# fig.show()

x_train_tensor = torch.from_numpy(x_train).float()
z_train_tensor = torch.from_numpy(z_train).float()

x_test_tensor = torch.from_numpy(time).float()


# Define model 
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        period = gpytorch.kernels.PeriodicKernel()
        # rbf = gpytorch.kernels.RBFKernel(active_dims = [1])
        # product = rbf*period
        self.covar_module = gpytorch.kernels.ScaleKernel(period)
     

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# initialize likelihood and model 
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = torch.tensor(1e-4)
model = ExactGPModel(x_train_tensor, z_train_tensor, likelihood)


# set hyperparameters and bounds
period = model.covar_module.base_kernel
# rbf = product.kernels[0]   
# period = product.kernels[1] 

f = 10.25355
period.period_length = torch.tensor(1 / f).float() # For fixed period

# # period.lengthscale = torch.tensor(1).float() # for fixed lengthscales 
# lower2 = torch.tensor(1)
# upper2 = torch.tensor(3)
# startpoint2 = lower2 + ((upper2-lower2))* torch.rand(1)
# period.initialize(lengthscale=startpoint2)

# # rbf.lengthscale = torch.tensor(0.2).float() # For fixed lengthscales 
# lower = torch.tensor(0.1)
# upper = torch.tensor(1)
# startpoint = lower+(upper-lower)*torch.rand(1)
# rbf.raw_lengthscale_constraint = Interval(lower, upper)
# rbf.initialize(lengthscale=startpoint)


# # model.covar_module.outputscale = torch.tensor(np.var(y)).float() # For fixed variance
# vary = np.var(z)
# model.covar_module.raw_outputscale_constraint = Interval(vary*0.9, vary*1.1)
# startpoint_var = (vary*0.9) + ((vary*1.1)-(vary*0.9)) * torch.rand_like(torch.tensor(vary*0.9))
# model.covar_module.initialize(outputscale =startpoint_var)

# model.mean_module.constant = torch.tensor(np.mean(z)).float()


# Fix some hyperparameters 
period.raw_period_length.requires_grad_(False)

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


with gpytorch.settings.cholesky_jitter(1e-1): 
    for i in range(training_iter): 
        # Zero gradients from previous iteration 
        optimizer.zero_grad() 
        # Output from model 
        output = model(x_train_tensor) 
        # Calc loss and backprop gradients 
        loss = -mll(output, z_train_tensor) 
        loss.backward() 
        # ls = rbf.lengthscale.detach().cpu().numpy()
        print('Iter %d/%d - Loss: %.3f  -  Noise: %.3f - Signal Variance: %.3f - Period: %.3f  - Period Lengthscale: %3f'  % ( i + 1, training_iter, loss.item(),  model.likelihood.noise.item(), model.covar_module.outputscale.item(), period.period_length.item(), period.lengthscale.item()))
        optimizer.step()
        
    
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #trained_pred_dist = likelihood(model(x_train_scaled_tensor))
    observed_pred = likelihood(model(x_test_tensor))
    
def MSE(ypred,ytest):
    MSE = np.mean(((ypred-ytest)**2))
    return MSE

def nMSE(ypred,ytest):
    nMSE = 100*(np.mean(((ypred-ytest)**2))/np.var(ytest))
    return nMSE
   
error = MSE(observed_pred.mean.numpy(),z) 
errorN = nMSE(observed_pred.mean.numpy(),z)

print(f"NMSE = {errorN}")

# Plot results 
prediction = go.Scatter(
    y=observed_pred.mean.numpy(),
    x=time.ravel(), 
    mode = "lines"
)

original = go.Scatter(
    x = time.ravel(), 
    y = z.ravel(), 
    mode = "lines"
)

training = go.Scatter(
    x=time_train.ravel(),
    y=z_train,
    mode='markers', 
    marker=dict(size=5, color='black', symbol='circle'),
    name='Training Data', 
    showlegend=False
)

fig = go.Figure(data=[prediction, original, training])
# fig.update_layout(
#     # title = f"NMSE = {errorN}",
#     # legend=dict(x=0.45, y=0.01, bgcolor='rgba(255,255,255,0.7)',
#     #     orientation="h",
#     #     bordercolor='black',
#     #     borderwidth=1,
#     #     xanchor="center",
#     #     yanchor="top",
#     #     font=dict(size=40),
#     #     itemsizing="constant"
#     #     ),
#     # scene=dict(
#     #     xaxis_title="x1",
#     #     yaxis_title="x2",
#     #     zaxis_title="y",  
#     #     camera=dict(
#     #         eye=dict(x=1.25, y=1.25, z=1.25),
#     #         center=dict(x=0, y=0.2, z=0),
#     #         up=dict(x=0, y=0, z=1)), 
#     #     xaxis=dict(title=dict(font=dict(size=40)), showticklabels=False, autorange = "reversed"),
#     #     yaxis=dict(title=dict(font=dict(size=40)), showticklabels=False),
#     #     zaxis=dict(title=dict(font=dict(size=40)), showticklabels=False)
#         )
# )

fig.show()