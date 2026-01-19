# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:35:53 2026

@author: mep24db
"""

import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.io as pio
import gpytorch
import torch 


# Sort out plotting 
pio.renderers.default = 'browser'


# Import data 
data = pd.read_csv('Sine10.25_0.1_5_data.csv')
locs = pd.read_csv('Accelerometer_Locations_shaker1.csv')


# Extract data
start = 30
end = 330
disp_21_TR = data.iloc[start:end,2].to_numpy().reshape(-1,1)
disp_21_C = data.iloc[start:end,3].to_numpy().reshape(-1,1)
disp_21_LE = data.iloc[start:end,4].to_numpy().reshape(-1,1)

Z = np.hstack((disp_21_LE,disp_21_C,disp_21_TR))
z = Z.ravel()
x_coords = locs.iloc[0,[5,10,15]].to_numpy(dtype=np.float64)
time = data.iloc[start:end,0].to_numpy()
Y, T = np.meshgrid(x_coords, time)

x_test = np.hstack((Y.ravel().reshape(-1,1),T.ravel().reshape(-1,1)))

# Make training data 
y_train = Y[0:100:10,:].ravel().reshape(-1,1)
time_train = T[0:100:10,:].ravel().reshape(-1,1)
z_train = Z[0:100:10,:].ravel()

x_train = np.hstack((y_train,time_train))


# # Plot to check 

# fig = go.Figure()
    
# fig.add_trace(
#     go.Surface(
#         x=T,
#         y=Y,
#         z=Z,
#         colorscale="Greys", 
#         opacity = 0.7,
#         showscale=False,
#         showlegend=False
#     )
# )

# fig.add_trace(
#     go.Scatter3d(
#         x = time_train, 
#         y = y_train, 
#         z = z_train, 
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

x_test_tensor = torch.from_numpy(x_test).float()


# Define model 
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
     

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# initialize likelihood and model 
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# likelihood.noise = torch.tensor(1e-4)
model = ExactGPModel(x_train_tensor, z_train_tensor, likelihood)


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
    loss = -mll(output, z_train_tensor) 
    loss.backward() 
    print('Iter %d/%d - Loss: %.3f  -  Noise: %.3f - Signal Variance: %.3f - Lengthscale: %.3f' % ( i + 1, training_iter, loss.item(),  model.likelihood.noise.item(), model.covar_module.outputscale.item(), model.covar_module.base_kernel.lengthscale.item()))
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
    nMSE = 100*(np.mean(((ypred-ytest)**2))/np.std(ytest))
    return nMSE
   
error = MSE(observed_pred.mean.numpy(),z) 
errorN = nMSE(observed_pred.mean.numpy(),z)

print(errorN)

# Plot results 
prediction = go.Surface(
    z=observed_pred.mean.numpy().reshape(T.shape),
    x=T, 
    y=Y, 
    colorscale="jet", 
    name='Prediction',
    opacity=0.9,
    showscale=False,
    showlegend=False
)

original = go.Surface(
    z=Z, 
    x=T, 
    y=Y, 
    colorscale='greys', 
    name='Target', 
    opacity=0.7,
    showscale=False,
    showlegend=False
)

training = go.Scatter3d(
    x=time_train.ravel(),
    y=y_train.ravel(), 
    z=z_train.ravel(), 
    mode='markers', 
    marker=dict(size=5, color='black', symbol='circle'),
    name='Training Data', 
    showlegend=False
)

fig = go.Figure(data=[prediction, original, training])
fig.update_layout(
    # title = f"NMSE = {errorN}",
    # legend=dict(x=0.45, y=0.01, bgcolor='rgba(255,255,255,0.7)',
    #     orientation="h",
    #     bordercolor='black',
    #     borderwidth=1,
    #     xanchor="center",
    #     yanchor="top",
    #     font=dict(size=40),
    #     itemsizing="constant"
    #     ),
    scene=dict(
        xaxis_title="x1",
        yaxis_title="x2",
        zaxis_title="y",  
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25),
            center=dict(x=0, y=0.2, z=0),
            up=dict(x=0, y=0, z=1)), 
        xaxis=dict(title=dict(font=dict(size=40)), showticklabels=False, autorange = "reversed"),
        yaxis=dict(title=dict(font=dict(size=40)), showticklabels=False),
        zaxis=dict(title=dict(font=dict(size=40)), showticklabels=False)
        )
)

fig.show()