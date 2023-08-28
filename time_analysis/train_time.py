from ../utils/utils import Net
from ../utils/utils import init_weights
from ../utils/utils import load_data
from GreenLossTime import GreenLossTime
from eval_set_time import eval_set_time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange
import copy

# Set device, GPU is highly recommended
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialise network
G = Net(3, 1, 8, 20).to(device)
G.apply(init_weights)

# Define loss function
loss_func = GreenLossTime()

# define the Adam optimizer and pass the combined parameters to it
optimizer = optim.Adam(G.parameters(), lr=0.001)

# Save loss profile and best model
n_epochs  = 10**4
history = np.zeros(n_epochs)
best_loss = 100

# Load data
eq       = "heat" # choose from heat wave_x wave_t wave_x is wave with spatial boundary condition, wave_t time boundary conditions
forcing  = "cheb" # choose from cheb sine pwl gaus
filename = "dataset/" + eq + "_" + forcing + ".mat"

u_train, u_test, f_train, f_test, fx = load_data(filename, device, 'time')
eval_G = eval_set_time(fx, device)

with trange(n_epochs, unit='epochs') as pbar:
    for epoch in pbar:

        G_output = G(eval_G)

        optimizer.zero_grad()

        loss = loss_func.loss_call(G_output, u_train, f_train)
        history[epoch] = loss.item()

        if best_loss > loss.item():
            best_model = copy.deepcopy(G)
            torch.save(best_model, "/model.pkl")
            best_loss = loss.item()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            pbar.set_postfix(loss=loss.item())
