import sys
sys.path.insert(1, '../utils')

from network import Net
from utils import init_weights
from utils import load_data
import settings
from GreenLossTime import GreenLossTime
from eval_set_time import eval_set_time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange
import copy

import matplotlib
import matplotlib.pyplot as plt

# Plot settings
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.set_cmap('jet')

settings.init()

# Initialise network
G = Net(3, 1, 10, 28).to(settings.device)
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
forcing  = "sine" # choose from cheb sine pwl gaus
filename = "dataset/" + eq + "_" + forcing + ".mat"

u_train, u_test, f_train, f_test, fx = load_data(filename, settings.device, 'time')
eval_G = eval_set_time(fx, settings.device)

with trange(n_epochs, unit='epochs') as pbar:
    for epoch in pbar:

        G_output = G(eval_G)

        optimizer.zero_grad()

        loss = loss_func.loss_call(G_output, u_train, f_train)
        history[epoch] = loss.item()

        if best_loss > loss.item():
            best_model = copy.deepcopy(G)
            torch.save(best_model, "networks/" + eq + "/G_" + forcing + ".pkl")
            best_loss = loss.item()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            pbar.set_postfix(loss=loss.item())

plt.plot(np.log(history))
plt.savefig('loss_profiles/' + eq + '/loss_profile_' + forcing + '.pdf', bbox_inches='tight')
