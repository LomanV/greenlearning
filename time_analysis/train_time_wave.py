import sys
sys.path.insert(1, '../utils')

from network import Net
from utils import init_weights
from utils import load_data_wave
import settings
from GreenLossTime import GreenLossTime2
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
H = Net(3, 1, 10, 28).to(settings.device)
G.apply(init_weights)
H.apply(init_weights)

# Define loss function
loss_func = GreenLossTime2()

# combine the parameters of both networks into a single list
params = list(G.parameters()) + list(H.parameters())

# define the Adam optimizer and pass the combined parameters to it
optimizer = optim.Adam(params, lr=0.001)

# Save loss profile and best model
n_epochs  = 10**5
history = np.zeros(n_epochs)
best_loss = 100

# Load data
eq       = "wave"
forcing  = "sine" # choose from cheb sine pwl gaus
filename = "dataset/" + eq + "_" + forcing + ".mat"

u_train, u_test, f_train, f_test, g_train, g_test, fx = load_data_wave(filename, settings.device)
eval_net = eval_set_time(fx, settings.device)

with trange(n_epochs, unit='epochs') as pbar:
    for epoch in pbar:

        G_output = G(eval_net)
        H_output = H(eval_net)

        optimizer.zero_grad()

        loss = loss_func.loss_call(G_output, H_output, u_train, f_train, g_train)
        history[epoch] = loss.item()

        if best_loss > loss.item():
            best_G = copy.deepcopy(G)
            best_H = copy.deepcopy(H)
            torch.save(best_G, "networks/" + eq + "/G_" + forcing + ".pkl")
            torch.save(best_H, "networks/" + eq + "/H_" + forcing + ".pkl")
            best_loss = loss.item()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            pbar.set_postfix(loss=loss.item())

plt.plot(np.log(history))
plt.savefig('loss_profiles/' + eq + '/loss_profile_' + forcing + '.pdf', bbox_inches='tight')
