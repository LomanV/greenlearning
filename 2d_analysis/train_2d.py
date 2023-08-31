import sys
sys.path.insert(1, '../utils')

from network import Net
from utils import init_weights
from utils import load_data
import settings
from GreenLoss2d import GreenLoss2d
from eval_set2d import eval_set2d

import torch
import numpy as np
from tqdm import tqdm, trange
import torch.optim as optim
import copy

settings.init()

# initialise networks
G     = Net(4, 1, 16, 32).to(settings.device)
U_hom = Net(2, 1, 16, 32).to(settings.device)
G.apply(init_weights)
U_hom.apply(init_weights)

# Define loss function
loss_func = GreenLoss2d()

# combine the parameters of both networks into a single list
params = list(G.parameters()) + list(U_hom.parameters())

# define the Adam optimizer and pass the combined parameters to it
optimizer = optim.Adam(params, lr=0.001)

# Save loss profile and best model
n_epochs  = 10**4 # might need up to 10**5 for biharmonic
history = np.zeros(n_epochs)
best_loss = 100

# Load data
eq       = "poisson" # choose from poisson biharmonic
forcing  = "cheb" # choose from cheb sine pwl gaus
filename = "dataset/" + eq + "_2d_" + forcing + ".mat"

u_train, u_test, f_train, f_test, fx = load_data(filename, settings.device, '2d')
eval_G, eval_N = eval_set2d(fx, settings.device)

with trange(n_epochs, unit='epochs') as pbar:
    for epoch in pbar:

        G_output = G(eval_G)
        N_output = U_hom(eval_N)

        optimizer.zero_grad()

        loss = loss_func.loss_call(G_output, N_output, u_train, f_train, fx)
        history[epoch] = loss.item()

        # Saving the best performing models
        if best_loss > loss.item():
            best_G = copy.deepcopy(G)
            best_U = copy.deepcopy(U_hom)
            torch.save(best_G, "G.pkl")
            torch.save(best_U, "U_hom.pkl")
            best_loss = loss.item()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            pbar.set_postfix(loss=loss.item())
