from ../utils/network import Net
from ../utils/utils import init_weights
from ../utils/utils import load_data
from GreenLoss2d import GreenLoss2d
from eval_set2d import eval_set2d

import torch
import numpy as np
from tqdm import tqdm, trange
import torch.optim as optim
import copy

# Set device, GPU is highly recommended
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialise networks
G     = Net(4, 1, 8, 20).to(device)
G.apply(init_weights)
U_hom = Net(2, 1, 8, 20).to(device)
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

u_train, u_test, f_train, f_test, fx = load_data(filename, device, '2d')
eval_G = eval_set2d(fx, device)

with trange(n_epochs, unit='epochs') as pbar:
    for epoch in pbar:

        G_output = G(training_G)
        N_output = U_hom(training_N)

        optimizer.zero_grad()

        loss = loss_func.loss_call(G_output, N_output, u_train, f_train)
        history[epoch] = loss.item()

        if best_loss > loss.item():
            torch.save(G.state_dict(), 'G.pth')
            torch.save(U_hom.state_dict(), 'U_hom.pth')
            best_loss = loss.item()

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            pbar.set_postfix(loss=loss.item())
