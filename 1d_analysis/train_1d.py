from ../utils/network import Net
from ../utils/utils import init_weights
from ../utils/utils import load_data
from GreenLoss1d import GreenLoss1d
from eval_set1d import eval_set1d

import torch
import numpy as np
from tqdm import tqdm, trange
import torch.optim as optim
import copy

n_epochs  = 10**4
G     = Net(2, 1, 4, 50)
U_hom = Net(1, 1, 4, 50)
loss_func = GreenLoss1d()

# combine the parameters of both networks into a single list
params = list(G.parameters()) + list(U_hom.parameters())

# define the Adam optimizer and pass the combined parameters to it
optimizer = optim.Adam(params, lr=0.001)

best_loss = 300

# Load the data
u_train, u_test, f_train, f_test, x, y = load_data(filename, 'cpu', '1d')

with trange(n_epochs, unit='epochs') as pbar:
    for epoch in pbar:

        G_output = G(eval_G)
        N_output = U_hom(x)

        optimizer.zero_grad()

        loss = loss_func.loss_call(G_output, N_output, u, f, x, y)
        loss.backward()
        optimizer.step()

        # Saving the best performing models
        if best_loss > loss.item():
            best_G = copy.deepcopy(G)
            best_U = copy.deepcopy(U_hom)
            torch.save(best_G, "/G.pkl")
            torch.save(best_U, "/U_hom.pkl")
            best_loss = loss.item()

        if (epoch + 1) % 10 == 0:
                pbar.set_postfix(loss=loss.item())
