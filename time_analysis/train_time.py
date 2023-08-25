from ../utils/utils import Net
from ../utils/utils import init_weights
from GreenLossTime import GreenLossTime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange
import copy

n_epochs  = 10**4
G     = Net(3, 1, 8, 20).to(device)
G.apply(init_weights)
loss_func = GreenLossTime()

# define the Adam optimizer and pass the combined parameters to it
optimizer = optim.Adam(G.parameters(), lr=0.001)

history = np.zeros(n_epochs)
best_loss = 100
with trange(n_epochs, unit='epochs') as pbar:
    for epoch in pbar:

        G_output = G(training_G)

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
