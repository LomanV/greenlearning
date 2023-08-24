from ../utils/network import Net
from ../utils/utils import load_data
from GreenLoss1d import GreenLoss1d
from eval_set1d import eval_set1d

import torch

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

        G_output = G(training_G)
        N_output = U_hom(x)

        optimizer.zero_grad()

        loss = loss_func.loss_call(G_output, N_output, u, f)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
                pbar.set_postfix(loss=loss.item())

torch.save(G.state_dict(), 'G.pth')
torch.save(U_hom.state_dict(), 'U_hom.pth')
