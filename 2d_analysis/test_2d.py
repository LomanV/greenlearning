import sys
sys.path.insert(1, '../utils')

from network import Net
from utils import load_data
import settings
from eval_set2d import eval_set2d
from predict_2d import predict_2d

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Getting the filename
eq       = "poisson"
forcing  = "cheb"
filename = "dataset/" + eq + "_2d_" + forcing + ".mat"

# Loading the network data and computing the prediction
settings.load_net()
settings.init()

# Loading the network data and computing the prediction
train_forcing = "gaus"
eq            = "poisson"
settings.load_net(train_forcing, eq)
settings.init()

forcings = ['sine', 'cheb', 'pwl', 'gaus']
NMSE = np.zeros(4)
NAVE = np.zeros(4)
Nf = 40
i  = 0
for forcing in forcings:

    # Getting the filename
    filename = "dataset/" + eq + "_2d_" + forcing + ".mat"

    u_train, u_test, f_train, f_test, fx = load_data(filename, settings.device, "2d")
    integral = predict_2d(f_test, fx)

    NMSEs   = torch.div(torch.sum((integral-u_test)**2, 0), torch.sum(u_test**2, 0))
    NMSE[i] = NMSEs.mean().item()
    NAVEs   = torch.div(torch.sum(torch.abs(integral-u_test), 0), torch.sum(torch.abs(u_test), 0))
    NAVE[i] = NAVEs.mean().item()
    i += 1

np.savetxt('NAVE_' + train_forcing + '.txt', NAVE, fmt='%.2e')
np.savetxt('NMSE_' + train_forcing + '.txt', NMSE, fmt='%.2e')
