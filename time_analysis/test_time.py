import sys
sys.path.insert(1, '../utils')

from network import Net
from utils import load_data, load_data_wave
import settings
from eval_set_time import eval_set_time
from predict_time import predict_time, predict_time_wave

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# Test settings
save_erros = True
save_plots = True

# Plot settings
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.set_cmap('jet')

# Settings for the test
save_errors = True
save_plots  = False
plot_kde    = False

# Getting the filename
eq       = "wave"
train_forcing  = "pwl"

# Loading the network data and computing the prediction
settings.init()
settings.load_net(train_forcing, eq)

forcings = ['sine', 'cheb', 'pwl', 'gaus']
NMSE = np.zeros(4)
NAVE = np.zeros(4)
Nf = 101
i  = 0
for forcing in forcings:

    # Getting the filename
    filename = "dataset/" + eq + "_" + forcing + ".mat"

    if eq == "heat":
        u_train, u_test, f_train, f_test, fx = load_data(filename, settings.device, "time")
        integral = predict_time(f_test, fx)
    elif eq == "wave":
        u_train, u_test, f_train, f_test, g_train, g_test, fx = load_data_wave(filename, "cpu")
        integral = predict_time_wave(f_test, g_test, fx)
    integral = integral.reshape((101*101, 100))

    NMSEs = torch.div(torch.sum((integral-u_test)**2, 0), torch.sum(u_test**2, 0))
    NMSE[i] = NMSEs.mean().item()
    NAVEs = torch.div(torch.sum(torch.abs(integral-u_test), 0), torch.sum(torch.abs(u_test), 0))
    NAVE[i] = NAVEs.mean().item()
    i += 1

    if save_plots:
        pred = integral[:,2].reshape((101,101)).detach().numpy()
        true = u_test[:,2].reshape((101,101)).detach().numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
        im1 = ax1.imshow(pred, extent=[0,1,0,1])
        ax1.set_title('Prediction')
        im2 = ax2.imshow(true, extent=[0,1,0,1])
        ax2.set_title('True solution')
        im3 = ax3.imshow(pred-true, extent=[0,1,0,1])
        ax3.set_title('Difference')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        plt.savefig(eq + "_" + forcing + ".pdf", bbox_inches='tight')
        plt.clf()

if save_errors:
    np.savetxt('NAVE_' + train_forcing + '.txt', NAVE, fmt='%.2e')
    np.savetxt('NMSE_' + train_forcing + '.txt', NMSE, fmt='%.2e')
