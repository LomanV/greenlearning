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
u_train, u_test, f_train, f_test, fx = load_data(filename, settings.device, "2d")
integral = predict_2d(f_test, fx)

Nf = 40
idx = 0
pred = integral[:, idx].reshape(Nf, Nf).detach().numpy()
true = u_test[:,idx].reshape(Nf, Nf).detach().numpy()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
im1 = ax1.imshow(pred, extent=[0,1,0,1])
ax1.set_title('Prediction')
im2 = ax2.imshow(true, extent=[0,1,0,1])
ax2.set_title('True solution')
im3 = ax3.imshow(pred-true, extent=[0,1,0,1])
ax3.set_title('Difference')
plt.colorbar(im1,fraction=0.046, pad=0.04)
plt.colorbar(im2,fraction=0.046, pad=0.04)
plt.colorbar(im3,fraction=0.046, pad=0.04)
plt.savefig('poisson2d.pdf', bbox_inches='tight')
