import sys
sys.path.insert(1, '../utils')

from network import Net
from utils import load_data
import settings
from eval_set1d import eval_set1d
from predict_1d import predict_1d

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Getting the filename
eq       = "poisson"
forcing  = "gaus"
filename = "dataset/" + eq + "_" + forcing + ".mat"

# Loading the network data and computing the prediction
settings.load_net()
u_train, u_test, f_train, f_test, x, y = load_data(filename, "cpu", "1d")
integral = predict_1d(f_test, x, y)

idx = 0
plt.plot(x, u_test[:,idx],   label='true')
plt.plot(x, integral[:,idx], label='pred')
plt.legend(prop={'size': 20})
plt.savefig('cheb.pdf', bbox_inches='tight')
