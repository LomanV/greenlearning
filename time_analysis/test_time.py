import sys
sys.path.insert(1, '../utils')

from network import Net
from utils import load_data
import settings
from eval_set_time import eval_set_time
from predict_time import predict_time

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Getting the filename
eq       = "heat"
forcing  = "gaus"
filename = "dataset/" + eq + "_" + forcing + ".mat"

# Loading the network data and computing the prediction
settings.load_net()
u_train, u_test, f_train, f_test, fx = load_data(filename, "cpu", "time")
integral = predict_time(f_test, x, y)
