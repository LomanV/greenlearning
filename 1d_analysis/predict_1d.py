import sys
sys.path.insert(1, '../utils')

import settings

import torch
import numpy as np
from scipy import integrate

def predict_1d(f, x, y):
    """
    Computes the solution u from the estimated Green's function for a given forcing f
    """
    u_pred = np.zeros((100, 100))
    for i in range(100):
        x_rep = x[i].repeat(200, 1)
        z = torch.cat((x_rep, y), dim=1)
        prod = settings.G(z)*f[:, :].reshape(200, 100)
        integral = integrate.trapezoid(prod.detach().numpy(), y.detach().numpy(), axis=0)
        u_pred[i, :] = integral + settings.U_hom(x[i]).detach().numpy().repeat(100)

    return u_pred
