import scipy
import scipy.io
import numpy as np
import torch

# Trapezoidal weights used for trapezoidal rule integration
def trapezoidal(x):
    """Trapezoidal weights for trapezoidal rule integration."""
    diff = np.diff(x, axis = 0)
    weights = np.zeros(x.shape)
    weights[1:-1] = diff[1:] + diff[:-1]
    weights[0] = diff[0]
    weights[-1] = diff[-1]
    weights = weights / 2
    return weights

def load_data(filename):
    data_idn = scipy.io.loadmat(filename)
    x = torch.tensor(data_idn['X']).float()
    y = torch.tensor(data_idn['Y']).float()
    u = torch.tensor(data_idn['U']).float()
    f = torch.tensor(data_idn['F']).float()

    Nu = x.shape[0]
    Nf = y.shape[0]
    xG = torch.repeat_interleave(x[:, 0].reshape(1, Nu), Nf, dim=0).reshape(Nu*Nf, 1)
    yG = torch.repeat_interleave(y[:, 0].reshape(Nf, 1), Nu, dim=1).reshape(Nu*Nf, 1)
    eval_G = torch.concat((xG,yG), 1).float()

    return x, y, u, f, eval_G

def load_data(filename, device, type):
    data_idn = scipy.io.loadmat(filename)

    u = torch.tensor(data_idn['U'], dtype=torch.float32)
    f = torch.tensor(data_idn['F'], dtype=torch.float32)

    u_train, u_test = u[:,0:100].to(device), u[:,100:200]
    f_train, f_test = f[:,0:100].to(device), f[:,100:200]

    if (type=='2d' or type=='time'):
        fx = torch.tensor(data_idn['fX'], dtype=torch.float32)
        return u_train, u_test, f_train, f_test, fx
    elif type=='1d':
        x = torch.tensor(data_idn['X']).float()
        y = torch.tensor(data_idn['Y']).float()
        return u_train, u_test, f_train, f_test, x, y
    else:
        return
