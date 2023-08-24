import scipy
import torch

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
