from evalset_1d import eval_set1d
from ../utils/utils import trapezoidal

import torch

def predict_1d(f,x,y):

    eval_G = eval_set1d(x,y)
    G_output = G(eval_G)

    w = torch.tensor(trapezoidal(y), dtype = torch.float32)
    f_weights = torch.multiply(w, f)

    integral = torch.matmul(torch.reshape(G_output, (Nu, Nf)), f_weights[:,:])
    return integral.detach().numpy()
