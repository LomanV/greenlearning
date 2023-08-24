from ../utils/utils import trapezoidal
from eval_set2d import eval_set2d

import torch
import numpy as np

def predict2d(f):
    """
    Computes the solution u from the estimated Green's function for a given set of forcings f in 2d
    Makes direct use of Fubini's theorem
    """

    Nf      = int(np.sqrt(f.shape[0]))
    fx      = np.linspace(0,1,Nf)
    Nsample = 100

    # Computing the weights for the trapezoidal integral G(x,y,s,t)*f(s,t)
    # This step is actually not necessary, as long as you stay consistent with predict_2d
    # Skipping it will change the values of G and take away the interpretability of results as a Green's function
  
    weights_f = trapezoidal(fx.reshape(-1))
    weights_f = torch.tensor(weights_f, dtype = torch.float32)
    weights_f = torch.stack(torch.meshgrid(weights_f, weights_f)).T.reshape(-1, 2)
    weights_f = torch.prod(weights_f, dim=1)
    weights_f = torch.transpose(weights_f.repeat(Nsample, 1), 0, 1)
    weights_f = weights_f

    # Getting the evaluation points for G to perform the convolution
    eval_G, eval_N = eval_set2d(fx, 'cpu')

    # Performing the convolution
    f_weights = weights_f*f
    G_output  = G(eval_G)
    integral  = torch.matmul(torch.transpose(torch.reshape(G_output, (Nf*Nf, -1)), 0, 1), f_weights[:,:])
  
    return integral + U_hom(eval_N)
    return integral + U_hom(eval_N)
