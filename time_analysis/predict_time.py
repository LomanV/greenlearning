import sys
sys.path.append('../utils')

from eval_set_time import eval_set_time
import settings

import torch

def predict_time(f, fx):
        """
        Computes the solution u from the estimated Green's function for a given
        set of forcings f in 1D
        The solution is time dependent hence in 2D
        """

        Nsample = 100
        Nx = 101
        Nt = 101

        # Be mindfull that in the implementation time and space have the same number of
        # discretisation points, the implementation will fail if not in this state
        # swapping Nx and Nt depending on the integration (dx or dt) will do.

        f_weights = f.reshape(Nx, Nsample)

        eval_G = eval_set_time(fx, 'cpu')
        G_output = settings.G(eval_G)
        pred = torch.tensordot(G_output.reshape((Nx, Nx, Nt)), f_weights, dims=([1], [0]))

        return pred

def predict_time_wave(f, g, fx):
        """
        Computes the solution u from the estimated Green's function for a given
        set of forcings f in 1D
        The solution is time dependent hence in 2D
        """

        Nsample = 100
        Nx = 101
        Nt = 101

        # Be mindfull that in the implementation time and space have the same number of
        # discretisation points, the implementation will fail if not in this state
        # swapping Nx and Nt depending on the integration (dx or dt) will do.

        f_weights = f.reshape(Nx, Nsample)
        g_weights = g.reshape(Nx, Nsample)

        eval_net = eval_set_time(fx, 'cpu')
        G_output = settings.G(eval_net)
        H_output = settings.H(eval_net)
        pred_1 = torch.tensordot(G_output.reshape((Nx, Nx, Nt)), f_weights, dims=([1], [0]))
        pred_2 = torch.tensordot(H_output.reshape((Nx, Nx, Nt)), g_weights, dims=([1], [0]))

        return pred_1 + pred_2
