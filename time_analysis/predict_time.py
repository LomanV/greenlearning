from eval_set_time import eval_set_time
import torch

def predict_time(f):
        """
        Computes the solution u from the estimated Green's function for a given
        set of forcings f in 1D
        The solution is time dependent hence in 2D
        """

        Nsample = 100
        Nx = 101
        Nt = 101
        dom  = fx

        # Be mindfull that in the implementation time and space have the same number of
        # discretisation points, the implementation will fail if not in this state
        # swapping Nx and Nt depending on the integration (dx or dt) will do.

        f_weights = f.reshape(Nx, Nsample)

        eval_G = eval_set_time(fx, 'cpu')
        G_output = G(eval_G)
        pred = torch.tensordot(G_output.reshape((Nx, Nx, Nt)), f_weights, dims=([1], [0]))

        return pred
