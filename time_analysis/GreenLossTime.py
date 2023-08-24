import torch
import torch.nn as nn

class GreenLossTime(nn.Module):

    def __init__(self):
        super(CustomLossTime2, self).__init__()

    def forward(self, G_output, u, f):
        return self.loss_call(G_output, u, f)

    def loss_call(self, G_output, u, f):

        Nsample = 100
        Nx = 101
        Nt = 101
        dom  = fx
        # For convenience we set time in such a way that is has the same domain
        # and discretisation as space, this is not general and should be upgraded
        time = fx

        # We do not use trapezoidal weights for the integral, as expected it still works 
        # but as G compensates for them it loses interpreatbility as a Green's function
        # Not having weights has a second advantage, only the number of points in dom and fx matter and not the value of their bounds
      
        f_weights = f_train.reshape(Nx, Nsample)
        pred = torch.tensordot(G_output.reshape((Nx, Nx, Nt)), f_weights, dims=([1], [0]))

        loss = torch.divide(torch.sum(torch.square(u-pred.reshape(Nx*Nt, Nsample)), axis=0), torch.sum(torch.square(u), axis=0))*100
        return torch.mean(loss)
