import torch
import torch.nn as nn

class GreenLossTime(nn.Module):

    def __init__(self):
        super(GreenLossTime, self).__init__()

    def forward(self, G_output, u, f):
        return self.loss_call(G_output, u, f)

    def loss_call(self, G_output, u, f):

        Nsample = 100
        Nx = 101
        Nt = 101

        f_weights = f.reshape(Nx, Nsample)
        pred = torch.tensordot(G_output.reshape((Nx, Nx, Nt)), f_weights, dims=([1], [0]))

        loss = torch.divide(torch.sum(torch.square(u-pred.reshape(Nx*Nt, Nsample)), axis=0), torch.sum(torch.square(u), axis=0))*100
        return torch.mean(loss)
