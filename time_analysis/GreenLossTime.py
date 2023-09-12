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

class GreenLossTime2(nn.Module):

    def __init__(self):
        super(GreenLossTime2, self).__init__()

    def forward(self, G_output, H_output, u, f, g):
        return self.loss_call(G_output, H_output, u, f, g)

    def loss_call(self, G_output, H_output, u, f, g):

        Nsample = 100
        Nx = 101
        Nt = 101

        f_weights = f.reshape(Nx, Nsample)
        g_weights = g.reshape(Nx, Nsample)

        pred_1 = torch.tensordot(G_output.reshape((Nx, Nx, Nt)), f_weights, dims=([1], [0]))
        pred_1 = pred_1.reshape(Nx*Nt, Nsample)
        pred_2 = torch.tensordot(H_output.reshape((Nx, Nx, Nt)), g_weights, dims=([1], [0]))
        pred_2 = pred_2.reshape(Nx*Nt, Nsample)

        loss = torch.divide(torch.sum(torch.square(u-pred_1-pred_2), axis=0), torch.sum(torch.square(u), axis=0))*100
        return torch.mean(loss)
