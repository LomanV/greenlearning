import sys
sys.path.insert(1, '../utils')

from utils import trapezoidal
import settings

import torch
import torch.nn as nn

class GreenLoss2d(nn.Module):

    def __init__(self):
        super(GreenLoss2d, self).__init__()

    def forward(self, G_output, N_output, u, f, fx):
        return self.loss_call(G_output, N_output, u, f, fx)

    def loss_call(self, G_output, N_output, u, f, fx):

        Nsample = 100
        # Nf and Nu are always equal with the data used, they might have to be changed to 40 depending on the grid size
        Nf = 40
        Nu = 40

        # The weights have shape [Nf*Nf, Nsample]
        weights_f = trapezoidal(fx.reshape(-1))
        weights_f = torch.tensor(weights_f, dtype = torch.float32)
        weights_f = torch.stack(torch.meshgrid(weights_f, weights_f)).T.reshape(-1, 2)
        weights_f = torch.prod(weights_f, dim=1)
        weights_f = torch.transpose(weights_f.repeat(Nsample, 1), 0, 1)
        weights_f = weights_f.to(settings.device)

        f_weights = weights_f*f

        weights_u = trapezoidal(fx.reshape(-1))
        weights_u = torch.tensor(weights_u, dtype = torch.float32)
        weights_u = torch.stack(torch.meshgrid(weights_u, weights_u)).T.reshape(-1, 2)
        weights_u = torch.prod(weights_u, dim=1)
        weights_u = torch.transpose(weights_u.repeat(Nsample, 1), 0, 1)
        weights_u = weights_u.to(settings.device)

        loss_N = torch.repeat_interleave(N_output, torch.tensor([u.shape[1]]).to(settings.device), dim=1)
        loss_G = torch.tensordot(G_output.reshape(Nu,Nu,Nf,Nf), f_weights.reshape(Nf,Nf, Nsample), dims=((2,3), (0,1))).reshape(Nu*Nu, Nsample)
        relative_error = torch.divide(torch.sum(torch.multiply(weights_u, torch.square(u[:,:] - loss_G - loss_N))), torch.sum(torch.multiply(weights_u, torch.square(u[:,:]))))
        loss = torch.mean(relative_error)
        return loss
