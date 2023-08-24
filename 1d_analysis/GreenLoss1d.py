from ../utils/utils import trapezoidal

import torch
import torch.nn as nn

class GreenLoss1d(nn.Module):

    def __init__(self):
        super(GreenLoss1d, self).__init__()

    def forward(self, G_output, N_output, u, f):
        return self.loss_call(G_output, N_output, u, f)

    def loss_call(self, G_output, N_output, u, f, x, y):
        weights_x = torch.tensor(trapezoidal(x), dtype = torch.float32)
        weights_y = torch.tensor(trapezoidal(y), dtype = torch.float32)

        f_weights = torch.multiply(weights_y, f)

        Nu = x.shape[0]
        Nf = y.shape[0]

        loss_N = torch.repeat_interleave(N_output, torch.tensor([u.shape[1]]), dim=1)
        loss_G = torch.matmul(torch.transpose(torch.reshape(G_output, (Nf, -1)), 0, 1), f_weights)
        relative_error = torch.divide(torch.sum(torch.multiply(weights_x, torch.square(u - loss_G - loss_N)),0), torch.sum(torch.multiply(weights_x, torch.square(u[:,:])),0))
        loss = torch.mean(relative_error)
        return loss
