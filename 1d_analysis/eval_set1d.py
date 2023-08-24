import torch

def eval_set1d(x, y):

    Nu = x.shape[0]
    Nf = y.shape[0]
    xG = torch.repeat_interleave(x[:, 0].reshape(1, Nu), Nf, dim=0).reshape(Nu*Nf, 1)
    yG = torch.repeat_interleave(y[:, 0].reshape(Nf, 1), Nu, dim=1).reshape(Nu*Nf, 1)
    eval_G = torch.concat((xG,yG), 1).float()
    return eval_G
