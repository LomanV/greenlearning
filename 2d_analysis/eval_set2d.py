import torch

def eval_set2d(fx, device):

    # For convenience we work on square domains with the same number of discretisation points in both directions
    # This code is not yet compatible with a more general setting but it should not be hard to implement

    training_G = torch.stack(torch.meshgrid(fx.reshape(-1), \
                                            fx.reshape(-1), \
                                            fx.reshape(-1), \
                                            fx.reshape(-1)))

    training_G = training_G.permute(*torch.arange(training_G.ndim - 1, -1, -1))
    training_G = training_G.reshape(-1, 4)
    training_G = training_G.to(device)

    training_N = torch.stack(torch.meshgrid(fx.reshape(-1), fx.reshape(-1)))
    training_N = training_N.permute(*torch.arange(training_N.ndim - 1, -1, -1))
    training_N = training_N.reshape(-1, 2)
    training_N = training_N.to(device)

    return training_G, training_N
