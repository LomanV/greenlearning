import torch

def eval_set_time(fx, device):

    # For time versus space location check the data formating
    # In our case time and space have the same number of discretisation points so this is not a problem

    training_G = torch.stack(torch.meshgrid(fx.reshape(-1)/2, \
                                            fx.reshape(-1), \
                                            fx.reshape(-1)))

    training_G = training_G.permute(*torch.arange(training_G.ndim - 1, -1, -1))
    training_G = training_G.reshape(-1, 3)
    training_G = training_G.to(device)

    return training_G
