import torch

def init():
    # Set device, GPU is highly recommended
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_net(forcing):
    global G
    global U_hom

    G     = torch.load("networks/G_" + forcing + ".pkl")
    U_hom = torch.load("networks/U_hom_" + forcing + ".pkl")
