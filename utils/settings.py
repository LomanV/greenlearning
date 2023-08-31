import torch

def init():
    # Set device, GPU is highly recommended
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_net():
    global G
    global U_hom

    G     = torch.load("G.pkl")
    U_hom = torch.load("U_hom.pkl")
