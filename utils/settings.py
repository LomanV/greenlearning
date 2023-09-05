import torch
import os

def init():
    # Set device, GPU is highly recommended
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_net(train_forcing, eq):
    global G
    G     = torch.load("networks/" + eq + "/G_" + train_forcing +".pkl", map_location='cpu')

    u_file = "networks/" + eq + "/U_hom_" + train_forcing + ".pkl"
    if os.path.exists(u_file):
        global U_hom
        U_hom = torch.load(u_file, map_location='cpu')
