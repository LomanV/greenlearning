import torch
import torch.nn as nn

# Basic class for feedforward neural networks, used to learn the Green's function in 1d 2d and time dependent problems
# It is also used to learn the homogeneous solutions

class Net(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons):
        super(Net, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers

        self.activation = nn.PReLU()

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers-1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

# Xavier initialisation of the network's weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
