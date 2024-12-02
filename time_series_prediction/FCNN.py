import torch as pt
import sys


class FCNNModel(pt.nn.Module):
    def __init__(
        self,
        n_outputs: int,
        n_layers: int = 3,
        n_neurons: int = 128,
        sequence_length: int = 5,
        n_features: int = 15,
        activation: callable = pt.nn.functional.leaky_relu,
    ):
        """
        implements a fully connected neural network

        :param n_inputs: input neurons, usually (n_probes + n_actions + n_cx + n_cy) * N time steps
        :param n_outputs: output neurons, usually (n_probes + n_cx + n_cy) * N time steps
        :param n_layers: number of hidden layers
        :param n_neurons: number of neurons per layer
        :param activation: activation function
        :return: None
        """
        super(FCNNModel, self).__init__()
        self.n_inputs = n_features * sequence_length
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.layers = pt.nn.ModuleList()

        # input layer to first hidden layer
        self.layers.append(pt.nn.Linear(self.n_inputs, self.n_neurons))
        self.layers.append(pt.nn.LayerNorm(self.n_neurons))

        # add more hidden layers if specified
        if self.n_layers > 1:
            for hidden in range(self.n_layers - 1):
                self.layers.append(pt.nn.Linear(self.n_neurons, self.n_neurons))
                self.layers.append(pt.nn.LayerNorm(self.n_neurons))

        # last hidden layer to output layer
        self.layers.append(pt.nn.Linear(self.n_neurons, self.n_outputs))

    def forward(self, x):
        for i_layer in range(len(self.layers) - 1):
            x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)
