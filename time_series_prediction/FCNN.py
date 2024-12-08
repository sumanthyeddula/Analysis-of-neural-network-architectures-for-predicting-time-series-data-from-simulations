import torch as pt


class FCNNModel(pt.nn.Module):
    def __init__(
        self,
        n_outputs: int,
        n_layers: int = 3,
        n_neurons: int = 128,
        sequence_length: int = 5,
        n_features: int = 15,
        activation: callable = pt.nn.functional.relu,
        dropout_rate: float = 0.5,  # Add dropout rate parameter
    ):
        """
        Implements a fully connected neural network with dropout.

        :param n_outputs: output neurons, usually (n_probes + n_cx + n_cy) * N time steps
        :param n_layers: number of hidden layers
        :param n_neurons: number of neurons per layer
        :param activation: activation function
        :param dropout_rate: dropout rate for regularization
        :return: None
        """
        super(FCNNModel, self).__init__()
        self.n_inputs = n_features * sequence_length
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layers = pt.nn.ModuleList()
        self.dropouts = pt.nn.ModuleList()  # Separate list for dropout layers

        # Input layer to first hidden layer
        self.layers.append(pt.nn.Linear(self.n_inputs, self.n_neurons))
        self.layers.append(pt.nn.LayerNorm(self.n_neurons))
        self.dropouts.append(pt.nn.Dropout(self.dropout_rate))

        # Add more hidden layers if specified
        if self.n_layers > 1:
            for hidden in range(self.n_layers - 1):
                self.layers.append(pt.nn.Linear(self.n_neurons, self.n_neurons))
                self.layers.append(pt.nn.LayerNorm(self.n_neurons))
                self.dropouts.append(pt.nn.Dropout(self.dropout_rate))

        # Last hidden layer to output layer (no dropout here)
        self.layers.append(pt.nn.Linear(self.n_neurons, self.n_outputs))

    def forward(self, x):
        for i_layer in range(len(self.layers) - 1):
            x = self.activation(self.layers[i_layer](x))
            if i_layer < len(self.dropouts):  # Apply dropout to hidden layers
                x = self.dropouts[i_layer](x)
        return self.layers[-1](x)
