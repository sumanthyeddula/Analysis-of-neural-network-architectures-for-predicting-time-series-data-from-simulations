<<<<<<< HEAD
import torch as pt
import sys


class FCNNModel(pt.nn.Module):
    def __init__(self, n_outputs: int, n_layers: int = 3, n_neurons: int = 128, sequence_length: int = 5, n_features: int = 15,
                 activation: callable = pt.nn.functional.leaky_relu):
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

# Autoregressive Prediction Function
def autoregressive_predict(model, initial_input, n_steps,complete_inputdata ):
    """
    Perform autoregressive prediction.

    :param model: The trained FCNN model.
    :param initial_input: Initial input tensor of shape [1, 5, n_features].
    :param n_steps: Number of future time steps to predict.
    :param n_features: Number of features per time step.
    :return: A list of predictions for the specified number of steps.
    """
    predictions = []

    
    # Flatten initial input for the model
    #current_input = initial_input.view(1, -1)  # Shape [1, 5 * n_features]
    current_input = initial_input.reshape(1, -1)  # Shape [1, 5 * n_features]
    print("current_input shape:", current_input)

    for _ in range(n_steps):

        print("current_input shape:", current_input)
        # Predict the next time step
        next_pred = model(current_input)  # Shape [1, 14]
        
        
        # Append the prediction as a list of 14 features
        predictions.append(next_pred.squeeze().tolist())

        # Generate a random value and concatenate it to the prediction
        rotational_speed = complete_inputdata[0][0][_ + 5,- 1]
        print("rotational_speed shape:",rotational_speed.shape)
        print("rotational_speed:",rotational_speed)
        

        #random_value = pt.randn(1, 1)  # Shape [1, 1] for the random value
        rotational_speed = pt.tensor([rotational_speed], dtype=pt.float32).unsqueeze(0)
        print("rotational_speed shape:",rotational_speed.shape)
        print("rotational_speed:",rotational_speed)

        next_feature = pt.cat((next_pred, rotational_speed), dim=1)  # Shape [1, 15]

        # Reshape next_feature for concatenation in sliding window
        next_feature = next_feature.view(1, 1, -1)  # Shape [1, 1, 15]

        # Concatenate the new feature and remove the oldest time step
        initial_input = pt.cat((initial_input[:, 1:], next_feature), dim=1)  # Shape [1, 5, 15]
        
        
        # Flatten the updated sequence for the next prediction
        current_input = initial_input.view(1, -1)  # Shape [1, 5 * n_features]

    return predictions

if __name__ == "__main__":

    # Initialize the model
    n_features = 15    # Number of input features (example)
    n_outputs = 14   # Number of output features (example)
    n_layers = 3      # Number of hidden layers
    n_neurons = 10   # Neurons per hidden layer
    sequence_length = 5  # Sliding window of 5 time steps
    batch_size = 1
    n_steps = 3

    # Instantiate the model
    model = FCNNModel(n_outputs=n_outputs,   n_layers=n_layers, n_neurons=n_neurons,  sequence_length=sequence_length, n_features=n_features)


    from Data_pipeline import process_all_simulations


    main_path = r'D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\ResearchProject\Research_project\exercises'

    # Process all simulations and get list of feature-label pairs
    all_dataframes = process_all_simulations(main_path)
    print("all data frames:",all_dataframes[0][0].shape)
    

    input_1 = all_dataframes[0][0][0:5]
    print("input_1 shape:",input_1.shape)
    print("input_1:",input_1)

    input_1 = pt.tensor(input_1, dtype=pt.float32).view(1, 5, 15)
    print("input_1 shape:",input_1.shape)
    print("input_1:",input_1)
    
    # Prepare example input data
    #example_input = pt.randn(batch_size, sequence_length, n_features)  # Shape [1, 5, 15]
    
    # Perform autoregressive prediction
    predictions = autoregressive_predict(model, input_1, n_steps, all_dataframes)

    # Print the predictions
=======
import torch as pt
import sys


class FCNNModel(pt.nn.Module):
    def __init__(self, n_outputs: int, n_layers: int = 3, n_neurons: int = 128, sequence_length: int = 5, n_features: int = 15,
                 activation: callable = pt.nn.functional.leaky_relu):
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

# Autoregressive Prediction Function
def autoregressive_predict(model, initial_input, n_steps,complete_inputdata ):
    """
    Perform autoregressive prediction.

    :param model: The trained FCNN model.
    :param initial_input: Initial input tensor of shape [1, 5, n_features].
    :param n_steps: Number of future time steps to predict.
    :param n_features: Number of features per time step.
    :return: A list of predictions for the specified number of steps.
    """
    predictions = []

    
    # Flatten initial input for the model
    #current_input = initial_input.view(1, -1)  # Shape [1, 5 * n_features]
    current_input = initial_input.reshape(1, -1)  # Shape [1, 5 * n_features]
    print("current_input shape:", current_input)

    for _ in range(n_steps):

        print("current_input shape:", current_input)
        # Predict the next time step
        next_pred = model(current_input)  # Shape [1, 14]
        
        
        # Append the prediction as a list of 14 features
        predictions.append(next_pred.squeeze().tolist())

        # Generate a random value and concatenate it to the prediction
        rotational_speed = complete_inputdata[0][0][_ + 5,- 1]
        print("rotational_speed shape:",rotational_speed.shape)
        print("rotational_speed:",rotational_speed)
        

        #random_value = pt.randn(1, 1)  # Shape [1, 1] for the random value
        rotational_speed = pt.tensor([rotational_speed], dtype=pt.float32).unsqueeze(0)
        print("rotational_speed shape:",rotational_speed.shape)
        print("rotational_speed:",rotational_speed)

        next_feature = pt.cat((next_pred, rotational_speed), dim=1)  # Shape [1, 15]

        # Reshape next_feature for concatenation in sliding window
        next_feature = next_feature.view(1, 1, -1)  # Shape [1, 1, 15]

        # Concatenate the new feature and remove the oldest time step
        initial_input = pt.cat((initial_input[:, 1:], next_feature), dim=1)  # Shape [1, 5, 15]
        
        
        # Flatten the updated sequence for the next prediction
        current_input = initial_input.view(1, -1)  # Shape [1, 5 * n_features]

    return predictions

if __name__ == "__main__":

    # Initialize the model
    n_features = 15    # Number of input features (example)
    n_outputs = 14   # Number of output features (example)
    n_layers = 3      # Number of hidden layers
    n_neurons = 10   # Neurons per hidden layer
    sequence_length = 5  # Sliding window of 5 time steps
    batch_size = 1
    n_steps = 3

    # Instantiate the model
    model = FCNNModel(n_outputs=n_outputs,   n_layers=n_layers, n_neurons=n_neurons,  sequence_length=sequence_length, n_features=n_features)


    from Data_pipeline import process_all_simulations


    main_path = r'D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\ResearchProject\Research_project\exercises'

    # Process all simulations and get list of feature-label pairs
    all_dataframes = process_all_simulations(main_path)
    print("all data frames:",all_dataframes[0][0].shape)
    

    input_1 = all_dataframes[0][0][0:5]
    print("input_1 shape:",input_1.shape)
    print("input_1:",input_1)

    input_1 = pt.tensor(input_1, dtype=pt.float32).view(1, 5, 15)
    print("input_1 shape:",input_1.shape)
    print("input_1:",input_1)
    
    # Prepare example input data
    #example_input = pt.randn(batch_size, sequence_length, n_features)  # Shape [1, 5, 15]
    
    # Perform autoregressive prediction
    predictions = autoregressive_predict(model, input_1, n_steps, all_dataframes)

    # Print the predictions
>>>>>>> 10a8ac84e53d92c691fa9cc1d84ec25d63c6d803
    print("Autoregressive predictions:", predictions)