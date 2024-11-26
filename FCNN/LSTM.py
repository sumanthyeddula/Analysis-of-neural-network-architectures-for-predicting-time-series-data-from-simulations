import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, n_outputs, sequence_length: int = 5):
        """
        LSTM model to predict the next step based on 5 previous steps.

        Args:
            n_features (int): Number of features per time step.
            hidden_size (int): Number of neurons in the LSTM's hidden state.
            num_layers (int): Number of stacked LSTM layers.
            n_outputs (int): Number of outputs to predict.
            sequence_length (int): Length of the input sequence (default is 5).
        """
        super(LSTMModel, self).__init__()
        
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.n_outputs = n_outputs
        
        # Define LSTM
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer to map hidden state to the output
        self.fc = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length * n_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_outputs].
        """
        # Reshape input to [batch_size, sequence_length, n_features]
        batch_size = x.size(0)
        x = x.view(batch_size, self.sequence_length, self.n_features)  # Reshape for LSTM
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, sequence_length, hidden_size]
        
        # Take the output of the last time step
        last_hidden_state = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Pass through fully connected layer to predict the next step
        output = self.fc(last_hidden_state)  # [batch_size, n_outputs]
        return output
