import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size,
        num_layers,
        n_outputs,
        sequence_length=5,
        dropout=0.3,
    ):
        super(LSTMModel, self).__init__()

        self.n_features = n_features
        self.sequence_length = sequence_length
        self.n_outputs = n_outputs

        # LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, n_outputs)
        self.relu = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.sequence_length, self.n_features)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Last hidden state
        last_hidden_state = lstm_out[:, -1, :]

        # Fully connected layers with ReLU and dropout
        x = self.fc1(last_hidden_state)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output
