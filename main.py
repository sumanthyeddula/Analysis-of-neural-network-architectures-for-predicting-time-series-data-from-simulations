import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from torch.autograd import Variable

# Load data
data = pd.read_csv('D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\cleaned_force_data.csv')
times = data['Time'].values  # Time values for reference
data = data[['Cd', 'Cl']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Create sequences and corresponding labels
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5
X, y = create_sequences(data_normalized, seq_length)
X = Variable(torch.Tensor(X))
y = Variable(torch.Tensor(y))

# Split data into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=2, num_layers=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(lstm_out[:, -1, :])
        return out

# Instantiate model
model = LSTMModel(input_dim=2, hidden_dim=50, batch_size=1, output_dim=2, num_layers=1)
criterion = torch.nn.MSELoss()  # Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    for i in range(len(X_train)):
        optimizer.zero_grad()
        outputs = model(X_train[i].unsqueeze(0))
        loss = criterion(outputs, y_train[i].unsqueeze(0))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")

# Prepare to make predictions for specific time steps
model.eval()
test_predictions = model(X_test)

# Invert predictions to original scale
predicted_cd_cl = scaler.inverse_transform(test_predictions.detach().numpy())

# Output predictions
for idx, (cd, cl) in enumerate(predicted_cd_cl):
    print(f"Time Step {times[len(X_train) + idx + seq_length]}: Cd = {cd}, Cl = {cl}")

