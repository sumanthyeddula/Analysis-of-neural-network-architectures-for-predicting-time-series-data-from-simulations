import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data_path = 'D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations/rotational_speed.csv'
df = pd.read_csv(data_path)

# Strip column names of extra spaces
df.columns = df.columns.str.strip()

# Data Preprocessing
input_columns = ['Rotational Speed (units)', 'Cd', 'Cl']
output_columns = ['Cd', 'Cl']

# Scale the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Create sliding windows
window_size = 5

def create_windows(df, window_size):
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[input_columns].iloc[i:i + window_size].values)
        y.append(df[output_columns].iloc[i + window_size].values)
    return np.array(X), np.array(y)

X, y = create_windows(df_scaled, window_size)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(window_size * len(input_columns), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Predicting cd and cl
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = FullyConnectedNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    train_loss = criterion(y_pred, y_train)
    train_loss.backward()
    optimizer.step()
    
    # Testing
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, y_test)
    
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Plot loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Over Epochs')
plt.legend()
plt.show()

# Visualize predictions vs actual
y_test_pred = y_test_pred.detach().numpy()
y_test = y_test.detach().numpy()

# Plot CD predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test[:, 0], label='Actual CD', color='blue')
plt.plot(y_test_pred[:, 0], label='Predicted CD', color='cyan')
plt.xlabel('Test Sample Index')
plt.ylabel('CD Value')
plt.title('Predicted vs Actual CD')
plt.legend()
plt.show()

# Plot CL predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test[:, 1], label='Actual CL', color='red')
plt.plot(y_test_pred[:, 1], label='Predicted CL', color='orange')
plt.xlabel('Test Sample Index')
plt.ylabel('CL Value')
plt.title('Predicted vs Actual CL')
plt.legend()
plt.show()

# How to add more data in the future:
# 1. Add the new CSV file to the same directory.
# 2. Load it using `pd.read_csv(new_file_path)` and follow the same scaling steps.
# 3. Concatenate the new data to the existing dataset:
#    df = pd.concat([df, new_df], axis=0).reset_index(drop=True)
# 4. Re-run the sliding window creation, scaling, and model training.

























