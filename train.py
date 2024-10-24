import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_csv_files, stack_feature_label_pairs, create_tensor_dataset, normalize_data
from model_utils import FCNN
import random

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train_model(model, train_loader, val_loader, epochs, learning_rate, weight_decay, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # L2 regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                y_val_pred = model(X_val)
                val_loss += criterion(y_val_pred, y_val).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}")

        # Update the learning rate based on the validation loss
        scheduler.step(val_loss)

        # Print learning rate if it has changed
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        if epoch == 0 or current_lr != optimizer.param_groups[0]['lr']:
            print(f"Learning rate adjusted to: {current_lr:.6f}")
            optimizer.param_groups[0]['lr'] = current_lr
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses

def auto_regressive_predict(model: nn.Module, initial_sequence: np.ndarray, steps: int, true_sequence: np.ndarray, prob_schedule: float) -> np.ndarray:
    """
    Perform auto-regressive predictions using the model with scheduled sampling.
    
    Args:
        model (nn.Module): The trained neural network model.
        initial_sequence (np.ndarray): The initial input sequence (window size).
        steps (int): Number of steps to predict into the future.
        true_sequence (np.ndarray): The true values sequence used for scheduled sampling.
        prob_schedule (float): Probability of using the true value instead of the predicted value (scheduled sampling).
        
    Returns:
        np.ndarray: Array of predicted values.
    """
    model.eval()
    predictions = []
    current_sequence = initial_sequence.copy()
    
    with torch.no_grad():
        for step in range(steps):
            input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0)
            prediction = model(input_tensor).squeeze(0).numpy()
            
            # Extract the rotational speed from the last row of the current sequence
            rotational_speed = current_sequence[-1, 0]  # Assume the first column is rotational speed
            
            # Determine if we use the true value or the predicted value for Cd and Cl
            if np.random.rand() < prob_schedule:
                # Use the true value (scheduled sampling)
                true_cd = true_sequence[step, 0]
                true_cl = true_sequence[step, 1]
                next_step = np.array([rotational_speed, true_cd, true_cl])
            else:
                # Use the model's predicted value
                next_step = np.array([rotational_speed, prediction[0], prediction[1]])
            
            # Append the predicted Cd and Cl values for evaluation
            predictions.append(prediction)
            
            # Update the current sequence: drop the first entry and append the new step
            current_sequence = np.vstack((current_sequence[1:], next_step))
    
    return np.array(predictions)

def visualize_results(model: nn.Module, test_loader: DataLoader, window_size: int, steps: int, prob_schedule: float) -> None:
    """
    Visualize the results using auto-regression with scheduled sampling, including error heatmaps and variation plots.
    
    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        window_size (int): The window size used in the sliding window approach.
        steps (int): Number of steps to predict using auto-regression.
        prob_schedule (float): Probability of using the true value instead of the predicted value (scheduled sampling).
    """
    all_true_values = []
    all_predicted_values = []
    
    model.eval()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            initial_sequence = X_test[0].numpy()
            true_sequence = y_test.numpy()
            max_steps = min(len(true_sequence), steps)
            predicted_sequence = auto_regressive_predict(model, initial_sequence, max_steps, true_sequence, prob_schedule)
            all_true_values.append(true_sequence[:max_steps])
            all_predicted_values.append(predicted_sequence[:max_steps])
    
    true_values = np.vstack(all_true_values)
    predicted_values = np.vstack(all_predicted_values)
    
    # Extract Cd and Cl values for true and predicted
    true_values_cd = true_values[:, 0]
    predicted_values_cd = predicted_values[:, 0]
    true_values_cl = true_values[:, 1]
    predicted_values_cl = predicted_values[:, 1]

    # Plot for Cd predictions
    plt.figure(figsize=(10, 5))
    plt.plot(true_values_cd, label="True Cd", linewidth=2)
    plt.plot(predicted_values_cd, label="Predicted Cd", linestyle='dashed', linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Cd Value")
    plt.title("Cd Predictions vs. True Values")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot for Cl predictions
    plt.figure(figsize=(10, 5))
    plt.plot(true_values_cl, label="True Cl", linewidth=2)
    plt.plot(predicted_values_cl, label="Predicted Cl", linestyle='dashed', linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Cl Value")
    plt.title("Cl Predictions vs. True Values")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Error Heatmap Visualization
    error_values = np.abs(true_values - predicted_values)
    plt.figure(figsize=(10, 5))
    plt.imshow(error_values, cmap='viridis', aspect='auto')
    plt.colorbar(label="Error")
    plt.xlabel("Prediction Horizon")
    plt.ylabel("Sample Index")
    plt.title("Error Heatmap")
    plt.show()
    """
    # Variation plot for Cd: Difference between true and predicted values (Cd)
    variation_cd = true_values_cd - predicted_values_cd
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_values_cd, label="True Cd", color='green', linewidth=2)
    plt.plot(variation_cd, label="Variation in Cd (True - Predicted)", color='red', linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Cd Value / Difference")
    plt.title("Cd Variation Between True and Predicted Cd")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Variation plot for Cl: Difference between true and predicted values (Cl)
    variation_cl = true_values_cl - predicted_values_cl
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_values_cl, label="True Cl", color='green', linewidth=2)
    plt.plot(variation_cl, label="Variation in Cl (True - Predicted)", color='blue', linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel("Cl Value / Difference")
    plt.title("Cl Variation Between True and Predicted Cl")
    plt.grid(True)
    plt.legend()
    plt.show()

    """

def plot_loss_curve(train_losses: list, val_losses: list):
    """
    Plot the training and validation loss curves.
    
    Args:
        train_losses (list): Training loss history.
        val_losses (list): Validation loss history.
    """
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

def main():
    directory = r'D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations'
    file_pattern = '*.csv'
    
    dfs = load_csv_files(directory, file_pattern)
    if not dfs:
        print("No data files found. Exiting.")
        return

    features, labels = stack_feature_label_pairs(dfs)
    
    # Normalize the features and labels
    features_normalized, labels_normalized, feature_scaler, label_scaler = normalize_data(features, labels)
    
    window_size = 5
    dataset = create_tensor_dataset(features_normalized, labels_normalized, window_size) 
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    num_features = 3
    input_dim = window_size * num_features
    hidden_layers = [512, 128, 64, 32]
    output_dim = 2  # Predict Cd and Cl
    model = FCNN(input_dim, hidden_layers, output_dim)
    
    # Set the number of epochs and learning rate
    epochs = 100
    learning_rate = 0.0003
    weight_decay = 0.001

    # Train the model and get loss history
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs, learning_rate, weight_decay)
    
    # Plot the loss curves
    plot_loss_curve(train_losses, val_losses)
    
    # Evaluate and visualize results using auto-regression with scheduled sampling
    steps = 100  # Number of steps to predict into the future
    prob_schedule = 0.0  # Probability of using the true value for scheduled sampling

    visualize_results(model, test_loader, window_size, steps, prob_schedule)

if __name__ == "__main__":
    main()









