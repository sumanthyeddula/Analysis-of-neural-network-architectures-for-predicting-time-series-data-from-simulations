
import matplotlib.pyplot as plt
import pandas as pd

def plot_loss(train_losses, val_losses, save_path):
    """
    Plots the training and validation losses on a log scale and saves the plot.
    
    :param train_losses: List of training losses over epochs.
    :param val_losses: List of validation losses over epochs.
    :param save_path: File path to save the plot (e.g., 'loss_plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training and Validation Loss Over Epochs (Log Scale)')
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend()
    plt.grid(True, which="both", linestyle='--')  # Grid lines for better visibility on log scale
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")


def plot_dataframe_columns(labels_data, prediction_data, save_dir=None):
    """
    Plots each column of the dataframe as a separate figure for two datasets.

    :param data1: The first dataset (NumPy array or DataFrame).
    :param data2: The second dataset (NumPy array or DataFrame).
    :param save_dir: Directory to save the plots. If None, plots are only displayed.
    """

    columns = ['Cd', 'Cl', 'p_probe_0', 'p_probe_1', 'p_probe_2', 'p_probe_3', 
               'p_probe_4', 'p_probe_5', 'p_probe_6', 'p_probe_7', 
               'p_probe_8', 'p_probe_9', 'p_probe_10', 'p_probe_11']

    # Convert both datasets to pandas DataFrame for easy manipulation
    df1 = pd.DataFrame(labels_data, columns=columns)
    df2 = pd.DataFrame(prediction_data, columns=columns)

    # Subset data starting from row 400
    df1 = df1.iloc[400:]
    df2 = df2.iloc[400:]
    
    for column in df1.columns:
        plt.figure(figsize=(10, 6))
        
        # Plot data1
        plt.plot(df1[column], label=f'{column} (labels)', linewidth=1.0)
        
        # Plot data2
        plt.plot(df2[column], label=f'{column} (predictions)', linewidth=1.0, linestyle='--')
        
        # Customize the plot
        plt.title(f'Column: {column}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save or show the plot
        if save_dir:
            plt.savefig(f"{save_dir}/{column}_plot.png", dpi=900)
            print(f"Plot for column '{column}' saved to {save_dir}/{column}_plot.png")