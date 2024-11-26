
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import seaborn as sns

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





# def plot_dataframe_columns_combined(labels_data, prediction_data, save_dir=None):
#     """
#     Combines all column plots into a single figure with subplots for two datasets.

#     :param labels_data: The first dataset (NumPy array or DataFrame).
#     :param prediction_data: The second dataset (NumPy array or DataFrame).
#     :param save_dir: Directory to save the plot. If None, the plot is only displayed.
#     """
#     columns = ['Cd', 'Cl', 'p_probe_0', 'p_probe_1', 'p_probe_2', 'p_probe_3', 
#                'p_probe_4', 'p_probe_5', 'p_probe_6', 'p_probe_7', 
#                'p_probe_8', 'p_probe_9', 'p_probe_10', 'p_probe_11']

#     # Convert both datasets to pandas DataFrame for easy manipulation
#     df1 = pd.DataFrame(labels_data, columns=columns)
#     df2 = pd.DataFrame(prediction_data, columns=columns)

#     # Subset data starting from row 400
#     df1 = df1.iloc[400:]
#     df2 = df2.iloc[400:]
    
#     # Determine the number of rows and columns for the subplots grid
#     n_cols = 4
#     n_rows = (len(columns) + n_cols - 1) // n_cols  # Ensure enough rows for all columns

#     # Create a figure and subplots
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=False)
#     axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

#     for i, column in enumerate(columns):
#         ax = axes[i]
#         ax.plot(df1[column], label=f'{column} (labels)', linewidth=1.0)
#         ax.plot(df2[column], label=f'{column} (predictions)', linewidth=1.0, linestyle='--')
#         ax.set_title(f'{column}')
#         ax.legend()
#         ax.grid(True)

#     # Hide any unused subplots
#     for j in range(len(columns), len(axes)):
#         fig.delaxes(axes[j])

#     # Adjust layout for better spacing
#     plt.tight_layout()

#     # Save or show the plot
#     if save_dir:
#         save_path = f"{save_dir}/combined_plot.png"
#         plt.savefig(save_path, dpi=1800)
#         print(f"Combined plot saved to {save_path}")
#     else:
#         plt.show()





def plot_dataframe_columns_combined(labels_data, prediction_data, save_dir=None):
    """
    Combines all column plots into a single figure with subplots for two datasets,
    excluding 'Cd' and 'Cl' columns.

    :param labels_data: The first dataset (NumPy array or DataFrame).
    :param prediction_data: The second dataset (NumPy array or DataFrame).
    :param save_dir: Directory to save the plot. If None, the plot is only displayed.
    """
    # Define the columns to include, excluding 'Cd' and 'Cl'
    columns = ['Cd', 'Cl', 'p_probe_0', 'p_probe_1', 'p_probe_2', 'p_probe_3', 
               'p_probe_4', 'p_probe_5', 'p_probe_6', 'p_probe_7', 
               'p_probe_8', 'p_probe_9', 'p_probe_10', 'p_probe_11']

    # Convert both datasets to pandas DataFrame for easy manipulation
    df1 = pd.DataFrame(labels_data, columns=columns )  # Include all columns initially
    df2 = pd.DataFrame(prediction_data, columns=columns )

    # Subset data starting from row 400 and filter out 'Cd' and 'Cl'
    df1 = df1.iloc[400:][columns]
    df2 = df2.iloc[400:][columns]
    
    # Determine the number of rows and columns for the subplots grid
    n_cols = 4
    n_rows = (len(columns) + n_cols - 1) // n_cols  # Ensure enough rows for all columns

    # Create a figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=False)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

    for i, column in enumerate(columns):
        ax = axes[i]
        ax.plot(df1[column], label=f'{column} (labels)', linewidth=1.0)
        ax.plot(df2[column], label=f'{column} (predictions)', linewidth=1.0, linestyle='--')
        ax.set_title(f'{column}')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save or show the plot
    if save_dir:
        save_path = f"{save_dir}/combined_plot.png"
        plt.savefig(save_path, dpi=900)
        print(f"Combined plot saved to {save_path}")
    else:
        plt.show()






# def plot_dataframe_columns_heatmap(labels_data, prediction_data, save_dir=None):
#     """
#     Plots a heatmap for the differences between two datasets for all columns.

#     :param labels_data: The first dataset (NumPy array or DataFrame) - ground truth.
#     :param prediction_data: The second dataset (NumPy array or DataFrame) - predictions.
#     :param save_dir: Directory to save the heatmap. If None, the heatmap is only displayed.
#     """
#     columns = ['Cd', 'Cl', 'p_probe_0', 'p_probe_1', 'p_probe_2', 'p_probe_3', 
#                'p_probe_4', 'p_probe_5', 'p_probe_6', 'p_probe_7', 
#                'p_probe_8', 'p_probe_9', 'p_probe_10', 'p_probe_11']

#     # Convert both datasets to pandas DataFrame for easy manipulation
#     df1 = pd.DataFrame(labels_data, columns=columns)
#     df2 = pd.DataFrame(prediction_data, columns=columns)

#     # Subset data starting from row 400
#     df1 = df1.iloc[400:]
#     df2 = df2.iloc[400:]
    
#     # Compute the differences
#     differences = (df1 - df2).abs()
    
#     # Compute the mean absolute difference for each column
#     mean_differences = differences.mean().to_frame(name='Mean Absolute Difference')
    
#     # Plot the heatmap
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(mean_differences.T, annot=True, cmap='coolwarm', cbar=True)
#     plt.title('Heatmap of Mean Absolute Differences')
#     plt.xlabel('Columns')
#     plt.ylabel('Metric')
#     plt.tight_layout()

#     # Save or show the heatmap
#     if save_dir:
#         heatmap_path = f"{save_dir}/heatmap_differences.png"
#         plt.savefig(heatmap_path, dpi=900)
#         print(f"Heatmap saved to {heatmap_path}")
#     else:
#         plt.show()





def compute_scaled_l2_loss_heatmap(labels_data, prediction_data, save_dir=None):
    """
    Computes the L2 loss, scales it with the L2 norm of the original trajectory,
    and generates a heatmap.

    :param labels_data: The original trajectory (NumPy array or DataFrame).
    :param prediction_data: The predicted trajectory (NumPy array or DataFrame).
    :param save_dir: Directory to save the heatmap. If None, the heatmap is only displayed.
    """
    # Define the columns (exclude 'Cd' and 'Cl')
    columns = ['Cd', 'Cl', 'p_probe_0', 'p_probe_1', 'p_probe_2', 'p_probe_3', 
               'p_probe_4', 'p_probe_5', 'p_probe_6', 'p_probe_7', 
               'p_probe_8', 'p_probe_9', 'p_probe_10', 'p_probe_11']

    # Convert datasets to DataFrame
    df1 = pd.DataFrame(labels_data, columns=columns)
    df2 = pd.DataFrame(prediction_data, columns=columns)

    # Subset data starting from row 400
    df1 = df1.iloc[400:][columns]
    df2 = df2.iloc[400:][columns]

    # Compute L2 loss and L2 norm
    l2_loss = np.square(df1.values - df2.values).sum(axis=0)  # Sum of squared differences
    l2_norm = np.sqrt(np.square(df1.values).sum(axis=0))  # L2 norm of original trajectory
    scaled_loss = l2_loss / (l2_norm + 1e-8)  # Scale by L2 norm (avoid division by zero)

    # Generate heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap([scaled_loss], annot=True, xticklabels=columns, yticklabels=['Scaled L2 Loss'],
                cmap='viridis', cbar=True, fmt='.4f')
    plt.title('Heatmap of Scaled L2 Loss')
    plt.xlabel('Probes')
    plt.ylabel('Metric')

    # Save or show the heatmap
    if save_dir:
        save_path = f"{save_dir}/scaled_l2_loss_heatmap.png"
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

