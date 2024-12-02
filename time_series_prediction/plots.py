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
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Training and Validation Loss Over Epochs (Log Scale)")
    plt.yscale("log")  # Set y-axis to log scale
    plt.legend()
    plt.grid(
        True, which="both", linestyle="--"
    )  # Grid lines for better visibility on log scale
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")


def plot_dataframe_columns(labels_data, prediction_data, save_dir=None):
    """
    Plots each column of the dataframe as a separate figure for two datasets.

    :param data1: The first dataset (NumPy array or DataFrame).
    :param data2: The second dataset (NumPy array or DataFrame).
    :param save_dir: Directory to save the plots. If None, plots are only displayed.
    """

    columns = [
        "Cd",
        "Cl",
        "p_probe_0",
        "p_probe_1",
        "p_probe_2",
        "p_probe_3",
        "p_probe_4",
        "p_probe_5",
        "p_probe_6",
        "p_probe_7",
        "p_probe_8",
        "p_probe_9",
        "p_probe_10",
        "p_probe_11",
    ]

    # Convert both datasets to pandas DataFrame for easy manipulation
    df1 = pd.DataFrame(labels_data, columns=columns)
    df2 = pd.DataFrame(prediction_data, columns=columns)

    # Subset data starting from row 400
    df1 = df1.iloc[400:]
    df2 = df2.iloc[400:]

    for column in df1.columns:
        plt.figure(figsize=(10, 6))

        # Plot data1
        plt.plot(df1[column], label=f"{column} (labels)", linewidth=1.0)

        # Plot data2
        plt.plot(
            df2[column], label=f"{column} (predictions)", linewidth=1.0, linestyle="--"
        )

        # Customize the plot
        plt.title(f"Column: {column}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # Save or show the plot
        if save_dir:
            plt.savefig(f"{save_dir}/{column}_plot.png", dpi=900)
            print(f"Plot for column '{column}' saved to {save_dir}/{column}_plot.png")


def plot_dataframe_columns_combined(labels_data, prediction_data, save_dir=None):
    """
    Plots 'Cd' and 'Cl' columns as side-by-side subplots and combines the rest of the columns into a separate grid.

    :param labels_data: The first dataset (NumPy array or DataFrame).
    :param prediction_data: The second dataset (NumPy array or DataFrame).
    :param save_dir: Directory to save the plot. If None, the plot is only displayed.
    """

    # Define the columns to include
    columns = [
        "Cd",
        "Cl",
        "p_probe_0",
        "p_probe_1",
        "p_probe_2",
        "p_probe_3",
        "p_probe_4",
        "p_probe_5",
        "p_probe_6",
        "p_probe_7",
        "p_probe_8",
        "p_probe_9",
        "p_probe_10",
        "p_probe_11",
    ]

    # Convert both datasets to pandas DataFrame
    df1 = pd.DataFrame(labels_data, columns=columns)
    df2 = pd.DataFrame(prediction_data, columns=columns)

    # Separate the 'Cd' and 'Cl' columns from the rest
    cd_cl_columns = ["Cd", "Cl"]
    probe_columns = [col for col in columns if col not in cd_cl_columns]

    # Plot 'Cd' and 'Cl' columns side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    for i, column in enumerate(cd_cl_columns):
        ax = axes[i]
        ax.plot(df1[column], label=f"{column} (labels)", linewidth=1.0)
        ax.plot(
            df2[column], label=f"{column} (predictions)", linewidth=1.0, linestyle="--"
        )
        ax.set_title(f"{column}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    # Save or show the Cd and Cl plot
    if save_dir:
        save_path = f"{save_dir}/cd_cl_plot.png"
        plt.savefig(save_path, dpi=900)
        print(f"'Cd' and 'Cl' plot saved to {save_path}")
    else:
        plt.show()

    # Plot the rest of the columns in a grid
    n_cols = 4
    n_rows = (len(probe_columns) + n_cols - 1) // n_cols  # Calculate rows for subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=False
    )
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

    for i, column in enumerate(probe_columns):
        ax = axes[i]
        ax.plot(df1[column], label=f"{column} (labels)", linewidth=1.0)
        ax.plot(
            df2[column], label=f"{column} (predictions)", linewidth=1.0, linestyle="--"
        )
        ax.set_title(f"{column}")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(probe_columns), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save or show the combined probe plot
    if save_dir:
        save_path = f"{save_dir}/probe_columns_plot.png"
        plt.savefig(save_path, dpi=900)
        print(f"Probe columns plot saved to {save_path}")
    else:
        plt.show()


def compute_scaled_l2_loss_heatmap(labels_data, prediction_data, save_dir=None):
    """
    Computes the L2 loss, scales it with the L2 norm of the original trajectory,
    and generates a heatmap.

    :param labels_data: The original trajectory (NumPy array or DataFrame).
    :param prediction_data: The predicted trajectory (NumPy array or DataFrame).
    :param save_dir: Directory to save the heatmap. If None, the heatmap is only displayed.
    """
    # Define the columns (exclude 'Cd' and 'Cl')
    columns = [
        "Cd",
        "Cl",
        "p_probe_0",
        "p_probe_1",
        "p_probe_2",
        "p_probe_3",
        "p_probe_4",
        "p_probe_5",
        "p_probe_6",
        "p_probe_7",
        "p_probe_8",
        "p_probe_9",
        "p_probe_10",
        "p_probe_11",
    ]

    # Convert datasets to DataFrame
    df1 = pd.DataFrame(labels_data, columns=columns)
    df2 = pd.DataFrame(prediction_data, columns=columns)

    # # Subset data starting from row 400
    # df1 = df1.iloc[400:][columns]
    # df2 = df2.iloc[400:][columns]

    # Compute L2 loss and L2 norm
    l2_loss = np.square(df1.values - df2.values).sum(
        axis=0
    )  # Sum of squared differences
    l2_norm = np.sqrt(
        np.square(df1.values).sum(axis=0)
    )  # L2 norm of original trajectory
    scaled_loss = l2_loss / (
        l2_norm + 1e-8
    )  # Scale by L2 norm (avoid division by zero)

    # Generate heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        [scaled_loss],
        annot=True,
        xticklabels=columns,
        yticklabels=["Scaled L2 Loss"],
        cmap="viridis",
        cbar=True,
        fmt=".4f",
    )
    plt.title("Heatmap of Scaled L2 Loss")
    plt.xlabel("Probes")
    plt.ylabel("Metric")

    # Save or show the heatmap
    if save_dir:
        save_path = f"{save_dir}/scaled_l2_loss_heatmap.png"
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()


def compute_scaled_l2_loss_scatter(labels_data, prediction_data, save_dir=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Define the columns (exclude 'Cd' and 'Cl')
    columns = [
        "Cd",
        "Cl",
        "p_probe_0",
        "p_probe_1",
        "p_probe_2",
        "p_probe_3",
        "p_probe_4",
        "p_probe_5",
        "p_probe_6",
        "p_probe_7",
        "p_probe_8",
        "p_probe_9",
        "p_probe_10",
        "p_probe_11",
    ]

    # Convert datasets to DataFrame
    df1 = pd.DataFrame(labels_data, columns=columns)
    df2 = pd.DataFrame(prediction_data, columns=columns)

    # Compute L2 loss and L2 norm
    l2_loss = np.square(df1.values - df2.values).sum(
        axis=0
    )  # Sum of squared differences
    l2_norm = np.sqrt(
        np.square(df1.values).sum(axis=0)
    )  # L2 norm of original trajectory
    scaled_loss = l2_loss / (
        l2_norm + 1e-8
    )  # Scale by L2 norm (avoid division by zero)

    # Generate a scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(columns, scaled_loss, color="red", label="Scaled L2 Loss")
    plt.title("Scatter Plot of Scaled L2 Loss", fontsize=16)
    plt.ylabel("Scaled L2 Loss", fontsize=14)
    plt.xlabel("Probes", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.legend()

    # Save or show the scatter plot
    if save_dir:
        save_path = f"{save_dir}/scaled_l2_loss_scatter_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to {save_path}")
    else:
        plt.show()
