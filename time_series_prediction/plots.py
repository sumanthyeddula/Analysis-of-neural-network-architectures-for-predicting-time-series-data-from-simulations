import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_loss(train_losses, val_losses, save_path, model_name):
    """
    Plot the training and validation losses on a log scale and save the plot.

    :param train_losses: List of training losses over epochs.
    :param val_losses: List of validation losses over epochs.
    :param save_path: Directory path to save the plot.
    :param model_name: Name of the model to include in the filename.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Training and Validation Loss Over Epochs (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.savefig(f"{save_path}/{model_name}_loss_plot.png")
    print(f"Loss plot saved to {save_path}/{model_name}_loss_plot.png")


# def plot_selected_columns(labels_data, prediction_data, save_dir=None, plot_name=""):
#     """
#     Plot selected columns ('Cd', 'Cl', and probe data) with subplots.

#     :param labels_data: Ground truth data (NumPy array or DataFrame).
#     :param prediction_data: Predicted data (NumPy array or DataFrame).
#     :param save_dir: Directory to save the plot. If None, the plot is displayed.
#     :param plot_name: Name to include in the saved plot file.
#     """
#     columns = [
#         "Cd",
#         "Cl",
#         "p_probe_0",
#         "p_probe_1",
#         "p_probe_2",
#         "p_probe_3",
#         "p_probe_4",
#         "p_probe_5",
#         "p_probe_6",
#         "p_probe_7",
#         "p_probe_8",
#         "p_probe_9",
#         "p_probe_10",
#         "p_probe_11",
#     ]

#     ground_truth_df = pd.DataFrame(labels_data, columns=columns)
#     predictions_df = pd.DataFrame(prediction_data, columns=columns)

#     cd_cl_columns = ["Cd", "Cl"]
#     probe_columns = [col for col in columns if col not in cd_cl_columns]

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     for i, column in enumerate(cd_cl_columns):
#         ax = axes[i]
#         ax.plot(ground_truth_df[column], label=f"{column} (labels)", linewidth=1.0)
#         ax.plot(
#             predictions_df[column],
#             label=f"{column} (predictions)",
#             linewidth=1.0,
#             linestyle="--",
#         )
#         ax.set_title(f"{column}")
#         ax.legend()
#         ax.grid(True)
#         ax.set_xlim([400, 1000])

#     plt.tight_layout()
#     if save_dir:
#         save_path = f"{save_dir}/{plot_name}_cd_cl_plot.png"
#         plt.savefig(save_path, dpi=900)
#         print(f"'Cd' and 'Cl' plot saved to {save_path}")
#     else:
#         plt.show()

#     n_cols = 4
#     n_rows = (len(probe_columns) + n_cols - 1) // n_cols
#     fig, axes = plt.subplots(
#         n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=False
#     )
#     axes = axes.flatten()

#     for i, column in enumerate(probe_columns):
#         ax = axes[i]
#         ax.plot(ground_truth_df[column], label=f"{column} (labels)", linewidth=1.0)
#         ax.plot(
#             predictions_df[column],
#             label=f"{column} (predictions)",
#             linewidth=1.0,
#             linestyle="--",
#         )
#         ax.set_title(f"{column}")
#         ax.legend()
#         ax.grid(True)
#         ax.set_xlim([400, 1000])

#     for j in range(len(probe_columns), len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     if save_dir:
#         save_path = f"{save_dir}/{plot_name}_probe_columns_plot.png"
#         plt.savefig(save_path, dpi=900)
#         print(f"Probe columns plot saved to {save_path}")
#     else:
#         plt.show()


def plot_selected_columns(labels_data, prediction_data, save_dir=None, plot_name=""):
    """
    Plot selected columns ('Cd', 'Cl', and probe data) with subplots in a 3×4 grid row-wise.

    :param labels_data: Ground truth data (NumPy array or DataFrame).
    :param prediction_data: Predicted data (NumPy array or DataFrame).
    :param save_dir: Directory to save the plot. If None, the plot is displayed.
    :param plot_name: Name to include in the saved plot file.
    """
    columns = [
        "Cd",
        "Cl",
        "probe_0",
        "probe_1",
        "probe_2",
        "probe_3",
        "probe_4",
        "probe_5",
        "probe_6",
        "probe_7",
        "probe_8",
        "probe_9",
        "probe_10",
        "probe_11",
    ]

    ground_truth_df = pd.DataFrame(labels_data, columns=columns)
    predictions_df = pd.DataFrame(prediction_data, columns=columns)

    # Transform x-axis: Map 0 → 400 and 600 → 1000
    x_original = np.linspace(0, 600, num=len(ground_truth_df))
    x_transformed = np.linspace(400, 1000, num=len(ground_truth_df))

    cd_cl_columns = ["Cd", "Cl"]
    probe_columns = [
        "probe_0",
        "probe_3",
        "probe_6",
        "probe_9",
        "probe_1",
        "probe_4",
        "probe_7",
        "probe_10",
        "probe_2",
        "probe_5",
        "probe_8",
        "probe_11",
    ]

    # Plot Cd and Cl
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, column in enumerate(cd_cl_columns):
        ax = axes[i]
        ax.plot(
            x_original,
            ground_truth_df[column],
            label=f"{column} (labels)",
            linewidth=1.0,
        )
        ax.plot(
            x_original,
            predictions_df[column],
            label=f"{column} (predictions)",
            linewidth=1.0,
            linestyle="--",
        )

        ax.set_title(f"{column}")
        ax.legend()
        ax.grid(True)

        # Set transformed x-axis values
        ax.set_xlim([0, 600])
        ax.set_xticks(np.linspace(0, 600, num=7))
        ax.set_xticklabels(np.linspace(400, 1000, num=7, dtype=int))

        ax.set_xlabel("No. of Samples")
        ax.set_ylabel("Values")

    plt.tight_layout()
    if save_dir:
        save_path = f"{save_dir}/{plot_name}_cd_cl_plot.png"
        plt.savefig(save_path, dpi=900)
        print(f"'Cd' and 'Cl' plot saved to {save_path}")
    else:
        plt.show()

    # Set up 3 rows × 4 columns grid for probe columns
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(20, 12), sharex=True, sharey=False
    )

    # Reshape axes to allow row-wise iteration
    axes = axes.reshape(n_rows, n_cols)

    probe_index = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if probe_index >= len(probe_columns):
                fig.delaxes(axes[row, col])  # Remove empty subplots
                continue

            column = probe_columns[probe_index]
            ax = axes[row, col]
            ax.plot(
                x_original,
                ground_truth_df[column],
                label=f"{column} (labels)",
                linewidth=1.0,
            )
            ax.plot(
                x_original,
                predictions_df[column],
                label=f"{column} (predictions)",
                linewidth=1.0,
                linestyle="--",
            )

            ax.set_title(f"{column}")
            ax.legend()
            ax.grid(True)

            # Set transformed x-axis values
            ax.set_xlim([0, 600])
            ax.set_xticks(np.linspace(0, 600, num=7))
            ax.set_xticklabels(np.linspace(400, 1000, num=7, dtype=int))

            ax.set_xlabel("No. of Samples")
            ax.set_ylabel("Values")

            probe_index += 1

    plt.tight_layout()
    if save_dir:
        save_path = f"{save_dir}/{plot_name}_probe_columns_plot.png"
        plt.savefig(save_path, dpi=900)
        print(f"Probe columns plot saved to {save_path}")
    else:
        plt.show()


# def plot_selected_columns(labels_data, prediction_data, save_dir=None, plot_name=""):
#     """
#     Plot selected columns ('Cd', 'Cl', and probe data) with subplots.

#     :param labels_data: Ground truth data (NumPy array or DataFrame).
#     :param prediction_data: Predicted data (NumPy array or DataFrame).
#     :param save_dir: Directory to save the plot. If None, the plot is displayed.
#     :param plot_name: Name to include in the saved plot file.
#     """
#     columns = [
#         "Cd",
#         "Cl",
#         "p_probe_0",
#         "p_probe_1",
#         "p_probe_2",
#         "p_probe_3",
#         "p_probe_4",
#         "p_probe_5",
#         "p_probe_6",
#         "p_probe_7",
#         "p_probe_8",
#         "p_probe_9",
#         "p_probe_10",
#         "p_probe_11",
#     ]

#     ground_truth_df = pd.DataFrame(labels_data, columns=columns)
#     predictions_df = pd.DataFrame(prediction_data, columns=columns)

#     # Transform x-axis: Map 0 → 400 and 600 → 1000
#     x_original = np.linspace(0, 600, num=len(ground_truth_df))
#     x_transformed = np.linspace(400, 1000, num=len(ground_truth_df))

#     cd_cl_columns = ["Cd", "Cl"]
#     probe_columns = [col for col in columns if col not in cd_cl_columns]

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     for i, column in enumerate(cd_cl_columns):
#         ax = axes[i]
#         ax.plot(
#             x_original,
#             ground_truth_df[column],
#             label=f"{column} (labels)",
#             linewidth=1.0,
#         )
#         ax.plot(
#             x_original,
#             predictions_df[column],
#             label=f"{column} (predictions)",
#             linewidth=1.0,
#             linestyle="--",
#         )

#         ax.set_title(f"{column}")
#         ax.legend()
#         ax.grid(True)

#         # Set transformed x-axis values
#         ax.set_xlim([0, 600])
#         ax.set_xticks(np.linspace(0, 600, num=7))  # Setting ticks at equal intervals
#         ax.set_xticklabels(
#             np.linspace(400, 1000, num=7, dtype=int)
#         )  # Transforming labels

#         ax.set_xlabel("No. of Samples")  # Naming x-axis
#         ax.set_ylabel("Values")

#     plt.tight_layout()
#     if save_dir:
#         save_path = f"{save_dir}/{plot_name}_cd_cl_plot.png"
#         plt.savefig(save_path, dpi=900)
#         print(f"'Cd' and 'Cl' plot saved to {save_path}")
#     else:
#         plt.show()

#     n_cols = 4
#     n_rows = (len(probe_columns) + n_cols - 1) // n_cols
#     fig, axes = plt.subplots(
#         n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=False
#     )
#     axes = axes.flatten()

#     for i, column in enumerate(probe_columns):
#         ax = axes[i]
#         ax.plot(
#             x_original,
#             ground_truth_df[column],
#             label=f"{column} (labels)",
#             linewidth=1.0,
#         )
#         ax.plot(
#             x_original,
#             predictions_df[column],
#             label=f"{column} (predictions)",
#             linewidth=1.0,
#             linestyle="--",
#         )

#         ax.set_title(f"{column}")
#         ax.legend()
#         ax.grid(True)

#         # Set transformed x-axis values
#         ax.set_xlim([0, 600])
#         ax.set_xticks(np.linspace(0, 600, num=7))  # Setting ticks at equal intervals
#         ax.set_xticklabels(
#             np.linspace(400, 1000, num=7, dtype=int)
#         )  # Transforming labels

#         ax.set_xlabel("No. of Samples")  # Naming x-axis
#         ax.set_ylabel("Values")

#     for j in range(len(probe_columns), len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     if save_dir:
#         save_path = f"{save_dir}/{plot_name}_probe_columns_plot.png"
#         plt.savefig(save_path, dpi=900)
#         print(f"Probe columns plot saved to {save_path}")
#     else:
#         plt.show()


def compute_scaled_l2_loss_heatmap(labels_data, prediction_data, save_dir=None):
    """
    Compute scaled L2 loss and generate a heatmap.

    :param labels_data: Ground truth data (NumPy array or DataFrame).
    :param prediction_data: Predicted data (NumPy array or DataFrame).
    :param save_dir: Directory to save the heatmap. If None, the heatmap is displayed.
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

    ground_truth_df = pd.DataFrame(labels_data, columns=columns)
    predictions_df = pd.DataFrame(prediction_data, columns=columns)

    l2_loss = np.square(ground_truth_df.values - predictions_df.values).sum(axis=0)
    l2_norm = np.sqrt(np.square(ground_truth_df.values).sum(axis=0))
    scaled_loss = l2_loss / (l2_norm + 1e-8)

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

    if save_dir:
        save_path = f"{save_dir}/scaled_l2_loss_heatmap.png"
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()


def compute_scaled_l2_loss_scatter(
    labels_data, prediction_data, save_dir=None, plot_name=""
):
    """
    Compute scaled L2 loss and generate a scatter plot.

    :param labels_data: Ground truth data (NumPy array or DataFrame).
    :param prediction_data: Predicted data (NumPy array or DataFrame).
    :param save_dir: Directory to save the scatter plot. If None, the plot is displayed.
    :param plot_name: Name to include in the saved scatter plot file.
    """
    columns = [
        "Cd",
        "Cl",
        "p_0",
        "p_1",
        "p_2",
        "p_3",
        "p_4",
        "p_5",
        "p_6",
        "p_7",
        "p_8",
        "p_9",
        "p_10",
        "p_11",
    ]  # Updated probe names to "p_x"

    # Create dataframes for ground truth and predictions
    ground_truth_df = pd.DataFrame(labels_data, columns=columns)
    predictions_df = pd.DataFrame(prediction_data, columns=columns)

    # Compute scaled L2 loss
    l2_loss = np.sqrt(
        np.square(ground_truth_df.values - predictions_df.values).sum(axis=0)
    )
    l2_norm = np.sqrt(np.square(ground_truth_df.values).sum(axis=0))
    scaled_loss = l2_loss / (l2_norm + 1e-8)

    # Plot scaled L2 loss scatter
    plt.figure(figsize=(12, 6))
    plt.scatter(columns, scaled_loss, color="red", label="Scaled $L_2$ Loss")
    plt.title("Scatter Plot of Scaled $L_2$ Loss", fontsize=16)
    plt.ylabel(
        "$\\text{Scaled } L_2 \\text{ Loss} = \\frac{\\sqrt{\\sum (p_{true} - p_{pred})^2}}{\\sqrt{\\sum (p_{true})^2}}$",
        fontsize=14,
    )
    plt.xlabel("Probes (p)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.legend(title="Legend")
    plt.text(
        0.5,
        -0.15,
        "p = Pressure Probes",
        fontsize=12,
        transform=plt.gca().transAxes,
        ha="center",
        va="top",
    )

    if save_dir:
        save_path = f"{save_dir}/{plot_name}_scaled_l2_loss_scatter_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to {save_path}")
    else:
        plt.show()


def prediction_accuracy_with_error_bars_each_feature(
    predictions, actuals, save_dir=None
):
    """
    Calculate and plot prediction accuracy with error bars for each feature as scatter plots.

    :param predictions: List of predicted values (numpy arrays).
    :param actuals: List of actual target values (numpy arrays).
    :param save_dir: Directory to save the plot (optional).
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

    predictions = [np.array(p) for p in predictions]
    actuals = [np.array(a) for a in actuals]

    n_features = len(columns)
    column_accuracies = []
    column_error_bars = []

    # Compute accuracy and error bars for each column
    for col in range(n_features):
        col_accuracies = []
        for pred, act in zip(predictions, actuals):
            mae = np.mean(np.abs(pred[:, col] - act[:, col]))
            accuracy = 1 - mae
            col_accuracies.append(accuracy)
        column_accuracies.append(np.mean(col_accuracies))
        column_error_bars.append(np.std(col_accuracies))

    # Plot scatter plot with error bars
    plt.figure(figsize=(12, 6))
    x_positions = range(n_features)
    plt.errorbar(
        x_positions,
        column_accuracies,
        yerr=column_error_bars,
        fmt="o",
        color="blue",
        ecolor="black",
        elinewidth=1,
        capsize=5,
        label="Accuracy with Error Bars",
    )

    # Formatting the plot
    plt.xticks(x_positions, columns, rotation=45, fontsize=10)
    plt.title(
        "Column-Wise Prediction Accuracy with Error Bars (Scatter Plot)", fontsize=16
    )
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    # Save or display the plot
    if save_dir:
        save_path = f"{save_dir}/prediction_accuracy_error_bars_scatter.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Prediction accuracy plot saved to {save_path}")
    else:
        plt.show()


def plot_3d_frequency_amplitude_accuracy(frequency, amplitude, accuracy):
    """
    Plot a 3D scatter plot of frequency, amplitude, and accuracy.

    :param frequency: Array of frequency values.
    :param amplitude: Array of amplitude values.
    :param accuracy: Array of accuracy values.
    """
    # Ensure inputs are NumPy arrays for consistency
    frequency = np.array(frequency)
    amplitude = np.array(amplitude)
    accuracy = np.array(accuracy)

    # Create a 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot
    sc = ax.scatter(frequency, amplitude, accuracy, c=accuracy, cmap="viridis", s=50)

    # Labels and title
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.set_zlabel("Accuracy", fontsize=12)
    ax.set_title("3D Plot of Frequency, Amplitude, and Accuracy", fontsize=14)

    # Add color bar
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Accuracy", fontsize=12)

    # Show plot
    plt.show()


from scipy.interpolate import griddata


def plot_contour_frequency_amplitude_accuracy(frequency, amplitude, accuracy):
    """
    Plot a contour plot of frequency, amplitude, and accuracy.

    :param frequency: Array of frequency values.
    :param amplitude: Array of amplitude values.
    :param accuracy: Array of accuracy values.
    """
    # Ensure inputs are NumPy arrays
    frequency = np.array(frequency)
    amplitude = np.array(amplitude)
    accuracy = np.array(accuracy)

    # Check for consistent shapes
    if not (frequency.shape == amplitude.shape == accuracy.shape):
        raise ValueError("frequency, amplitude, and accuracy must have the same shape.")

    # Create a grid for frequency and amplitude
    grid_x, grid_y = np.meshgrid(
        np.linspace(frequency.min(), frequency.max(), 100),
        np.linspace(amplitude.min(), amplitude.max(), 100),
    )

    # Interpolate accuracy values on the grid
    points = np.column_stack((frequency, amplitude))  # Combine frequency and amplitude
    grid_accuracy = griddata(points, accuracy, (grid_x, grid_y), method="cubic")

    # Check for NaNs in interpolated data (due to extrapolation limits)
    if np.any(np.isnan(grid_accuracy)):
        print(
            "Warning: Some values in the grid are NaN. Adjust the grid range or use 'linear' interpolation."
        )

    # Create a contour plot
    plt.figure(figsize=(10, 7))
    contour = plt.contourf(grid_x, grid_y, grid_accuracy, levels=50, cmap="viridis")
    plt.colorbar(contour, label="Accuracy")
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.title("Contour Plot of Frequency, Amplitude, and Accuracy", fontsize=14)
    plt.show()


# def plot_contour_accuracy_for_features_array(
#     frequency, amplitude, accuracies, save_dir=None, plot_name=""
# ):
#     """
#     Plot contour accuracy plots for 'Cd', 'Cl', and probe data in separate figures.

#     :param frequency: List or array of frequency values.
#     :param amplitude: List or array of amplitude values.
#     :param accuracies: List or array where each entry contains the accuracies for one feature.
#     :param save_dir: Directory to save the plot. If None, the plot is displayed.
#     :param plot_name: Name to include in the saved plot file.
#     """
#     # Convert frequency and amplitude to NumPy arrays if they are not already
#     frequency = np.array(frequency)
#     amplitude = np.array(amplitude)

#     # Define feature names based on the given format
#     feature_names = [
#         "Cd",
#         "Cl",
#         "p_probe_0",
#         "p_probe_1",
#         "p_probe_2",
#         "p_probe_3",
#         "p_probe_4",
#         "p_probe_5",
#         "p_probe_6",
#         "p_probe_7",
#         "p_probe_8",
#         "p_probe_9",
#         "p_probe_10",
#         "p_probe_11",
#     ]

#     # Validate that the number of features matches the given accuracies
#     if len(accuracies) != len(feature_names):
#         raise ValueError(
#             f"The number of accuracy arrays ({len(accuracies)}) "
#             f"does not match the expected number of features ({len(feature_names)})."
#         )

#     cd_cl_columns = ["Cd", "Cl"]
#     probe_columns = [name for name in feature_names if name not in cd_cl_columns]

#     # Map feature names to indices in the accuracies list
#     feature_indices = {name: i for i, name in enumerate(feature_names)}

#     # Plot Cd and Cl
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     for i, column in enumerate(cd_cl_columns):
#         ax = axes[i]

#         # Extract accuracy for the current feature
#         accuracy = np.array(accuracies[feature_indices[column]])
#         points = np.column_stack((frequency, amplitude))
#         grid_x, grid_y = np.meshgrid(
#             np.linspace(frequency.min(), frequency.max(), 100),
#             np.linspace(amplitude.min(), amplitude.max(), 100),
#         )
#         grid_accuracy = griddata(points, accuracy, (grid_x, grid_y), method="cubic")

#         # Handle NaN in the interpolated grid
#         if np.any(np.isnan(grid_accuracy)):
#             print(
#                 f"Warning: Some values in the grid for {column} are NaN. "
#                 f"Adjusting interpolation method to 'linear'."
#             )
#             grid_accuracy = griddata(
#                 points, accuracy, (grid_x, grid_y), method="linear"
#             )

#         contour = ax.contourf(grid_x, grid_y, grid_accuracy, levels=50, cmap="viridis")
#         fig.colorbar(contour, ax=ax, label="Accuracy")
#         ax.set_title(f"{column} Contour")
#         ax.set_xlabel("Frequency")
#         ax.set_ylabel("Amplitude")
#         ax.grid(True)

#     plt.tight_layout()
#     if save_dir:
#         save_path = f"{save_dir}/{plot_name}_cd_cl_contour.png"
#         plt.savefig(save_path, dpi=900)
#         print(f"'Cd' and 'Cl' contour plot saved to {save_path}")
#     else:
#         plt.show()

#     # Plot Probe Columns
#     n_cols = 4
#     n_rows = (len(probe_columns) + n_cols - 1) // n_cols
#     fig, axes = plt.subplots(
#         n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=False
#     )
#     axes = axes.flatten()

#     for i, column in enumerate(probe_columns):
#         ax = axes[i]

#         # Extract accuracy for the current feature
#         accuracy = np.array(accuracies[feature_indices[column]])
#         points = np.column_stack((frequency, amplitude))
#         grid_x, grid_y = np.meshgrid(
#             np.linspace(frequency.min(), frequency.max(), 100),
#             np.linspace(amplitude.min(), amplitude.max(), 100),
#         )
#         grid_accuracy = griddata(points, accuracy, (grid_x, grid_y), method="cubic")

#         # Handle NaN in the interpolated grid
#         if np.any(np.isnan(grid_accuracy)):
#             print(
#                 f"Warning: Some values in the grid for {column} are NaN. "
#                 f"Adjusting interpolation method to 'linear'."
#             )
#             grid_accuracy = griddata(
#                 points, accuracy, (grid_x, grid_y), method="linear"
#             )

#         contour = ax.contourf(grid_x, grid_y, grid_accuracy, levels=50, cmap="viridis")
#         fig.colorbar(contour, ax=ax, label="Accuracy")
#         ax.set_title(f"{column} Contour")
#         ax.set_xlabel("Frequency")
#         ax.set_ylabel("Amplitude")
#         ax.grid(True)

#     # Remove empty subplots
#     for j in range(len(probe_columns), len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     if save_dir:
#         save_path = f"{save_dir}/{plot_name}_probe_contour.png"
#         plt.savefig(save_path, dpi=900)
#         print(f"Probe contour plot saved to {save_path}")
#     else:
#         plt.show()


def plot_contour_accuracy_for_features_array(
    frequency, amplitude, accuracies, save_dir=None, plot_name=""
):
    """
    Plot contour accuracy plots for 'Cd', 'Cl', and probe data in separate figures.
    Also overlays scatter points for given frequency and amplitude values.

    :param frequency: List or array of frequency values.
    :param amplitude: List or array of amplitude values.
    :param accuracies: List or array where each entry contains the accuracies for one feature.
    :param save_dir: Directory to save the plot. If None, the plot is displayed.
    :param plot_name: Name to include in the saved plot file.
    """
    # Convert frequency and amplitude to NumPy arrays if they are not already
    frequency = np.array(frequency)
    amplitude = np.array(amplitude)

    # Define feature names
    feature_names = [
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

    # Validate number of features
    if len(accuracies) != len(feature_names):
        raise ValueError(
            f"The number of accuracy arrays ({len(accuracies)}) does not match "
            f"the expected number of features ({len(feature_names)})."
        )

    cd_cl_columns = ["Cd", "Cl"]
    probe_columns = [name for name in feature_names if name not in cd_cl_columns]
    feature_indices = {name: i for i, name in enumerate(feature_names)}

    # Plot Cd and Cl
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, column in enumerate(cd_cl_columns):
        ax = axes[i]

        # Extract accuracy data
        accuracy = np.array(accuracies[feature_indices[column]])
        points = np.column_stack((frequency, amplitude))

        # Create grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(frequency.min(), frequency.max(), 100),
            np.linspace(amplitude.min(), amplitude.max(), 100),
        )

        # Interpolate accuracy values
        grid_accuracy = griddata(points, accuracy, (grid_x, grid_y), method="cubic")

        # Handle NaN values in interpolation
        if np.any(np.isnan(grid_accuracy)):
            print(
                f"Warning: Some values in the grid for {column} are NaN. Switching to 'linear'."
            )
            grid_accuracy = griddata(
                points, accuracy, (grid_x, grid_y), method="linear"
            )

        # Contour plot
        contour = ax.contourf(grid_x, grid_y, grid_accuracy, levels=50, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="Accuracy")

        # Scatter points (Red)
        ax.scatter(frequency, amplitude, c="red", marker="o", s=10, label="Data Points")

        ax.set_title(f"{column} Contour")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_dir:
        save_path = f"{save_dir}/{plot_name}_cd_cl_contour.png"
        plt.savefig(save_path, dpi=900)
        print(f"'Cd' and 'Cl' contour plot saved to {save_path}")
    else:
        plt.show()

    # Plot Probe Columns
    n_cols = 4
    n_rows = (len(probe_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=False
    )
    axes = axes.flatten()

    for i, column in enumerate(probe_columns):
        ax = axes[i]

        # Extract accuracy data
        accuracy = np.array(accuracies[feature_indices[column]])
        points = np.column_stack((frequency, amplitude))

        # Create grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(frequency.min(), frequency.max(), 100),
            np.linspace(amplitude.min(), amplitude.max(), 100),
        )

        # Interpolate accuracy values
        grid_accuracy = griddata(points, accuracy, (grid_x, grid_y), method="cubic")

        # Handle NaN values in interpolation
        if np.any(np.isnan(grid_accuracy)):
            print(
                f"Warning: Some values in the grid for {column} are NaN. Switching to 'linear'."
            )
            grid_accuracy = griddata(
                points, accuracy, (grid_x, grid_y), method="linear"
            )

        # Contour plot
        contour = ax.contourf(grid_x, grid_y, grid_accuracy, levels=50, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="Accuracy")

        # Scatter points (Red)
        ax.scatter(frequency, amplitude, c="red", marker="o", s=10, label="Data Points")

        ax.set_title(f"{column} Contour")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

    # Remove empty subplots
    for j in range(len(probe_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_dir:
        save_path = f"{save_dir}/{plot_name}_probe_contour.png"
        plt.savefig(save_path, dpi=900)
        print(f"Probe contour plot saved to {save_path}")
    else:
        plt.show()


import optuna.visualization as vis


def plot_optimization_results(study, model_type):
    """
    Plot optimization results from an Optuna study.

    :param study: The Optuna study object after optimization.
    :param model_type: A string to identify the model type (e.g., "FCNN" or "LSTM").
    """

    # Plot the validation loss over trials
    plt.figure(figsize=(10, 6))
    trials = study.trials
    losses = [trial.value for trial in trials]
    plt.plot(
        range(len(losses)), losses, marker="o", linestyle="--", label="Validation Loss"
    )
    plt.xlabel("Trial Number")
    plt.ylabel("Validation Loss")
    plt.title(f"Optimization Progress for {model_type}")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot optimization history using Optuna's built-in visualization
    vis.plot_optimization_history(study).show()

    # Plot parallel coordinate for hyperparameter relationships
    vis.plot_parallel_coordinate(study).show()

    # Plot parameter importance
    vis.plot_param_importances(study).show()

    # Plot slice plots
    vis.plot_slice(study).show()
