import torch as pt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle


def set_seed(seed_value=42):
    """
    Set the random seed for reproducibility.

    :param seed_value: The seed value to use for all libraries.
    """
    # Set the seed for Python's built-in random library
    random.seed(seed_value)

    # Set the seed for NumPy
    np.random.seed(seed_value)

    # Set the seed for PyTorch (for both CPU and GPU if available)
    pt.manual_seed(seed_value)
    if pt.cuda.is_available():
        pt.cuda.manual_seed(seed_value)
        pt.cuda.manual_seed_all(seed_value)  # if using multi-GPU setups

    # For deterministic behavior (may impact performance)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False


def split_dataset(data, test_size=0.25):

    # Split train data into smaller train and validation sets
    train_split, val_split = train_test_split(
        data, test_size=test_size, random_state=42
    )

    # Output results
    print("Training Features:")
    print(len(train_split))
    print("\nValidation Features:")
    print(len(val_split))

    return train_split, val_split


def save_model(model, file_path):
    """
    Saves the PyTorch model to the specified file path.

    :param model: Trained PyTorch model to be saved.
    :param file_path: Path to save the model (e.g., 'model.pth').
    """
    pt.save(model.state_dict(), f"{file_path}/{file_path}_best_model.pth")
    print(f"Model saved to {file_path}")


def normalize_column_data(data, sequence_length, n_steps, n_features):
    """
    Preprocess the dataset by normalizing each column separately in input sequences.

    Args:
        data: Dataset to preprocess (list of numpy arrays).
        sequence_length: Length of input sequences.
        n_steps: Number of prediction steps.
        n_features: Number of input features.

    Returns:
        List of normalized inputs and corresponding targets.
    """
    # Initialize scalers for each feature
    scalers = [MinMaxScaler() for _ in range(n_features)]

    # Collect all data column-wise for normalization
    combined_columns = [
        np.hstack([seq[:, i] for seq in data if seq.shape[1] == n_features]).reshape(
            -1, 1
        )
        for i in range(n_features)
    ]

    # Fit scalers to each column and normalize
    normalized_columns = [
        scaler.fit_transform(col) for scaler, col in zip(scalers, combined_columns)
    ]

    # Combine normalized columns back into sequences
    normalized_combined_data = np.hstack(normalized_columns)

    # Split the normalized data back into sequences and extract inputs/targets
    normalized_data = []
    current_index = 0
    for sequence in data:
        if (
            sequence.shape[1] != n_features
            or sequence.shape[0] < sequence_length + n_steps
        ):
            continue

        num_rows = sequence.shape[0]
        normalized_sequence = normalized_combined_data[
            current_index : current_index + num_rows
        ]
        current_index += num_rows

        # Prepare input and target data
        initial_input = normalized_sequence[0:sequence_length]
        target_data = normalized_sequence[
            sequence_length : sequence_length + n_steps, :14
        ]

        normalized_data.append((initial_input, target_data, sequence[:, -1]))

    with open("scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    print("Normalization done")
    return normalized_data


def denormalize_target_data(target_data, scalers=None):
    """
    Denormalize the target data column-wise using the provided or created scalers.

    Args:
        target_data: Normalized target data (list or numpy array of shape [n_samples, n_columns]).
        scalers: Optional list of MinMaxScaler objects, one for each column.
                 If None, new scalers will be created and used to fit the data.

    Returns:
        Denormalized target data (numpy array of shape [n_samples, n_columns]).
    """
    # Convert target_data to a NumPy array if it's a list
    if isinstance(target_data, list):
        target_data = np.array(
            target_data[0]
        )  # Extract the first element if it's a nested list
        if target_data.shape[1] == 1:  # Check if the second dimension is 1
            target_data = np.squeeze(target_data, axis=1)

        # target_data = np.squeeze(target_data, axis=1)  # Remove unnecessary dimensions

    # Ensure target_data is in the correct shape
    if (
        target_data.ndim == 3 and target_data.shape[1] == 1
    ):  # Shape (n_samples, 1, n_columns)
        target_data = target_data.reshape(target_data.shape[0], -1)

    # Extract shape information
    n_samples, n_columns = target_data.shape

    with open("scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # Ensure the number of scalers matches the number of columns
    assert (
        n_columns == len(scalers) - 1
    ), "The number of scalers must match the number of columns in target_data"

    # Split the data into columns
    columns = [target_data[:, i].reshape(-1, 1) for i in range(n_columns)]

    # Denormalize each column using the respective scaler
    denormalized_columns = [
        scalers[i].inverse_transform(columns[i]) for i in range(n_columns)
    ]

    # Combine the columns back into the original shape
    denormalized_data = np.hstack(denormalized_columns)

    print("Denormalization complete.")
    return denormalized_data


def renormalize_data_column_wise(data):
    """
    Normalize a 2D dataset column-wise using MinMaxScaler.

    Args:
        data: Numpy array of shape (n_samples, n_features).

    Returns:
        normalized_data: Numpy array of normalized data with the same shape as input.
        scalers: List of fitted MinMaxScaler objects, one for each column.
    """
    n_features = data.shape[1]  # Number of features

    with open("scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # Normalize each column
    normalized_columns = [
        scalers[i].transform(data[:, i].reshape(-1, 1)) for i in range(n_features)
    ]

    # Combine normalized columns back into a single array
    normalized_data = np.hstack(normalized_columns)

    print("Normalization complete.")
    return normalized_data


def calculate_prediction_accuracy(predictions, actuals):
    """
    Calculate prediction accuracy as the inverse of Mean Absolute Error (MAE).

    :param predictions: List of predicted values (numpy arrays).
    :param actuals: List of actual target values (numpy arrays).
    :return: List of accuracies for each dataset.
    """
    accuracies = []
    for pred, act in zip(predictions, actuals):
        mae = np.mean(np.abs(np.array(pred) - np.array(act)))
        accuracy = 1 - mae  # Accuracy metric as (1 - Mean Absolute Error)
        accuracies.append(accuracy)
    return accuracies
