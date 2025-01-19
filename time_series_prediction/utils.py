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
    """
    Split the dataset into training and validation subsets.

    :param data: Dataset to split.
    :param test_size: Proportion of the dataset to include in the validation split.
    :return: Tuple containing training and validation subsets.
    """
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


def save_model(model, file_path, model_name):
    """
    Saves the PyTorch model to the specified file path.

    :param model: Trained PyTorch model to be saved.
    :param file_path: Path to save the model (e.g., 'model.pth').
    :param model_name: Name for the model file.
    """
    pt.save(model.state_dict(), f"{file_path}/{model_name}_best_model.pth")
    print(f"Model saved to {file_path}")


def normalize_column_data(data, sequence_length, n_steps, n_features):
    """
    Preprocess the dataset by normalizing each column separately in input sequences.

    :param data: Dataset to preprocess (list of numpy arrays).
    :param sequence_length: Length of input sequences.
    :param n_steps: Number of prediction steps.
    :param n_features: Number of input features.
    :return: List of normalized inputs and corresponding targets.
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

    # Save the scalers for future use
    with open("scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    print("Normalization done")
    return normalized_data


def denormalize_target_data(target_data, scalers=None):
    """
    Denormalize the target data column-wise using the provided or loaded scalers.

    :param target_data: Normalized target data (list or numpy array of shape [n_samples, n_columns]).
    :param scalers: Optional list of MinMaxScaler objects, one for each column.
                    If None, scalers will be loaded from 'scalers.pkl'.
    :return: Denormalized target data (numpy array of shape [n_samples, n_columns]).
    """
    # Convert target_data to a NumPy array if it's a list
    if isinstance(target_data, list):
        target_data = np.array(target_data[0])
        if target_data.ndim == 3 and target_data.shape[1] == 1:
            target_data = target_data.reshape(target_data.shape[0], -1)

    # Ensure target_data is in the correct shape
    if target_data.ndim == 3 and target_data.shape[1] == 1:
        target_data = target_data.reshape(target_data.shape[0], -1)

    # Check for invalid values and cast to float32
    target_data = np.nan_to_num(
        target_data,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=np.finfo(np.float32).min,
    )
    target_data = target_data.astype(np.float32)

    # Load scalers if not provided
    if scalers is None:
        with open("scalers.pkl", "rb") as f:
            scalers = pickle.load(f)

    # Split the data into columns
    columns = [target_data[:, i].reshape(-1, 1) for i in range(target_data.shape[1])]

    # Denormalize each column using the respective scaler
    denormalized_columns = []
    for i in range(len(columns)):
        try:
            denormalized_column = scalers[i].inverse_transform(columns[i])
            denormalized_columns.append(denormalized_column)
        except Exception as e:
            print(f"Error in denormalizing column {i}: {e}")
            raise

    # Combine the columns back into the original shape
    denormalized_data = np.hstack(denormalized_columns)

    print("Denormalization complete.")
    return denormalized_data


def renormalize_data_column_wise(data):
    """
    Normalize a 2D dataset column-wise using MinMaxScaler.

    :param data: Numpy array of shape (n_samples, n_features).
    :return: Normalized data and fitted scalers.
    """
    n_features = data.shape[1]  # Number of features

    # Load scalers
    with open("scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # Normalize each column
    # normalized_columns = [
    #     scalers[i].transform(data[:, i].reshape(-1, 1)) for i in range(n_features)
    # ]

    # Normalize the first 15 columns
    normalized_columns = [
        scalers[i].transform(data[:, i].reshape(-1, 1))
        for i in range(min(15, n_features))
    ]

    # For the remaining columns (if any), keep them as they are
    if n_features > 15:
        unnormalized_columns = [
            data[:, i].reshape(-1, 1) for i in range(15, n_features)
        ]
        normalized_columns.extend(unnormalized_columns)

    # Combine normalized columns back into a single array
    normalized_data = np.hstack(normalized_columns)

    print("Normalization complete.")
    return normalized_data


def calculate_prediction_accuracy(predictions, actuals):
    """
    Calculate normalized prediction accuracy as 1 - Normalized Mean Absolute Error (MAE).

    :param predictions: List or array of predicted values.
    :param actuals: List or array of actual ground truth values.
    :return: List of accuracies for each prediction-actual pair.
    """
    accuracies = []
    for pred, act in zip(predictions, actuals):
        act = np.array(act)
        pred = np.array(pred)
        mae = np.mean(np.abs(pred - act))
        range_act = np.ptp(act)  # Range of actual values (max - min)
        if range_act > 0:
            mae_normalized = mae / range_act
            accuracy = 1 - mae_normalized
        else:
            accuracy = 1.0 if mae == 0 else 0.0  # Perfect or invalid case
        accuracies.append(accuracy)
    return accuracies


def calculate_prediction_accuracy_foreach_feature(predictions, actuals, column=0):
    """
    Calculate normalized prediction accuracy as 1 - Normalized Mean Absolute Error (MAE),
    based on a specified column of predictions and actuals.

    :param predictions: List or array of predicted values.
    :param actuals: List or array of actual ground truth values.
    :param column: Index of the column to use for accuracy calculation (default: 0).
    :return: List of accuracies for each prediction-actual pair.
    """
    accuracies = []
    for idx, (pred, act) in enumerate(zip(predictions, actuals)):
        # Ensure pred and act are arrays
        act = np.array(act)[:, column]  # Select the specified column of actual values
        pred = np.array(pred)[
            :, column
        ]  # Select the specified column of predicted values

        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(pred - act))

        # Calculate range of actual values
        range_act = np.ptp(act)  # np.ptp = max - min

        # Debugging: Log intermediate values
        print(f"Pair {idx}:")
        print(f"Actual values: {act}")
        print(f"Predicted values: {pred}")
        print(f"MAE: {mae}")
        print(f"Range of actual values: {range_act}")

        # Handle small range or perfect match cases
        if range_act > 1e-6:  # Avoid division by very small numbers
            mae_normalized = mae / range_act
            accuracy = 1 - mae_normalized
        else:
            accuracy = 1.0 if mae < 1e-6 else 0.0  # Perfect match or invalid case

        # Clamp accuracy to [0, 1] to avoid unexpected negatives
        accuracy = max(0.0, min(accuracy, 1.0))

        # Debugging: Log accuracy
        print(f"Normalized MAE: {mae / range_act if range_act > 0 else 'N/A'}")
        print(f"Accuracy: {accuracy}")
        print("-" * 50)

        accuracies.append(accuracy)

    return accuracies
