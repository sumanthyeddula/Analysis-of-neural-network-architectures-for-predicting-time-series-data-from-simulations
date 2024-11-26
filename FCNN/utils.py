import torch as pt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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



def split_dataset(data, test_size=0.2):


  # Split train data into smaller train and validation sets
    train_split, val_split = train_test_split(data, test_size=test_size, random_state=42)

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
    pt.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")



def normalize_data(data, sequence_length, n_steps, n_features):
    """
    Preprocess the dataset by normalizing input sequences.

    Args:
        data: Dataset to preprocess.
        sequence_length: Length of input sequences.
        n_steps: Number of prediction steps.
        n_features: Number of input features.

    Returns:
        List of normalized inputs and corresponding targets.
    """
    normalized_data = []

    scaler = MinMaxScaler()  # Use a single scaler for all sequences
    for sequence in data:
        dataframe = sequence[0]
        
        # Skip sequences with unexpected shapes or lengths
        if dataframe.shape[1] != n_features or dataframe.shape[0] < sequence_length + n_steps:
            continue
        
        # Prepare input and target data
        initial_input = dataframe[0:sequence_length]
        target_data = dataframe[sequence_length:sequence_length + n_steps, :14]

        # Normalize input
        normalized_input = scaler.fit_transform(initial_input)
        normalized_data.append((normalized_input, target_data, dataframe[:, -1]))

        
    print("normalization done")

    return normalized_data


