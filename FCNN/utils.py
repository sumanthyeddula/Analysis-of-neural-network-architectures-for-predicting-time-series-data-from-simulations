<<<<<<< HEAD
import torch as pt
import numpy as np
import random
from sklearn.model_selection import train_test_split

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



def split_dataset(data):
    

    # Example DataFrame
    # data = {
    #     'feature1': [10, 20, 30, 40, 50, 60],
    #     'feature2': [100, 200, 300, 400, 500, 600],
    #     'label': [0, 1, 0, 1, 0, 1]
    # }
    # df = pd.DataFrame(data)


  # Split train data into smaller train and validation sets
    train_split, val_split = train_test_split(data, test_size=0.2, random_state=42)

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

=======
import torch as pt
import numpy as np
import random
from sklearn.model_selection import train_test_split

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



def split_dataset(data):
    

    # Example DataFrame
    # data = {
    #     'feature1': [10, 20, 30, 40, 50, 60],
    #     'feature2': [100, 200, 300, 400, 500, 600],
    #     'label': [0, 1, 0, 1, 0, 1]
    # }
    # df = pd.DataFrame(data)


  # Split train data into smaller train and validation sets
    train_split, val_split = train_test_split(data, test_size=0.2, random_state=42)

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

>>>>>>> 10a8ac84e53d92c691fa9cc1d84ec25d63c6d803
