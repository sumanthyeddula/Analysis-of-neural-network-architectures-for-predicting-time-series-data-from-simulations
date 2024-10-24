import pandas as pd
import numpy as np
import glob
import os
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

def load_csv_files(directory: str, file_pattern: str) -> List[pd.DataFrame]:
    search_pattern = os.path.join(directory, file_pattern)
    all_files = glob.glob(search_pattern)
    dataframes = [pd.read_csv(file) for file in all_files]
    if not dataframes:
        print("No CSV files found in the specified directory.")
    return dataframes

def normalize_data(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

    # Normalize features and labels
    features_normalized = feature_scaler.fit_transform(features)
    labels_normalized = label_scaler.fit_transform(labels)

    return features_normalized, labels_normalized, feature_scaler, label_scaler

def create_feature_label_pairs(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    rotational_speed = df.get('rotational_speed', None)
    cd = df.get('cd', None)
    cl = df.get('cl', None)
    
    if rotational_speed is None or cd is None or cl is None:
        raise KeyError("One of the required columns ('rotational_speed', 'Cd', 'Cl') is missing.")
    
    features = np.column_stack((rotational_speed, cd, cl))
    labels = np.column_stack((cd, cl))
    return features, labels

def stack_feature_label_pairs(dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    features_list = []
    labels_list = []
    for df in dfs:
        features, labels = create_feature_label_pairs(df)
        features_list.append(features)
        labels_list.append(labels)
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    return features_array, labels_array


def create_tensor_dataset(features: np.ndarray, labels: np.ndarray, window_size: int) -> TensorDataset:
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(labels[i+window_size])
    X = np.array(X)
    y = np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)





