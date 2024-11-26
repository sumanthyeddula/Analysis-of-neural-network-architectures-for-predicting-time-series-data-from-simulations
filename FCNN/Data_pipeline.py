import os
import pandas as pd
import numpy as np
from typing import Tuple, List
from utils import split_dataset

def extract_force_coefficients(filepath: str) -> pd.DataFrame:
    """Extracts only the drag (Cd) and lift (Cl) coefficients from a specified file, excluding time."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    # Skip initial non-data lines and select Cd, Cl columns
    force_df = pd.read_csv(filepath, delim_whitespace=True, skiprows=13, usecols=[1, 2], names=['Cd', 'Cl'])

    return force_df

def extract_pressure_probes(filepath: str) -> pd.DataFrame:
    """Extracts only pressure values from probe file, excluding time."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    
    # Load numeric pressure data while handling initial non-numeric metadata rows
    pressure_data_rows = []
    with open(filepath, 'r') as file:
        for line in file:
            try:
                values = [float(x) for x in line.split()]
                pressure_data_rows.append(values[1:13])  # Extracting only 12 probe columns
            except ValueError:
                continue  # Skip non-numeric rows
    
    # Create DataFrame with 12 pressure probe columns
    probe_df = pd.DataFrame(pressure_data_rows, columns=[f'p_probe_{i}' for i in range(12)])
    return probe_df

def extract_actuation_parameters(filepath: str) -> Tuple[float, float]:
    """Extracts amplitude and frequency from the U file, ignoring expression lines."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    
    amplitude, frequency = None, None
    with open(filepath, 'r') as file:
        for line in file:
            if 'amplitude' in line and 'valueExpr' not in line:
                try:
                    amplitude = float(line.split()[-1].strip(';'))
                except ValueError:
                    print(f"Warning: Unable to parse amplitude value in line: {line.strip()}")
            elif 'frequency' in line and 'valueExpr' not in line:
                try:
                    frequency = float(line.split()[-1].strip(';'))
                except ValueError:
                    print(f"Warning: Unable to parse frequency value in line: {line.strip()}")

    if amplitude is None or frequency is None:
        raise ValueError(f"Amplitude or frequency not found or invalid in file: {filepath}")
    
    return amplitude, frequency

def calculate_rotational_speed(amplitude: float, frequency: float, times: np.ndarray) -> pd.DataFrame:
    """Calculates the rotational speed over a time range using the specified amplitude and frequency."""
    omega = 2 * np.pi * frequency  # Angular frequency
    area = 1.0  # Area of the face (assumed to be 1.0)
    theta = 0  # Angle for face normal vector (0 degrees)

    # Unit vector in the z-direction and face normal vector
    k_vector = np.array([0, 0, 1])
    face_normal = np.array([np.cos(theta), np.sin(theta), 0])

    # Cross product and expression factor
    cross_product = np.cross(k_vector, face_normal)
    expression_factor = cross_product / area

    # Calculate rotational speed for each time step
    sine_term = np.sin(omega * times)
    rotational_speed_vector = np.outer(sine_term * amplitude, expression_factor)
    rotational_speed_magnitude = np.linalg.norm(rotational_speed_vector, axis=1)

    return pd.DataFrame({'rotational_speed': rotational_speed_magnitude})

def load_base_data(base_dir: str) -> pd.DataFrame:
    """Loads base data from folder 0/cylinder2D in the directory."""
    # Update paths to include 'cylinder2D'
    force_path = os.path.join(base_dir, 'cylinder2D', 'postProcessing', 'forces', '0', 'coefficient.dat')
    probe_path = os.path.join(base_dir, 'cylinder2D', 'postProcessing', 'probes', '0', 'p')
    actuation_path = os.path.join(base_dir, 'cylinder2D', '0.org', 'U')

    # Extract force, probe, and actuation parameters
    base_force_df = extract_force_coefficients(force_path)

    base_probe_df = extract_pressure_probes(probe_path)
    amplitude, frequency = extract_actuation_parameters(actuation_path)

    # Time array for base data (assuming time steps correspond to data rows)
    num_rows = base_force_df.shape[0]
    dt = 0.01  # Assuming a time step of 0.01 seconds
    times = np.arange(0, num_rows * dt, dt)

    # Calculate rotational speeds for base data
    rotational_speeds_df = calculate_rotational_speed(amplitude, frequency, times)

    # Concatenate force, probe, and rotational speed data
    base_combined_df = pd.concat([base_force_df.reset_index(drop=True), base_probe_df.reset_index(drop=True), rotational_speeds_df], axis=1)

    return base_combined_df

def create_simulation_dataframe(sim_dir: str, base_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame for a specific simulation folder by combining base data and new data."""
    # Define paths using the specific simulation directory
    force_path = os.path.join(sim_dir, 'cylinder2D', 'postProcessing', 'forces', '0', 'coefficient.dat')
    probe_path = os.path.join(sim_dir, 'cylinder2D', 'postProcessing', 'probes', '0', 'p')
    actuation_path = os.path.join(sim_dir, 'cylinder2D', '0.org', 'U')

    # Log paths being used to verify unique folders are accessed
    print(f"\nReading data from: {sim_dir}")
    print(f"Force data path: {force_path}")
    print(f"Probe data path: {probe_path}")
    print(f"Actuation data path: {actuation_path}")

    # Re-extract amplitude and frequency for this specific simulation
    amplitude, frequency = extract_actuation_parameters(actuation_path)
    
    # Extract force and probe data
    sim_force_df = extract_force_coefficients(force_path)
    sim_probe_df = extract_pressure_probes(probe_path)

    # Time array for simulation data (assuming time steps correspond to data rows)
    num_rows = sim_force_df.shape[0]
    dt = 0.01  # Assuming a time step of 0.01 seconds
    start_time = base_df.shape[0] * dt  # Continue from where base data ended
    times = np.arange(start_time, start_time + num_rows * dt, dt)

    # Calculate rotational speeds for simulation data using simulation's amplitude and frequency
    rotational_speeds_df = calculate_rotational_speed(amplitude, frequency, times)

    # Concatenate force, probe, and rotational speed data
    sim_combined_df = pd.concat([sim_force_df.reset_index(drop=True), sim_probe_df.reset_index(drop=True), rotational_speeds_df], axis=1)

    # Combine base and simulation data
    final_df = pd.concat([base_df, sim_combined_df], ignore_index=True)

    # Select only the required columns that exist in final_df
    required_columns = ['Cd', 'Cl'] + list(sim_probe_df.columns) + ['rotational_speed']
    final_df = final_df[required_columns]

    print(final_df.head())  # Display first few rows of the combined DataFrame

    return final_df

def process_all_simulations(base_path: str, train: bool, test_size: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Processes all simulations in the exercises directory, combining base and simulation data."""
    # Load base data from folder 0/cylinder2D
    base_dir = os.path.join(base_path, '0')
    base_df = load_base_data(base_dir)

    # Initialize list to hold all DataFrames for each simulation
    dataframes = []

    # Loop through each simulation folder (1, 2, ..., n)
    for sim_folder in sorted(os.listdir(base_path)):
        if sim_folder.isdigit() and int(sim_folder) > 0:  # Skip the base folder (0)
            sim_dir = os.path.join(base_path, sim_folder)
            if os.path.isdir(sim_dir):
                print(f"Processing simulation folder: {sim_folder}")
                # Create DataFrame for this specific simulation with updated amplitude and frequency
                sim_df = create_simulation_dataframe(sim_dir, base_df)
                
                # Extract features and labels
                features = sim_df.values  # All columns as features
                labels = sim_df.drop(columns=['rotational_speed']).values  # Exclude 'rotational_speed' as labels
                
                dataframes.append((features, labels))  # Store feature-label pairs as tuples

    train_data, test_data = split_dataset(dataframes, test_size=test_size)

    if train:
        return train_data
    else:   
        return test_data    

# Define the main path for the exercises directory
# main_path = r'D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\exercises'

# Process all simulations and get list of feature-label pairs
# all_dataframes = process_all_simulations(main_path)












