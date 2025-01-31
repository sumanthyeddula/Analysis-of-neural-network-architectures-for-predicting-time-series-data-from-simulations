import os
import pandas as pd
import numpy as np
from typing import Tuple, List
from utils import split_dataset, normalize_column_data


def extract_force_coefficients(filepath: str) -> pd.DataFrame:
    """Extracts only the drag (Cd) and lift (Cl) coefficients from a specified file, excluding time."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    # Skip initial non-data lines and select Cd, Cl columns
    force_df = pd.read_csv(
        filepath, delim_whitespace=True, skiprows=13, usecols=[1, 2], names=["Cd", "Cl"]
    )

    return force_df


def extract_pressure_probes(filepath: str) -> pd.DataFrame:
    """Extracts only pressure values from probe file, excluding time."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")

    # Load numeric pressure data while handling initial non-numeric metadata rows
    pressure_data_rows = []
    with open(filepath, "r") as file:
        for line in file:
            try:
                values = [float(x) for x in line.split()]
                pressure_data_rows.append(
                    values[1:13]
                )  # Extracting only 12 probe columns
            except ValueError:
                continue  # Skip non-numeric rows

    # Create DataFrame with 12 pressure probe columns
    probe_df = pd.DataFrame(
        pressure_data_rows, columns=[f"p_probe_{i}" for i in range(12)]
    )
    return probe_df


def extract_actuation_parameters(filepath: str) -> Tuple[float, float]:
    """Extracts amplitude and frequency from the U file, ignoring expression lines."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")

    amplitude, frequency = None, None
    with open(filepath, "r") as file:
        for line in file:
            if "amplitude" in line and "valueExpr" not in line:
                try:
                    amplitude = float(line.split()[-1].strip(";"))
                except ValueError:
                    print(
                        f"Warning: Unable to parse amplitude value in line: {line.strip()}"
                    )
            elif "frequency" in line and "valueExpr" not in line:
                try:
                    frequency = float(line.split()[-1].strip(";"))
                except ValueError:
                    print(
                        f"Warning: Unable to parse frequency value in line: {line.strip()}"
                    )

    if amplitude is None or frequency is None:
        raise ValueError(
            f"Amplitude or frequency not found or invalid in file: {filepath}"
        )

    return amplitude, frequency


def calculate_rotational_speed(
    amplitude: float, frequency: float, times: np.ndarray
) -> pd.DataFrame:

    # Calculate results for all time steps
    rotational_speed_magnitudes = []
    for t in times:
        # Sinusoidal modulation
        rotational_speed = amplitude * np.sin(2 * np.pi * frequency * t)
        # rotational_speed_magnitude = np.linalg.norm(rotational_speed)
        rotational_speed_magnitudes.append(rotational_speed)
    # print("rotational speed maginitude", rotational_speed_magnitudes)

    return pd.DataFrame({"rotational_speed": rotational_speed_magnitudes})


def create_simulation_dataframe(sim_dir: str, dt: float) -> pd.DataFrame:
    """Creates a DataFrame for a specific simulation folder by combining base data and new data."""
    # Define paths using the specific simulation directory
    force_path = os.path.join(
        sim_dir, "postProcessing", "forces", "0", "coefficient.dat"
    )
    probe_path = os.path.join(sim_dir, "postProcessing", "probes", "0", "p")
    actuation_path = os.path.join(sim_dir, "0.org", "U")

    # Log paths being used to verify unique folders are accessed
    # print(f"\nReading data from: {sim_dir}")
    # print(f"Force data path: {force_path}")
    # print(f"Probe data path: {probe_path}")
    # print(f"Actuation data path: {actuation_path}")

    # Re-extract amplitude and frequency for this specific simulation
    amplitude, frequency = extract_actuation_parameters(actuation_path)

    # Extract force and probe data
    sim_force_df = extract_force_coefficients(force_path)
    sim_probe_df = extract_pressure_probes(probe_path)

    # Time array for simulation data (assuming time steps correspond to data rows)
    num_rows = sim_force_df.shape[0]
    # print(num_rows)
    times = np.arange(0.01, (num_rows + 1) * dt, dt)

    # Calculate rotational speeds for simulation data using simulation's amplitude and frequency
    rotational_speeds_df = calculate_rotational_speed(amplitude, frequency, times)

    amplitude = pd.DataFrame({"amplitude": [amplitude] * num_rows})
    frequency = pd.DataFrame({"frequency": [frequency] * num_rows})

    # Concatenate force, probe, and rotational speed data
    final_df = pd.concat(
        [
            sim_force_df.reset_index(drop=True),
            sim_probe_df.reset_index(drop=True),
            rotational_speeds_df,
            # amplitude,
            # frequency,
        ],
        axis=1,
    )

    # Combine base and simulation data
    # final_df = pd.concat([base_df, sim_combined_df], ignore_index=True)
    # final_df = sim_combined_df

    # Select only the required columns that exist in final_df
    required_columns = (
        ["Cd", "Cl"]
        + list(sim_probe_df.columns)
        + ["rotational_speed"]
        # + ["amplitude"]
        # + ["frequency"]
    )
    final_df = final_df[required_columns]

    # print(final_df.head())  # Display first few rows of the combined DataFrame

    return final_df


def process_all_simulations(
    base_path: str,
    dt: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Processes all simulations in the exercises directory, combining base and simulation data."""

    # Initialize list to hold all DataFrames for each simulation
    dataframes = []

    for root, dirs, files in os.walk(base_path):
        # for directory in dirs:
        #     print(directory)
        for directory in dirs:
            sim_dir = os.path.join(base_path, directory)

            # print(f"Processing simulation folder: {sim_dir}")

            sim_df = create_simulation_dataframe(sim_dir, dt)

            # Extract features and labels
            features = sim_df.values  # All columns as features
            labels = sim_df.drop(
                columns=["rotational_speed"]
            ).values  # Exclude 'rotational_speed' as labels

            dataframes.append(sim_df.values)  # Store feature-label pairs as tuples
        break

    new_dataframes = []

    for i in range(len(dataframes)):
        new_data = dataframes[i][400:1001]
        new_dataframes.append(new_data)

    return new_dataframes


# # Define the main path for the exercises directory
# main_path = r"D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\RE_100"

# # # Process all simulations and get list of feature-label pairs
# train = process_all_simulations(main_path, train=True, test_size=0.2, dt=0.01)


# print(train[0].shape)  # Display shape of features for first simulation
