import torch as pt
import numpy as np
import pandas as pd

from autoregress_func import autoregressive_func
from sklearn.preprocessing import MinMaxScaler
from utils import save_model, calculate_prediction_accuracy, denormalize_target_data
from plots import plot_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time


def test(
    model,
    test_Data,
    sequence_length,
    n_steps,
    n_features,
    criterion,
    batch_size=1,
    test_single=True,
    device="cpu",  # Specify device for testing
):
    """
    Test the trained model on a single or multiple test datasets.

    :param model: The trained model instance.
    :param test_Data: The test dataset(s).
    :param sequence_length: Length of the input sequence window.
    :param n_steps: Number of future time steps to predict.
    :param n_features: Number of input features.
    :param criterion: Loss function for evaluation.
    :param batch_size: Batch size for testing (only applicable for multiple datasets).
    :param test_single: If True, test on a single dataset; otherwise, test on multiple datasets.
    :param device: The device to run the testing on ("cpu" or "cuda").
    :return: Test loss and predictions.
    """
    print("Testing the model...")
    model.to(device)  # Move the model to the specified device
    model.eval()  # Set the model to evaluation mode

    total_loss = 0
    all_predictions = []
    all_actuals = []
    all_amplitudes = []
    all_frequency = []

    with pt.no_grad():  # No gradient computation during testing
        if test_single:
            # Test on a single dataset
            dataframe = test_Data

            # Prepare input and target data
            initial_input = dataframe[0:sequence_length]
            target_data = dataframe[sequence_length : sequence_length + n_steps, :14]

            # Extract the last column for rotational speed
            rotational_Speed = dataframe[:, -1]
            rotational_Speed = pd.DataFrame(
                rotational_Speed, columns=["last_column"]
            ).to_numpy()

            # Convert input and target data to PyTorch tensors on the specified device
            normalized_input_tensor = pt.tensor(
                initial_input, dtype=pt.float32, device=device
            ).view(1, sequence_length, n_features)
            target_data_tensor = pt.tensor(target_data, dtype=pt.float32, device=device)

            # Perform autoregressive predictions
            avg_loss, predictions = autoregressive_func(
                model,
                normalized_input_tensor,
                target_data_tensor,
                n_steps,
                optimizer=None,
                criterion=criterion,
                rotational_speed_list=rotational_Speed,
                sequence_length=sequence_length,
                is_training=False,
                trained_model=True,
                device=device,
            )

            all_predictions.append(predictions)
            all_actuals.append(
                target_data_tensor.cpu().numpy()
            )  # Move to CPU for output
            total_loss += avg_loss

        else:
            # Test on multiple datasets
            for batch_start in range(0, len(test_Data), batch_size):
                batch = test_Data[batch_start : batch_start + batch_size]
                batch_loss = 0
                batch_predictions = []
                batch_actuals = []

                for sequence in batch:
                    dataframe = sequence

                    # # Ensure the sequence has the correct shape and length
                    # if (
                    #     dataframe.shape[1] != n_features
                    #     or dataframe.shape[0] < sequence_length + n_steps
                    # ):
                    #     print(
                    #         f"Skipping sequence due to unexpected shape or insufficient length."
                    #     )
                    #     continue

                    # Prepare input and target data
                    initial_input = dataframe[0:sequence_length]
                    initial_input = initial_input[:, :15]
                    target_data = dataframe[
                        sequence_length : sequence_length + n_steps, :14
                    ]

                    # Convert input and target data to PyTorch tensors on the specified device
                    normalized_input_tensor = pt.tensor(
                        initial_input, dtype=pt.float32, device=device
                    ).view(1, sequence_length, n_features)
                    target_data_tensor = pt.tensor(
                        target_data, dtype=pt.float32, device=device
                    )

                    # Extract amplitude and frequency
                    frequency = dataframe[
                        sequence_length : sequence_length + n_steps, 15:16
                    ]
                    amplitude = dataframe[
                        sequence_length : sequence_length + n_steps, 16:17
                    ]

                    all_amplitudes.append(amplitude)
                    all_frequency.append(frequency)

                    # Extract the last column for rotational speed
                    rotational_Speed = dataframe[:, 14:15]
                    rotational_Speed = pd.DataFrame(
                        rotational_Speed, columns=["last_column"]
                    ).to_numpy()

                    # Perform autoregressive predictions
                    _, predictions = autoregressive_func(
                        model,
                        normalized_input_tensor,
                        target_data_tensor,
                        n_steps,
                        optimizer=None,
                        criterion=criterion,
                        rotational_speed_list=rotational_Speed,
                        sequence_length=sequence_length,
                        is_training=False,
                        trained_model=True,
                        device=device,
                    )

                    predictions_tensor = pt.tensor(predictions, device=device).view(
                        -1, target_data.shape[1]
                    )
                    loss = criterion(predictions_tensor, target_data_tensor)
                    batch_loss += loss.item()

                    batch_predictions.append(
                        predictions_tensor.cpu().numpy()
                    )  # Move to CPU for output
                    batch_actuals.append(target_data)

                total_loss += batch_loss / len(batch)
                all_predictions.extend(batch_predictions)
                all_actuals.extend(batch_actuals)

    avg_test_loss = total_loss / (1 if test_single else (len(test_Data) // batch_size))

    return avg_test_loss, all_predictions, all_actuals, all_amplitudes, all_frequency


def train(
    model,
    n_epochs,
    n_steps,
    n_features,
    train_Data_preprocessed,
    val_Data_preprocessed,
    sequence_length,
    optimizer,
    criterion,
    batch_size,
    save_path,
    model_name,
    patience,
    shuffle=True,
    start_sampling_prob=0.0,
    end_sampling_prob=1.0,
    sampling_schedule_type="linear",
    device="cpu",
):
    """
    Train the model with L2 norm error tracking for first column, second column, and rest of the columns.

    :param model: The model instance to train.
    :param n_epochs: Number of training epochs.
    :param n_steps: Number of prediction steps.
    :param n_features: Number of input features.
    :param train_Data_preprocessed: Preprocessed training data.
    :param val_Data_preprocessed: Preprocessed validation data.
    :param sequence_length: Length of the input sequence window.
    :param optimizer: Optimizer for model training.
    :param criterion: Loss function for model training.
    :param batch_size: Batch size for training.
    :param save_path: Path to save the best model.
    :param model_name: Name for the saved model file.
    :param patience: Early stopping patience.
    :param shuffle: Whether to shuffle training data each epoch.
    :param start_sampling_prob: Initial sampling probability for teacher forcing.
    :param end_sampling_prob: Final sampling probability for teacher forcing.
    :param sampling_schedule_type: Type of sampling schedule ("linear", "exponential", "constant").
    :param device: Device to run training on ("cpu" or "cuda").
    :return: L2 norm errors for training and validation (for three groups of columns), and train/val losses.
    """

    print("********************************************************* Code starts")

    # Move model to the specified device
    model.to(device)

    train_losses = []
    val_losses = []
    val_predictions = []
    # train_l2_norms = {"col_1": [], "col_2": [], "rest": []}
    # val_l2_norms = {"col_1": [], "col_2": [], "rest": []}

    best_val_loss = float("inf")
    early_stop_counter = 0

    # Initialize the scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = 0
        val_loss = 0
        # train_epoch_l2 = {"col_1": 0, "col_2": 0, "rest": 0}
        # val_epoch_l2 = {"col_1": 0, "col_2": 0, "rest": 0}

        # Shuffle training data for each epoch
        if shuffle:
            np.random.shuffle(train_Data_preprocessed)

        # Calculate Sampling Probability
        if sampling_schedule_type == "linear":
            sampling_probability = min(
                start_sampling_prob
                + (end_sampling_prob - start_sampling_prob) * (epoch / n_epochs),
                1.0,
            )
        elif sampling_schedule_type == "exponential":
            sampling_probability = start_sampling_prob * (
                end_sampling_prob / start_sampling_prob
            ) ** (epoch / n_epochs)
        elif sampling_schedule_type == "constant":
            sampling_probability = start_sampling_prob

        # Training phase
        model.train()
        for batch_start in range(0, len(train_Data_preprocessed), batch_size):
            batch = train_Data_preprocessed[batch_start : batch_start + batch_size]
            batch_loss = 0
            # batch_l2 = {"col_1": 0, "col_2": 0, "rest": 0}

            for normalized_input, target_data, rotational_speed in batch:
                normalized_input_tensor = pt.tensor(
                    normalized_input, dtype=pt.float32, device=device
                ).view(1, sequence_length, n_features)
                target_data_tensor = pt.tensor(
                    target_data, dtype=pt.float32, device=device
                )
                rotational_speed_tensor = pd.DataFrame(
                    rotational_speed, columns=["last_column"]
                ).to_numpy()

                # Train the model on the current sequence
                avg_loss = autoregressive_func(
                    model,
                    normalized_input_tensor,
                    target_data_tensor,
                    n_steps,
                    optimizer,
                    criterion,
                    rotational_speed_tensor,
                    sequence_length,
                    sampling_probability=sampling_probability,
                    is_training=True,
                    trained_model=False,
                    device=device,
                )
                batch_loss += avg_loss
                # batch_l2["col_1"] += l2_norms["col_1"]
                # batch_l2["col_2"] += l2_norms["col_2"]
                # batch_l2["rest"] += l2_norms["rest"]

            train_loss += batch_loss / len(batch)
            # train_epoch_l2["col_1"] += batch_l2["col_1"] / len(batch)
            # train_epoch_l2["col_2"] += batch_l2["col_2"] / len(batch)
            # train_epoch_l2["rest"] += batch_l2["rest"] / len(batch)

        # Validation phase
        model.eval()
        with pt.no_grad():
            for (
                normalized_input,
                target_data,
                rotational_speed,
            ) in val_Data_preprocessed:
                normalized_input_tensor = pt.tensor(
                    normalized_input, dtype=pt.float32, device=device
                ).view(1, sequence_length, n_features)
                target_data_tensor = pt.tensor(
                    target_data, dtype=pt.float32, device=device
                )
                rotational_speed_tensor = pd.DataFrame(
                    rotational_speed, columns=["last_column"]
                ).to_numpy()

                avg_loss, predictions = autoregressive_func(
                    model,
                    normalized_input_tensor,
                    target_data_tensor,
                    n_steps,
                    optimizer=None,
                    criterion=criterion,
                    rotational_speed_list=rotational_speed_tensor,
                    sequence_length=sequence_length,
                    is_training=False,
                    trained_model=True,
                    device=device,
                )
                val_loss += avg_loss
                val_predictions.append(predictions)
                # val_epoch_l2["col_1"] += l2_norms["col_1"]
                # val_epoch_l2["col_2"] += l2_norms["col_2"]
                # val_epoch_l2["rest"] += l2_norms["rest"]

        train_loss /= len(train_Data_preprocessed)
        val_loss /= len(val_Data_preprocessed)

        for i in range(len(val_predictions)):
            val_predictions[i] = denormalize_target_data(val_predictions[i])
        actual_data = []
        for i, (_, target_data, _) in enumerate(val_Data_preprocessed):
            actual_data.append(denormalize_target_data(target_data))

        accuracies = calculate_prediction_accuracy(actual_data, val_predictions)
        average_accuracy = sum(accuracies) / len(accuracies)

        # train_epoch_l2 = {
        #     k: v / len(train_Data_preprocessed) for k, v in train_epoch_l2.items()
        # }
        # val_epoch_l2 = {
        #     k: v / len(val_Data_preprocessed) for k, v in val_epoch_l2.items()
        # }

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # for key in train_l2_norms:
        #     train_l2_norms[key].append(train_epoch_l2[key])
        #     val_l2_norms[key].append(val_epoch_l2[key])

        # Update the learning rate
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{n_epochs}] -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            # f"Train L2: {train_epoch_l2}, Val L2: {val_epoch_l2}"
            f"Average Accuracy: {average_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_model(model, f"{save_path}", model_name)
        else:
            early_stop_counter += 1

        plot_loss(train_losses, val_losses, f"{save_path}", model_name)

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

        end_time = time.time()

        print(
            f"Epoch [{epoch+1}/{n_epochs}] completed in {end_time - start_time:.2f} seconds."
        )

    return train_losses, val_losses
