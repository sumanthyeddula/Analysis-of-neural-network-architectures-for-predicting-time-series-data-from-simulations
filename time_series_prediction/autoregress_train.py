import torch as pt
import numpy as np
import pandas as pd

from autoregress_func import autoregressive_func
from sklearn.preprocessing import MinMaxScaler
from utils import save_model
from plots import plot_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau


def test(
    model,
    test_Data,
    sequence_length,
    n_steps,
    n_features,
    criterion,
    batch_size=1,
    test_single=True,
):
    """
    Test the trained model on a single or multiple test datasets.

    :param model: The trained FCNNModel instance.
    :param test_Data: The test dataset(s).
    :param sequence_length: Length of the input sequence window.
    :param n_steps: Number of future time steps to predict.
    :param n_features: Number of input features.
    :param criterion: Loss function for evaluation.
    :param batch_size: Batch size for testing (only applicable for multiple datasets).
    :param test_single: If True, test on a single dataset; otherwise, test on multiple datasets.
    :return: Test loss and predictions.
    """
    print("Testing the model...")
    model.eval()  # Set the model to evaluation mode

    total_loss = 0
    all_predictions = []
    all_actuals = []

    with pt.no_grad():  # No gradient computation during testing
        if test_single:
            # Test on a single dataset
            dataframe = test_Data

            # Prepare input and target data
            initial_input = dataframe[0:sequence_length]
            target_data = dataframe[sequence_length : sequence_length + n_steps, :14]

            # Extract the last column for rotational speed
            rotational_Speed = dataframe[:, -1]
            rotational_Speed = pd.DataFrame(rotational_Speed, columns=["last_column"])

            # Normalize the input

            # scaler = MinMaxScaler()
            # normalized_input = scaler.fit_transform(initial_input)
            normalized_input_tensor = pt.tensor(initial_input, dtype=pt.float32).view(
                1, sequence_length, n_features
            )

            # Convert target data to PyTorch tensor
            target_data_tensor = pt.tensor(target_data, dtype=pt.float32)

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
            )

            all_predictions.append(predictions)
            all_actuals.append(target_data_tensor.numpy())
            total_loss += avg_loss

        else:
            # Test on multiple datasets
            for batch_start in range(0, len(test_Data), batch_size):
                batch = test_Data[batch_start : batch_start + batch_size]
                batch_loss = 0
                batch_predictions = []
                batch_actuals = []

                for sequence in batch:
                    dataframe = sequence[0]

                    # Ensure the sequence has the correct shape and length
                    if (
                        dataframe.shape[1] != n_features
                        or dataframe.shape[0] < sequence_length + n_steps
                    ):
                        print(
                            f"Skipping sequence due to unexpected shape or insufficient length."
                        )
                        continue

                    # Prepare input and target data
                    initial_input = dataframe[0:sequence_length]
                    target_data = dataframe[
                        sequence_length : sequence_length + n_steps, :14
                    ]

                    # # Normalize the input
                    # scaler = MinMaxScaler()
                    # normalized_input = scaler.fit_transform(initial_input)
                    normalized_input_tensor = pt.tensor(
                        initial_input, dtype=pt.float32
                    ).view(1, sequence_length, n_features)

                    # Convert target data to PyTorch tensor
                    target_data_tensor = pt.tensor(target_data, dtype=pt.float32)

                    # Perform autoregressive predictions
                    _, predictions = autoregressive_func(
                        model,
                        normalized_input_tensor,
                        target_data_tensor,
                        n_steps,
                        optimizer=None,
                        criterion=criterion,
                        rotational_speed_list=dataframe[:, -1],
                        sequence_length=sequence_length,
                        is_training=False,
                        trained_model=True,
                    )

                    predictions_tensor = pt.tensor(predictions).view(
                        -1, target_data.shape[1]
                    )
                    loss = criterion(predictions_tensor, target_data_tensor)
                    batch_loss += loss.item()

                    batch_predictions.append(predictions_tensor.numpy())
                    batch_actuals.append(target_data)

                total_loss += batch_loss / len(batch)
                all_predictions.extend(batch_predictions)
                all_actuals.extend(batch_actuals)

    avg_test_loss = total_loss / (1 if test_single else (len(test_Data) // batch_size))

    return avg_test_loss, all_predictions, all_actuals


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
    patience,
    shuffle=True,
    start_sampling_prob=0.0,  # Start probability of using model's predictions
    end_sampling_prob=1.0,  # Final probability
    sampling_schedule_type="linear",  # Type of schedule: "linear" or "exponential"
):
    print("********************************************************* Code starts")

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    early_stop_counter = 0

    # Initialize the scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    for epoch in range(n_epochs):
        train_loss = 0
        val_loss = 0

        # Shuffle training data for each epoch
        if shuffle:
            np.random.shuffle(train_Data_preprocessed)

        # **Calculate Sampling Probability**
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

        model.train()
        for batch_start in range(0, len(train_Data_preprocessed), batch_size):
            batch = train_Data_preprocessed[batch_start : batch_start + batch_size]
            batch_loss = 0

            for normalized_input, target_data, rotational_speed in batch:
                # Convert normalized input and target to PyTorch tensors
                normalized_input_tensor = pt.tensor(
                    normalized_input, dtype=pt.float32
                ).view(1, sequence_length, n_features)
                target_data_tensor = pt.tensor(target_data, dtype=pt.float32)
                rotational_speed_tensor = pd.DataFrame(
                    rotational_speed, columns=["last_column"]
                )

                # Train the model on the current sequence with scheduled sampling
                avg_loss = autoregressive_func(
                    model,
                    normalized_input_tensor,
                    target_data_tensor,
                    n_steps,
                    optimizer,
                    criterion,
                    rotational_speed_tensor,
                    sequence_length,
                    sampling_probability=sampling_probability,  # Pass the dynamic probability
                    is_training=True,
                    trained_model=False,
                )
                batch_loss += avg_loss

            train_loss += batch_loss / len(batch)

        # Validation phase
        model.eval()
        with pt.no_grad():
            for (
                normalized_input,
                target_data,
                rotational_speed,
            ) in val_Data_preprocessed:
                normalized_input_tensor = pt.tensor(
                    normalized_input, dtype=pt.float32
                ).view(1, sequence_length, n_features)
                target_data_tensor = pt.tensor(target_data, dtype=pt.float32)
                rotational_speed_tensor = pd.DataFrame(
                    rotational_speed, columns=["last_column"]
                )

                avg_loss = autoregressive_func(
                    model,
                    normalized_input_tensor,
                    target_data_tensor,
                    n_steps,
                    optimizer=None,
                    criterion=criterion,
                    rotational_speed_list=rotational_speed_tensor,
                    sequence_length=sequence_length,
                    is_training=False,
                    trained_model=False,
                )
                val_loss += avg_loss

        train_loss /= len(train_Data_preprocessed)
        val_loss /= len(val_Data_preprocessed)

        # Update the learning rate
        scheduler.step(val_loss)

        current_lr = scheduler.get_last_lr()[
            0
        ]  # Access the last learning rate from the scheduler
        print(
            f"Epoch [{epoch+1}/{n_epochs}] -> Sampling Probability: {sampling_probability:.4f}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        )

        # Early stopping logic remains the same
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_model(model, f"{save_path}_best.pth")
            print(f"Validation loss improved to {val_loss:.4f}. Model saved.")
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} out of {patience}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        plot_loss(train_losses, val_losses, f"{save_path}_loss_plot.png")

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses
