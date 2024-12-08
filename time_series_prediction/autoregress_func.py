import torch as pt
import numpy as np


def autoregressive_func(
    model,
    initial_input,
    target_data,
    n_steps,
    optimizer,
    criterion,
    rotational_speed_list,
    sequence_length,
    sampling_probability=0.0,  # Probability of using model's prediction
    is_training=True,
    trained_model=False,
):
    """
    Perform autoregressive training or validation on a specific sequence with scheduled sampling.

    :param sampling_probability: Probability of using the model's prediction instead of ground truth.
    (Other parameters remain unchanged.)
    """
    total_loss = 0
    predictions = []  # Store predictions for trained_model=True

    # Flatten the initial input for the model
    current_input = initial_input.reshape(
        1, -1
    )  # Shape [1, sequence_length * n_features]

    for step in range(n_steps):
        # Perform forward pass to predict the next time step
        next_pred = model(current_input)  # Shape [1, n_outputs]

        # Store predictions only if trained_model is True
        if trained_model:
            predictions.append(next_pred.detach().cpu().numpy())

        # Calculate the loss between predicted output and the actual next step in target_data
        loss = criterion(next_pred, target_data[step].unsqueeze(0))
        total_loss += loss.item()

        if is_training:
            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Detach the next_pred to prevent graph accumulation
        next_pred = next_pred.detach()

        # **Scheduled Sampling**: Choose between ground truth and model prediction
        use_prediction = np.random.rand() < sampling_probability  # Sample decision
        if use_prediction:
            next_feature = next_pred  # Use model's prediction
        else:
            next_feature = target_data[step].unsqueeze(0)  # Use ground truth

        # Extract rotational speed from the sequence for the autoregressive window update
        rotational_speed = rotational_speed_list.iloc[sequence_length + step, 0]
        rotational_speed = pt.tensor([rotational_speed], dtype=pt.float32).unsqueeze(
            0
        )  # Shape [1, 1]

        # Concatenate the chosen feature (prediction or ground truth) and rotational speed
        next_feature = pt.cat(
            (next_feature, rotational_speed), dim=1
        )  # Shape [1, n_features]

        # Reshape for sliding window update
        next_feature = next_feature.view(1, 1, -1)  # Shape [1, 1, n_features]

        # Update initial input sequence by sliding window to include next feature and remove oldest step
        initial_input = pt.cat(
            (initial_input[:, 1:], next_feature), dim=1
        )  # Shape [1, sequence_length, n_features]

        # Flatten the updated sequence for the next prediction
        current_input = initial_input.view(
            1, -1
        )  # Shape [1, sequence_length * n_features]

    avg_loss = total_loss / n_steps

    # Return average loss and optionally predictions
    if trained_model:
        return avg_loss, predictions
    else:
        return avg_loss
