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
    sampling_probability=0.0,
    is_training=True,
    trained_model=False,
    device="cpu",
):
    """
    Perform autoregressive training or validation on a specific sequence.

    Returns:
    - avg_loss: Average loss over the steps.
    - l2_norms: Dictionary containing L2 norms for the first column, second column, and the rest.
    """
    total_loss = 0
    predictions = []
    l2_norms = {"col_1": 0, "col_2": 0, "rest": 0}

    initial_input = initial_input.to(device)
    target_data = target_data.to(device)

    current_input = initial_input.view(1, -1)  # Flatten for input

    for step in range(n_steps):
        next_pred = model(current_input)  # Forward pass

        # Loss
        loss = criterion(next_pred, target_data[step].unsqueeze(0))
        total_loss += loss.item()

        # Backpropagation if training
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate L2 norms
            l2_norms["col_1"] += pt.norm(next_pred[:, 0] - target_data[step, 0]) ** 2
            l2_norms["col_2"] += pt.norm(next_pred[:, 1] - target_data[step, 1]) ** 2
            l2_norms["rest"] += pt.norm(next_pred[:, 2:] - target_data[step, 2:]) ** 2

        # Store predictions for trained_model=True
        if trained_model:
            predictions.append(next_pred.detach().cpu().numpy())

        # Scheduled Sampling
        use_prediction = np.random.rand() < sampling_probability
        next_feature = next_pred if use_prediction else target_data[step].unsqueeze(0)

        # Add rotational speed
        rotational_speed = pt.tensor(
            [rotational_speed_list[sequence_length + step, 0]],
            dtype=pt.float32,
            device=device,
        ).unsqueeze(0)
        next_feature = pt.cat((next_feature, rotational_speed), dim=1)

        # Update sequence
        initial_input = pt.cat(
            (initial_input[:, 1:], next_feature.view(1, 1, -1)), dim=1
        )
        current_input = initial_input.view(1, -1)

    avg_loss = total_loss / n_steps
    if is_training:
        l2_norms = {k: (v / n_steps).sqrt().item() for k, v in l2_norms.items()}

    if trained_model:
        return avg_loss, predictions
    else:
        return avg_loss, l2_norms
