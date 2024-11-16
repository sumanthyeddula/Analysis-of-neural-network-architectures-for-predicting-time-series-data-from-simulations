
import torch as pt

# def autoregressive_func(model, initial_input, target_data, n_steps, optimizer, criterion, rotational_speed_list, sequence_length, is_training=True):
#     """
#     Perform autoregressive training or validation on a specific sequence.

#     :param model: The FCNNModel instance to be trained or evaluated.
#     :param initial_input: Initial input tensor of shape [1, sequence_length, n_features].
#     :param target_data: Target output tensor of shape [n_steps, n_outputs].
#     :param n_steps: Number of future time steps to predict.
#     :param optimizer: Optimizer for training (ignored during validation).
#     :param criterion: Loss function.
#     :param rotational_speed_list: The rotational speed DataFrame.
#     :param sequence_length: Length of the input sequence window.
#     :param is_training: Whether the function is being used for training or validation.
#     :return: Average loss for the session.
#     """
#     total_loss = 0

#     # Flatten the initial input for the model
#     current_input = initial_input.reshape(1, -1)  # Shape [1, sequence_length * n_features]

#     for step in range(n_steps):
#         # Perform forward pass to predict the next time step
#         next_pred = model(current_input)  # Shape [1, n_outputs]

#         # Calculate the loss between predicted output and the actual next step in target_data
#         loss = criterion(next_pred, target_data[step].unsqueeze(0))
#         total_loss += loss.item()

#         if is_training:
#             # Backpropagate and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # Detach the next_pred to prevent graph accumulation
#         next_pred = next_pred.detach()

#         # Extract rotational speed from the sequence for the autoregressive window update
#         rotational_speed = rotational_speed_list.iloc[sequence_length + step, 0]
#         rotational_speed = pt.tensor([rotational_speed], dtype=pt.float32).unsqueeze(0)  # Shape [1, 1]

#         # Concatenate the prediction and rotational_speed to form the next input
#         next_feature = pt.cat((next_pred, rotational_speed), dim=1)  # Shape [1, n_features]

#         # Reshape for sliding window update
#         next_feature = next_feature.view(1, 1, -1)  # Shape [1, 1, n_features]

#         # Update initial input sequence by sliding window to include next feature and remove oldest step
#         initial_input = pt.cat((initial_input[:, 1:], next_feature), dim=1)  # Shape [1, sequence_length, n_features]

#         # Flatten the updated sequence for the next prediction
#         current_input = initial_input.view(1, -1)  # Shape [1, sequence_length * n_features]

#     avg_loss = total_loss / n_steps
#     return avg_loss


import torch as pt

def autoregressive_func(
    model, initial_input, target_data, n_steps, optimizer, criterion, rotational_speed_list, sequence_length, is_training=True, trained_model=False
):
    """
    Perform autoregressive training or validation on a specific sequence.

    :param model: The FCNNModel instance to be trained or evaluated.
    :param initial_input: Initial input tensor of shape [1, sequence_length, n_features].
    :param target_data: Target output tensor of shape [n_steps, n_outputs].
    :param n_steps: Number of future time steps to predict.
    :param optimizer: Optimizer for training (ignored during validation).
    :param criterion: Loss function.
    :param rotational_speed_list: The rotational speed DataFrame.
    :param sequence_length: Length of the input sequence window.
    :param is_training: Whether the function is being used for training or validation.
    :param trained_model: If True, returns predictions for external dataset.
    :return: Average loss for the session and optionally the predicted values.
    """
    total_loss = 0
    predictions = []  # Store predictions for trained_model=True

    # Flatten the initial input for the model
    current_input = initial_input.reshape(1, -1)  # Shape [1, sequence_length * n_features]

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

        # Extract rotational speed from the sequence for the autoregressive window update
        rotational_speed = rotational_speed_list.iloc[sequence_length + step, 0]
        rotational_speed = pt.tensor([rotational_speed], dtype=pt.float32).unsqueeze(0)  # Shape [1, 1]

        # Concatenate the prediction and rotational_speed to form the next input
        next_feature = pt.cat((next_pred, rotational_speed), dim=1)  # Shape [1, n_features]

        # Reshape for sliding window update
        next_feature = next_feature.view(1, 1, -1)  # Shape [1, 1, n_features]

        # Update initial input sequence by sliding window to include next feature and remove oldest step
        initial_input = pt.cat((initial_input[:, 1:], next_feature), dim=1)  # Shape [1, sequence_length, n_features]

        # Flatten the updated sequence for the next prediction
        current_input = initial_input.view(1, -1)  # Shape [1, sequence_length * n_features]

    avg_loss = total_loss / n_steps

    # Return average loss and optionally predictions
    if trained_model:
        return avg_loss, predictions
    else:
        return avg_loss
