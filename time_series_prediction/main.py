import torch as pt
from FCNN import FCNNModel
from LSTM import LSTMModel

from autoregress_train import train, test
from utils import (
    set_seed,
    split_dataset,
    normalize_column_data,
    denormalize_target_data,
    renormalize_data_column_wise,
)
from hyperparameter_tuning import hyperparameter_tuning


from plots import (
    plot_loss,
    plot_dataframe_columns_combined,
    compute_scaled_l2_loss_scatter,
)
import numpy as np
import optuna.visualization as vis

from hyperparameter_tuning import hyperparameter_tuning


# Example of training loop over multiple epochs and sequences
if __name__ == "__main__":

    test_mode = False
    hyperparameter_tune = False

    # Set model type
    model = FCNNModel
    # Set model parameters
    n_features = 15
    n_outputs = 14
    n_layers = 5
    n_neurons = 128
    sequence_length = 5
    n_steps = 600 - sequence_length

    # Set training parameters
    n_epochs = 100
    learning_rate = 0.005
    batch_Size = 1
    save_path = "./lstm_test"
    early_stopping_patience = 50
    test_size = 0.25

    set_seed(40)

    # Hyperparameter tuning parameters
    n_trails = 15

    # test model path
    model_path = "./fcnn_test_best.pth"

    if test_mode:
        # Load and prepare training data
        from Data_pipeline import process_all_simulations

        main_path = r"D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\RE_100"
        all_dataframes = process_all_simulations(
            main_path, train=False, test_size=test_size, dt=0.01
        )

        actual_data = all_dataframes[2]

        # data = renormalize_data_column_wise(actual_data)

        if model is FCNNModel:
            model = FCNNModel(
                n_outputs=n_outputs,
                n_layers=n_layers,
                n_neurons=n_neurons,
                sequence_length=sequence_length,
                n_features=n_features,
            )

        # Load the state dictionary
        # model.load_state_dict(pt.load(model_path))

        checkpoint = pt.load(model_path)

        # Check if it's a state_dict or a checkpoint
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        criterion = pt.nn.MSELoss()

        avg_test_loss, all_predictions, all_actuals = test(
            model=model,
            n_steps=n_steps,
            n_features=n_features,
            test_Data=actual_data,
            sequence_length=sequence_length,
            criterion=criterion,
            test_single=True,
        )

        all_predictions = denormalize_target_data(all_predictions)

        # all_actuals = actual_data[sequence_length : sequence_length + n_steps, :14]

        print(f"Average test loss: {avg_test_loss:.4f}")
        print(f"Predictions shape: {all_predictions}")
        print(f"Actuals shape: {all_actuals}")

        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        all_predictions = all_predictions.reshape(n_steps, 14)
        all_actuals = all_actuals.reshape(n_steps, 14)

        # Plot the predictions and actuals
        plot_dataframe_columns_combined(
            all_actuals, all_predictions, save_dir="fcnn_test"
        )
        # plot_dataframe_columns_heatmap(all_actuals, all_predictions, save_dir="test")
        # compute_scaled_l2_loss_heatmap(all_actuals, all_predictions, save_dir="test1")
        compute_scaled_l2_loss_scatter(
            all_actuals, all_predictions, save_dir="fcnn_test"
        )

        exit(0)

    else:
        # Load and prepare training data
        from Data_pipeline import process_all_simulations

        main_path = r"D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\RE_100"
        all_dataframes = process_all_simulations(
            main_path, train=True, test_size=test_size, dt=0.01
        )

        # Instantiate the model
        if model is FCNNModel:
            model = FCNNModel(
                n_outputs=n_outputs,
                n_layers=n_layers,
                n_neurons=n_neurons,
                sequence_length=sequence_length,
                n_features=n_features,
            )
        elif model is LSTMModel:
            model = LSTMModel(
                n_features=n_features,
                hidden_size=n_neurons,
                num_layers=n_layers,
                n_outputs=n_outputs,
                sequence_length=sequence_length,
            )

        optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = pt.nn.MSELoss()
        # all_dataframes = normalize_data(
        #     all_dataframes, sequence_length, n_steps, n_features
        # )

        all_dataframes = normalize_column_data(
            all_dataframes, sequence_length, n_steps, n_features
        )

        # all_dataframes = data_rearrange(
        #     train_data, sequence_length, n_steps, n_features
        # )
        train_data, val_data = split_dataset(all_dataframes)

        if hyperparameter_tune:
            fcnn_study = hyperparameter_tuning(
                "FCNN",
                train_data,
                val_data,
                n_outputs,
                sequence_length,
                n_features,
                n_steps,
                n_trails=n_trails,
            )

            print(f"Best trial: {fcnn_study.best_trial.number}")
            print(f"Best parameters: {fcnn_study.best_trial.params}")
            print(f"Best value: {fcnn_study.best_value}")

            # Visualize the study
            vis.plot_optimization_history(fcnn_study).show()
            vis.plot_param_importances(fcnn_study).show()
            vis.plot_slice(fcnn_study).show()
            vis.plot_parallel_coordinate(fcnn_study).show()
            vis.plot_contour(fcnn_study).show()
            vis.plot_edf(fcnn_study).show()
            vis.plot_intermediate_values(fcnn_study).show()

            lstm_study = hyperparameter_tuning(
                "LSTM",
                train_data,
                val_data,
                n_outputs,
                sequence_length,
                n_features,
                n_steps,
                n_trails=n_trails,
            )

            print(f"Best trial: {lstm_study.best_trial.number}")
            print(f"Best parameters: {lstm_study.best_trial.params}")
            print(f"Best value: {lstm_study.best_value}")

            # Visualize the study
            vis.plot_optimization_history(lstm_study).show()
            vis.plot_param_importances(lstm_study).show()
            vis.plot_slice(lstm_study).show()
            vis.plot_parallel_coordinate(lstm_study).show()
            vis.plot_contour(lstm_study).show()
            vis.plot_edf(lstm_study).show()

            print("Hyperparameter tuning complete.")
            exit(0)

        train_losses, val_losses = train(
            model,
            n_epochs,
            n_steps,
            n_features,
            train_data,
            val_data,
            sequence_length,
            optimizer,
            criterion,
            batch_size=batch_Size,
            save_path=save_path,
            patience=early_stopping_patience,
            shuffle=True,
            start_sampling_prob=0.2,
            sampling_schedule_type="linear",
        )

        # Plot the losses after training
        plot_loss(train_losses, val_losses, f"{save_path}_loss_plot.png")

        print("Training complete.")
