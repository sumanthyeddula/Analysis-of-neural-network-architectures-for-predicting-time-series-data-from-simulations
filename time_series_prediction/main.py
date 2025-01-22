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
    calculate_prediction_accuracy,
    calculate_prediction_accuracy_foreach_feature,
    calculate_prediction_accuracy_per_column,
)
from hyperparameter_tuning import hyperparameter_tuning
from plots import (
    plot_loss,
    plot_selected_columns,
    compute_scaled_l2_loss_scatter,
    compute_scaled_l2_loss_scatter,
    prediction_accuracy_with_error_bars_each_feature,
    plot_3d_frequency_amplitude_accuracy,
    plot_contour_frequency_amplitude_accuracy,
    plot_contour_accuracy_for_features_array,
    plot_optimization_results,
)
import numpy as np
import optuna.visualization as vis
from Data_pipeline import process_all_simulations

# Main execution block
if __name__ == "__main__":

    # Flags to toggle modes
    enable_testing_mode = True
    enable_hyperparameter_tuning = False

    # Model and architecture settings
    model_type = FCNNModel
    input_features = 15
    output_features = 14
    num_hidden_layers = 3
    hidden_layer_neurons = 155
    sequence_length = 5
    prediction_steps = 600 - sequence_length

    # Training parameters
    num_epochs = 2
    learning_rate = 0.0005
    batch_size = 30
    model_save_directory = "./FCNN_JAN"
    early_stopping_patience = 30
    data_split_test_ratio = 0.25
    initial_sampling_probability = 0.0
    sampling_strategy = "constant"

    # Seed configuration for reproducibility
    random_seeds = [53]

    # Hyperparameter tuning configuration
    num_trials = 5
    num_hyper_epochs = 2
    tuning_model_type = "FCNN"
    tuning_save_directory = "./fcnn_hyperparameter_tuning"

    # Paths for testing and data
    pretrained_model_directory = "./FCNN_JAN/"
    enable_test_single_sequence = False
    test_results_save_directory = "FCNN_JAN"

    # Device selection
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data directory path
    simulation_data_directory = r"D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\RE_100"

    # Load and preprocess all simulations
    all_simulations_dataframes = process_all_simulations(
        simulation_data_directory, dt=0.01
    )

    if enable_testing_mode:
        # Split data for training and testing
        training_data, testing_data = split_dataset(
            all_simulations_dataframes, test_size=data_split_test_ratio
        )

        print("Testing :", testing_data)

        seed_accuracies = []

        for seed in random_seeds:

            if enable_test_single_sequence:
                # Process single training sequence
                testing_data = renormalize_data_column_wise(testing_data[0])
            else:
                # Process the full test dataset
                testing_data = [
                    renormalize_data_column_wise(df)
                    for df in all_simulations_dataframes
                ]

            # Initialize model based on type
            if model_type is FCNNModel:
                model = FCNNModel(
                    n_outputs=output_features,
                    n_layers=num_hidden_layers,
                    n_neurons=hidden_layer_neurons,
                    sequence_length=sequence_length,
                    n_features=input_features,
                )
            elif model_type is LSTMModel:
                model = LSTMModel(
                    n_features=input_features,
                    hidden_size=hidden_layer_neurons,
                    num_layers=num_hidden_layers,
                    n_outputs=output_features,
                    sequence_length=sequence_length,
                )

            # Load pretrained model
            model_path = f"{pretrained_model_directory}{seed}_best_model.pth"
            checkpoint = pt.load(model_path)

            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

            # Loss function for evaluation
            loss_criterion = pt.nn.MSELoss()

            # Test the model
            average_test_loss, predictions, ground_truth, frequency, amplitude = test(
                model=model,
                n_steps=prediction_steps,
                n_features=input_features,
                test_Data=testing_data,
                sequence_length=sequence_length,
                criterion=loss_criterion,
                test_single=enable_test_single_sequence,
            )

            # Post-process predictions and evaluate accuracy
            if enable_test_single_sequence:
                predictions = denormalize_target_data(predictions)
                ground_truth = denormalize_target_data(ground_truth)

                predictions = np.array(predictions).reshape(prediction_steps, 14)
                ground_truth = np.array(ground_truth).reshape(prediction_steps, 14)

                print(f"Average test loss: {average_test_loss:.4f}")

                initial_inputs = training_data[0][:sequence_length, :-1]
                predictions = np.vstack((initial_inputs, predictions))
                ground_truth = np.vstack((initial_inputs, ground_truth))

                plot_selected_columns(
                    ground_truth,
                    predictions,
                    save_dir=test_results_save_directory,
                    plot_name=seed,
                )

                compute_scaled_l2_loss_scatter(
                    ground_truth,
                    predictions,
                    save_dir=test_results_save_directory,
                    plot_name=seed,
                )

            else:
                predictions = [denormalize_target_data(pred) for pred in predictions]
                ground_truth = [denormalize_target_data(gt) for gt in ground_truth]
                # extract frequency and amplitude

                frequency = [f[0][0] for f in frequency]
                amplitude = [a[0][0] for a in amplitude]

                cd_accuracy = calculate_prediction_accuracy_foreach_feature(
                    predictions, ground_truth, column=0
                )

                accuracies_for_each_feature = calculate_prediction_accuracy_per_column(
                    predictions, ground_truth
                )

                accuracies = calculate_prediction_accuracy(ground_truth, predictions)

                # plot_3d_frequency_amplitude_accuracy(frequency, amplitude, cd_accuracy)

                plot_contour_frequency_amplitude_accuracy(
                    frequency, amplitude, cd_accuracy
                )

                plot_contour_accuracy_for_features_array(
                    frequency,
                    amplitude,
                    accuracies_for_each_feature,
                    save_dir=test_results_save_directory,
                    plot_name=seed,
                )

                # compute_scaled_l2_loss_scatter(
                #     predictions, ground_truth, save_dir=test_results_save_directory
                # )

                prediction_accuracy_with_error_bars_each_feature(
                    predictions, ground_truth, save_dir=test_results_save_directory
                )

                seed_accuracies.append(sum(accuracies) / len(accuracies))

                print(f"Accuracies for seed {seed}: {accuracies}")

        print("Average accuracies across seeds:", seed_accuracies)

    else:
        # Normalize and prepare data for training
        normalized_data = normalize_column_data(
            all_simulations_dataframes,
            sequence_length,
            prediction_steps,
            input_features,
        )
        train_data, validation_data = split_dataset(
            normalized_data, test_size=data_split_test_ratio
        )

        for seed in random_seeds:
            print(f"Training with seed: {seed}")
            set_seed(seed)

            # Instantiate the model
            if model_type is FCNNModel:
                model = FCNNModel(
                    n_outputs=output_features,
                    n_layers=num_hidden_layers,
                    n_neurons=hidden_layer_neurons,
                    sequence_length=sequence_length,
                    n_features=input_features,
                )
            elif model_type is LSTMModel:
                model = LSTMModel(
                    n_features=input_features,
                    hidden_size=hidden_layer_neurons,
                    num_layers=num_hidden_layers,
                    n_outputs=output_features,
                    sequence_length=sequence_length,
                )

            optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
            loss_criterion = pt.nn.MSELoss()

            if enable_hyperparameter_tuning:
                tuning_results = hyperparameter_tuning(
                    best_model_type=tuning_model_type,
                    train_data=train_data,
                    val_data=validation_data,
                    n_outputs=output_features,
                    sequence_length=sequence_length,
                    n_features=input_features,
                    n_steps=prediction_steps,
                    n_trails=num_trials,
                    device=device,
                    save_path=tuning_save_directory,
                    seed=seed,
                    patience=early_stopping_patience,
                    n_epochs=num_hyper_epochs,
                )

                print(f"Best hyperparameters: {tuning_results.best_trial.params}")

                # Plot optimization results
                plot_optimization_results(tuning_results, tuning_model_type)

                continue

            # Train the model
            training_loss, validation_loss = train(
                model=model,
                n_epochs=num_epochs,
                n_steps=prediction_steps,
                n_features=input_features,
                train_Data_preprocessed=train_data,
                val_Data_preprocessed=validation_data,
                sequence_length=sequence_length,
                optimizer=optimizer,
                criterion=loss_criterion,
                batch_size=batch_size,
                save_path=model_save_directory,
                model_name=str(seed),
                patience=early_stopping_patience,
                shuffle=True,
                start_sampling_prob=initial_sampling_probability,
                sampling_schedule_type=sampling_strategy,
                device=device,
            )

            plot_loss(training_loss, validation_loss, f"{model_save_directory}", seed)

        print("Training complete.")
