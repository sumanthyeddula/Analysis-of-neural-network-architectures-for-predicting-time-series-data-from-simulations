import torch as pt
from FCNN import FCNNModel
from LSTM import LSTMModel

from autoregress_train import train, test
from utils import set_seed, split_dataset
from hyperparameter_tuning import hyperparameter_tuning

from plots import  plot_loss, plot_dataframe_columns_combined, compute_scaled_l2_loss_heatmap
import numpy as np
import optuna.visualization as vis

from hyperparameter_tuning import hyperparameter_tuning






# Example of training loop over multiple epochs and sequences
if __name__ == "__main__":
    
    test_mode = False
    hyperparameter_tune = True

    
    # Set model type
    model = FCNNModel


    # Set model parameters
    n_features = 15
    n_outputs = 14
    n_layers = 3
    n_neurons = 256
    sequence_length = 5
    n_steps = 1000 - sequence_length

    # Set training parameters
    n_epochs = 75
    learning_rate = 0.0001
    batch_Size = 4
    save_path = './fcnn_test'
    early_stopping_patience = 30
    test_size = 0.2

    set_seed(42)

    # Hyperparameter tuning parameters
    n_trails = 20


    # test model path
    model_path = './test/model_best.pth'
    
    if test_mode:
        # Load and prepare training data
        from Data_pipeline import process_all_simulations
        main_path = r'D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\exercises'
        all_dataframes = process_all_simulations(main_path, train=False, test_size=test_size)

        data = all_dataframes[6][0]

        if model is FCNNModel:
            model = FCNNModel(n_outputs=n_outputs, n_layers=n_layers, n_neurons=n_neurons, sequence_length=sequence_length, n_features=n_features)


        # Load the state dictionary
        model.load_state_dict(pt.load(model_path))


        criterion = pt.nn.MSELoss()

        avg_test_loss, all_predictions, all_actuals = test(model= model, n_steps= n_steps, n_features = n_features, test_Data=data , sequence_length = sequence_length, criterion = criterion, test_single=True)

        print(f"Average test loss: {avg_test_loss:.4f}")
        print(f"Predictions shape: {all_predictions}")    
        print(f"Actuals shape: {all_actuals}")

        all_predictions=np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        all_predictions = all_predictions.reshape(995, 14)
        all_actuals = all_actuals.reshape(995, 14)

        # Plot the predictions and actuals
        plot_dataframe_columns_combined(all_actuals, all_predictions, save_dir="test")
        # plot_dataframe_columns_heatmap(all_actuals, all_predictions, save_dir="test")
        compute_scaled_l2_loss_heatmap(all_actuals, all_predictions, save_dir="test")
        
        
        exit(0)

    
   

    else:
        # Load and prepare training data
        from Data_pipeline import process_all_simulations
        main_path = r'D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\exercises'
        all_dataframes = process_all_simulations(main_path, train=True, test_size=test_size)

    

        



        # Instantiate the model
        if model is FCNNModel:
            model = FCNNModel(n_outputs=n_outputs, n_layers=n_layers, n_neurons=n_neurons, sequence_length=sequence_length, n_features=n_features)
        elif model is LSTMModel:
            model = LSTMModel(n_features= n_features, hidden_size= n_neurons, num_layers= n_layers, n_outputs=n_outputs, sequence_length= sequence_length)

        optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = pt.nn.MSELoss()
        train_data , val_data = split_dataset(all_dataframes)

        if hyperparameter_tune:
           fcnn_study = hyperparameter_tuning('FCNN', train_data, val_data, n_outputs, sequence_length, n_features, n_steps, n_trails=n_trails)

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

           lstm_study = hyperparameter_tuning('LSTM', train_data, val_data, n_outputs, sequence_length, n_features, n_steps, n_trails=n_trails)

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
        



        train_losses, val_losses = train(model, n_epochs, n_steps, n_features, train_data, val_data, sequence_length, optimizer, criterion, batch_size= batch_Size, save_path=save_path, patience= early_stopping_patience, shuffle=True)

        # Plot the losses after training
        plot_loss(train_losses, val_losses, f"{save_path}_loss_plot.png")

        print("Training complete.")  