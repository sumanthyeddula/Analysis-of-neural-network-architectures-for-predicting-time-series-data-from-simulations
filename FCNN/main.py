import torch as pt
from FCNN import FCNNModel
from LSTM import LSTMModel

from autoregress_train import train, test
from utils import set_seed, split_dataset

from plots import plot_dataframe_columns, plot_loss, plot_dataframe_columns_heatmap
import numpy as np




# Example of training loop over multiple epochs and sequences
if __name__ == "__main__":
    
    test_mode = False
    
    # Set model parameters
    model = FCNNModel
    n_features = 15
    n_outputs = 14
    n_layers = 4
    n_neurons = 128
    sequence_length = 5
    n_steps = 1000 - sequence_length
    n_epochs = 10
    learning_rate = 0.0001
    batch_Size = 4
    save_path = './test/model'
    early_stopping_patience = 5

    set_seed(42)
    
    model_path = './test/model_best.pth'
    
    if test_mode:
        # Load and prepare training data
        from Data_pipeline import process_all_simulations
        main_path = r'D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\exercises'
        all_dataframes = process_all_simulations(main_path)

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
        plot_dataframe_columns(all_actuals, all_predictions, save_dir="test")
        plot_dataframe_columns_heatmap(all_actuals, all_predictions, save_dir="test")
        
        exit(0)

    
   

    else:
        # Load and prepare training data
        from Data_pipeline import process_all_simulations
        main_path = r'D:\Research Project\Analysis-of-neural-network-architectures-for-predicting-time-series-data-from-simulations\exercises'
        all_dataframes = process_all_simulations(main_path)

    



        # Instantiate the model
        if model is FCNNModel:
            model = FCNNModel(n_outputs=n_outputs, n_layers=n_layers, n_neurons=n_neurons, sequence_length=sequence_length, n_features=n_features)
        elif model is LSTMModel:
            model = LSTMModel(n_outputs=n_outputs, n_layers=n_layers, hidden_size=n_neurons, sequence_length=sequence_length, n_features=n_features)

        optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = pt.nn.MSELoss()

        train_data , val_data = split_dataset(all_dataframes)

    

        train_losses, val_losses = train(model, n_epochs, n_steps, n_features, train_data, val_data, sequence_length, optimizer, criterion, batch_size= batch_Size, save_path=save_path, patience= early_stopping_patience)

        # Plot the losses after training
        plot_loss(train_losses, val_losses, f"{save_path}_loss_plot.png")

        print("Training complete.")  