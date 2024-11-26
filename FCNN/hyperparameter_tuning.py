




import torch
from torch import nn
from torch.optim import Adam
import optuna

from FCNN import FCNNModel
from LSTM import LSTMModel
from autoregress_train import train





def hyperparameter_tuning(best_model_type: str, train_Data, val_Data, n_outputs, sequence_length, n_features, n_steps, n_trails=50):
    
    if best_model_type == 'FCNN':
        def fcnn_objective(trial):
            n_layers = trial.suggest_int('n_layers', 1, 5)
            n_neurons = trial.suggest_int('n_neurons', 32, 256, log=True)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
            n_epochs = trial.suggest_int('n_epochs', 5, 50)
            patience = trial.suggest_int('patience', 3, 10)

            model = FCNNModel(
                n_outputs=n_outputs,
                n_layers=n_layers,
                n_neurons=n_neurons,
                sequence_length=sequence_length,
                n_features=n_features,
                activation=torch.nn.functional.leaky_relu
            )

            criterion = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr=learning_rate)

            try:
                _, val_losses = train(
                    model=model,
                    n_epochs=n_epochs,
                    n_steps=n_steps,
                    n_features=n_features,
                    train_Data=train_Data,
                    val_Data=val_Data,
                    sequence_length=sequence_length,
                    optimizer=optimizer,
                    criterion=criterion,
                    batch_size=batch_size,
                    save_path=f"fcnn_trial_{trial.number}",
                    patience=patience
                )
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')

            return min(val_losses)

        fcnn_study = optuna.create_study(direction='minimize')
        fcnn_study.optimize(fcnn_objective, n_trials=n_trails)

        print("Best FCNN hyperparameters:", fcnn_study.best_params)
        
        return fcnn_study




    if best_model_type == 'LSTM':
        def lstm_objective(trial):
            hidden_size = trial.suggest_int('hidden_size', 32, 256, log=True)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
            n_epochs = trial.suggest_int('n_epochs', 5, 50)
            patience = trial.suggest_int('patience', 3, 10)

            model = LSTMModel(
                n_features=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                n_outputs=n_outputs,
                sequence_length=sequence_length
            )

            criterion = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr=learning_rate)

            try:
                _, val_losses = train(
                    model=model,
                    n_epochs=n_epochs,
                    n_steps=n_steps,
                    n_features=n_features,
                    train_Data=train_Data,
                    val_Data=val_Data,
                    sequence_length=sequence_length,
                    optimizer=optimizer,
                    criterion=criterion,
                    batch_size=batch_size,
                    save_path=f"lstm_trial_{trial.number}",
                    patience=patience
                )
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')

            return min(val_losses)

        lstm_study = optuna.create_study(direction='minimize')
        lstm_study.optimize(lstm_objective, n_trials=n_trails)

        print("Best LSTM hyperparameters:", lstm_study.best_params)

        return lstm_study
