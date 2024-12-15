import torch
from torch import nn
from torch.optim import Adam


import optuna

from FCNN import FCNNModel
from LSTM import LSTMModel
from autoregress_train import train
from utils import set_seed


def hyperparameter_tuning(
    best_model_type: str,
    train_data,
    val_data,
    n_outputs,
    sequence_length,
    n_features,
    n_steps,
    n_trails=50,
    device="cpu",
    save_path=".",
):

    if best_model_type == "FCNN":

        def fcnn_objective(trial):
            seed = trial.suggest_int("seed", 0, 100)
            n_layers = trial.suggest_int("n_layers", 1, 7)
            n_neurons = trial.suggest_int("n_neurons", 32, 256, log=True)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_int("batch_size", 16, 128, log=True)
            n_epochs = trial.suggest_int("n_epochs", 5, 15)
            patience = trial.suggest_int("patience", 3, 10)
            # start_sampling_prob = trial.suggest_float("start_sampling_prob", 0.1, 1.0)

            set_seed(seed)

            model = FCNNModel(
                n_outputs=n_outputs,
                n_layers=n_layers,
                n_neurons=n_neurons,
                sequence_length=sequence_length,
                n_features=n_features,
                activation=torch.nn.functional.leaky_relu,
            )

            criterion = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr=learning_rate)

            try:
                trainL2, valL2, _, val_losses = train(
                    model=model,
                    n_epochs=n_epochs,
                    n_steps=n_steps,
                    n_features=n_features,
                    train_Data_preprocessed=train_data,
                    val_Data_preprocessed=val_data,
                    sequence_length=sequence_length,
                    optimizer=optimizer,
                    criterion=criterion,
                    batch_size=batch_size,
                    save_path=f"{save_path}",
                    patience=patience,
                    start_sampling_prob=0.0,
                    sampling_schedule_type="constant",
                    device=device,
                )
                for epoch, val_loss in enumerate(val_losses):
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            except Exception as e:
                print(f"Trial failed: {e}")
                return float("inf")

            trial.set_user_attr(
                "best_model_path", f"fcnn_best_trial_{trial.number}.pth"
            )
            torch.save(model.state_dict(), f"fcnn_best_trial_{trial.number}.pth")

            return min(val_losses)

        fcnn_study = optuna.create_study(
            direction="minimize", pruner=optuna.pruners.MedianPruner()
        )
        fcnn_study.optimize(fcnn_objective, n_trials=n_trails)

        print("Best FCNN hyperparameters:", fcnn_study.best_params)

        return fcnn_study

    if best_model_type == "LSTM":

        def lstm_objective(trial):
            seed = trial.suggest_int("seed", 0, 1000)
            hidden_size = trial.suggest_int("hidden_size", 32, 256, log=True)
            num_layers = trial.suggest_int("num_layers", 1, 3)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_int("batch_size", 16, 128, log=True)
            n_epochs = trial.suggest_int("n_epochs", 5, 50)
            patience = trial.suggest_int("patience", 3, 10)

            set_seed(seed)

            model = LSTMModel(
                n_features=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                n_outputs=n_outputs,
                sequence_length=sequence_length,
            )

            criterion = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr=learning_rate)

            try:
                trainL2, valL2, _, val_losses = train(
                    model=model,
                    n_epochs=n_epochs,
                    n_steps=n_steps,
                    n_features=n_features,
                    train_Data_preprocessed=train_data,
                    val_Data_preprocessed=val_data,
                    sequence_length=sequence_length,
                    optimizer=optimizer,
                    criterion=criterion,
                    batch_size=batch_size,
                    save_path=f"lstm_trial_{trial.number}",
                    patience=patience,
                    start_sampling_prob=0.0,
                    sampling_schedule_type="constant",
                    device=device,
                )
                for epoch, val_loss in enumerate(val_losses):
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            except Exception as e:
                print(f"Trial failed: {e}")
                return float("inf")

            trial.set_user_attr(
                "best_model_path", f"lstm_best_trial_{trial.number}.pth"
            )
            torch.save(model.state_dict(), f"lstm_best_trial_{trial.number}.pth")

            return min(val_losses)

        lstm_study = optuna.create_study(
            direction="minimize", pruner=optuna.pruners.MedianPruner()
        )
        lstm_study.optimize(lstm_objective, n_trials=n_trails)

        print("Best LSTM hyperparameters:", lstm_study.best_params)

        return lstm_study
