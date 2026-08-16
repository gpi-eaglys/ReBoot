import random
import sys
from typing import Any


from reboot.parser import get_parser_args

args = get_parser_args()
print(args)

import numpy as np
import torch
import wandb
from reboot.models.backprop_models import BackpropMLP
from reboot.utils.data import load_float_dataset
from reboot.utils.nn import accuracy
from reboot.utils.train import epoch_log, wandb_init

from torch.utils.data import DataLoader, TensorDataset

# Experiment config
dataset_config = dict(
    dataset=args.dataset,
    subsample=args.subsample,
    resize=args.resize,
    normalize=False,
    pad_data=False,
    pad_labels=False,
)

network_config = dict(
    num_fc_layers=args.num_layers,
    num_fc_hidden=args.num_hidden,
    non_linearity="relu",
    seed=args.seed,
    bias=True,
    debug=False,
    encrypted=False,
)

training_config = dict(
    batch_size=args.batch_size,
    num_epochs=args.num_epochs,
    lr=args.lr,
    fwd_decay=args.weight_decay,
    lrn_decay=args.weight_decay,
    momentum=0.9,
    wandb=False,
    algorithm="backprop",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


configs: dict[str, Any] = dataset_config | network_config | training_config

if __name__ == "__main__":
    # Device-agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # W&B integration
    wandb_init(configs)
    train_accuracies, val_accuracies = [], []

    for i in range(args.num_runs):
        print(f"{'-' * 80}\nRunning experiment with seed {args.seed + i}\n{'-' * 80}")
        set_seed(configs["seed"] + i)
        # Data
        X_train, _, X_test, y_train, _, y_test = load_float_dataset(
            configs, data_path=args.data_path, show_images=False
        )

        # Model
        set_seed(configs["seed"] + i)
        model = BackpropMLP(
            num_layers=configs["num_fc_layers"],
            num_hidden_neurons=configs["num_fc_hidden"],
            input_dim=X_train.shape[1],
            num_classes=y_train.shape[1],
            non_linearity=configs["non_linearity"],
        ).to(device)
        print(model)

        # Optimizer
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=configs["lr"], weight_decay=configs["fwd_decay"], momentum=configs["momentum"], nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, eta_min=1e-7
        )
        
        # Instantiate dataloaders
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
        train_dataloader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=configs["batch_size"], shuffle=False, num_workers=0)    

        # Training
        loss_history, acc_history = [], []
        val_loss_history, val_acc_history = [], []
        n_train_batches = len(train_dataloader)
        n_val_batches = len(test_dataloader)

        # Training loop
        for epoch_id in range(configs["num_epochs"]):
            loss, acc = 0.0, 0.0
            val_loss, val_acc = 0.0, 0.0

            # Training step
            model.train()
            for batch in train_dataloader:
                lr_scheduler.step()
                # Generate the batch
                X, y = batch
                # Send data to device
                X, y = X.to(device), y.to(device)
                # 1. Optimizer zero grad
                optimizer.zero_grad()
                # 2. Forward pass
                y_pred = model(X)
                # 3. Calculate loss and accuracy (per batch)
                loss_batch = loss_fn(y_pred, y)
                acc_batch = accuracy(y_pred.detach(), y)
                # 4. Loss backward
                loss_batch.backward()
                # 5. Optimizer step
                optimizer.step()
                # Update epoch's stats
                loss += loss_batch.item()
                acc += acc_batch

            # Validation step
            model.eval()
            with torch.inference_mode():
                for batch in test_dataloader:
                    # Generate the batch
                    X, y = batch
                    # Send data to device
                    X, y = X.to(device), y.to(device)
                    # 1. Forward pass
                    test_pred = model(X)
                    # 2. Calculate loss (cumulatively)
                    val_loss += loss_fn(test_pred, y)
                    # 3. Calculate accuracy
                    val_acc += accuracy(test_pred, y)

            # Update epoch's stats
            loss = loss / n_train_batches
            acc = acc / n_train_batches
            val_loss = val_loss / n_val_batches
            val_acc = val_acc / n_val_batches
            loss_history.append(loss)
            acc_history.append(acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            epoch_log(epoch_id, configs["num_epochs"], loss, acc, val_acc)

        # Find the best validation accuracy
        best_val_acc_idx = np.argmax(val_acc_history)
        best_train_acc = acc_history[best_val_acc_idx]
        best_val_acc = val_acc_history[best_val_acc_idx]
        print(f"Best train accuracy: {best_train_acc:.2%}")
        print(f"Best validation accuracy: {best_val_acc:.2%}")
        train_accuracies.append(best_train_acc)
        val_accuracies.append(best_val_acc)

    # Compute mean and std of the best accuracies across runs
    train_mean = np.mean(train_accuracies)
    train_std = np.std(train_accuracies)
    val_mean = np.mean(val_accuracies)
    val_std = np.std(val_accuracies)
    print(f"{'-' * 80}")
    print(f"     Mean train accuracy: {train_mean:.2%} ± {train_std * 2:.2%}")
    print(f"Mean validation accuracy: {val_mean:.2%} ± {val_std * 2:.2%}")
    print(f"{'-' * 80}")

    if configs["wandb"]:
        wandb.run.summary["train/epoch_acc"] = train_mean
        wandb.run.summary["train/epoch_acc_std"] = train_std
        wandb.run.summary["val/epoch_acc"] = val_mean
        wandb.run.summary["val/epoch_acc_std"] = val_std
        wandb.finish(quiet=True)