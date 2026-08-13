import sys
from typing import Any

import numpy as np
import wandb

# sys.path.append("/workspaces/ReBoot/")

from reboot.models.local_loss_models import LocalLossMLP
from reboot.optim.schedulers import CosineLR
from reboot.parser import get_parser_args
from reboot.utils.data import load_float_dataset, shuffle_dataset
from reboot.utils.enums import NonLinearity, OptimizerName
from reboot.utils.train import (
    epoch_log,
    train_step_plain,
    validation_plain,
    wandb_init,
)

args = get_parser_args()

# Experiment configs
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
    non_linearity=NonLinearity.POLY_RELU.name,
    dropout_rate=0.0,
    seed=args.seed,
    bias=False,
    debug=False,
    encrypted=False,
    instantiate_last_block=False,
    subnet_trainable=True,
)

training_config = dict(
    batch_size=args.batch_size,
    num_epochs=args.num_epochs,
    lr=args.lr,
    fwd_decay=args.weight_decay,
    lrn_decay=args.weight_decay,
    momentum=0.9,
    optimizer=OptimizerName.PLAIN_NESTEROV_SGD.name,
    wandb=False,
    algorithm="reboot",
)

configs: dict[str, Any] = dataset_config | network_config | training_config

if __name__ == "__main__":
    # W&B integration
    wandb_init(configs)
    train_accuracies, val_accuracies = [], []

    for i in range(args.num_runs):
        # Setup reproducibility
        np.random.seed(args.seed + i)
        print(f"{'-' * 80}\nRunning experiment with seed {args.seed + i}\n{'-' * 80}")
        # Data
        X_train, _, X_test, y_train, _, y_test = load_float_dataset(
            configs, show_images=False
        )

        # Network
        np.random.seed(args.seed + i)
        model = LocalLossMLP(
            cc=None,
            num_fc_layers=configs["num_fc_layers"],
            num_fc_hidden=configs["num_fc_hidden"],
            num_classes=y_train.shape[1],
            input_dim=X_train.shape[1],
            dropout_rate=configs["dropout_rate"],
            non_linearity=NonLinearity[configs["non_linearity"]],
            fwd_decay_inv=configs["fwd_decay"],
            subnet_decay_inv=configs["lrn_decay"],
            bias=configs["bias"],
            debug=configs["debug"],
            optimizer=OptimizerName[configs["optimizer"]],
            momentum=configs["momentum"],
            encrypted=False,
            instantiate_last_block=configs["instantiate_last_block"],
            subnet_trainable=configs["subnet_trainable"],
        )
        print(model.extra_repr(1))

        # Training
        n_train_batches = len(X_train) // configs["batch_size"]
        loss_history, acc_history = [], []
        val_loss_history, val_acc_history = [], []
        lr_scheduler = CosineLR(
            base_lr=configs["lr"],
            steps_before_restart=50,
            from_step=n_train_batches * 5,
        )
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Training loop
        for epoch_id in range(configs["num_epochs"]):
            X_train, y_train = shuffle_dataset(X_train, y_train)
            loss, acc = 0, 0

            for batch_id in range(n_train_batches):
                lr_scheduler.step()
                # Generate the batch
                batch_start = batch_id * configs["batch_size"]
                batch_end = batch_start + configs["batch_size"]
                X = X_train[batch_start:batch_end]
                y = y_train[batch_start:batch_end]
                # Train plaintext model
                loss_batch, acc_batch, step_time, pred = train_step_plain(
                    lr=lr_scheduler.get_lr(), model=model, X_train=X, y_train=y
                )
                # Update epoch's stats
                loss += loss_batch
                acc += acc_batch

            # Validation step
            val_loss, val_acc = validation_plain(
                batch_size=configs["batch_size"],
                model=model,
                X_val=X_test,
                y_val=y_test,
            )

            # Update epoch's stats
            loss /= n_train_batches
            acc /= n_train_batches
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
