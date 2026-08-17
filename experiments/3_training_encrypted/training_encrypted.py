import os
import sys
from typing import Any

import numpy as np
import wandb

import reboot_py
from reboot.cryptocontext import CryptoContext
from reboot.models.local_loss_models import LocalLossMLP
from reboot.optim.schedulers import CosineLR
from reboot.parser import get_parser_args
from reboot.utils.data import load_float_dataset, shuffle_dataset
from reboot.utils.enums import NonLinearity, OptimizerName
from reboot.utils.train import (
    batch_log,
    batch_wandb_log,
    compute_mean_weights,
    compute_precisions,
    epoch_log,
    epoch_wandb_log,
    partial_epoch_log,
    partial_epoch_wandb_log,
    save_precisions_csv,
    train_step_encrypted,
    wandb_init,
)

args = get_parser_args()


dataset_config = dict(
    dataset=args.dataset,
    subsample=args.subsample,
    resize=args.resize,
    normalize=False,
    pad_data=False,
    pad_labels=True,
)

network_config = dict(
    config_dir=args.config_dir,
    num_fc_layers=args.num_layers,
    num_fc_hidden=args.num_hidden,
    non_linearity=NonLinearity.POLY_RELU.name,
    dropout_rate=0.0,
    seed=args.seed,
    bias=False,
    debug=True,
    encrypted=True,
    instantiate_last_block=False,
    subnet_trainable=True,
    output_non_linearity=False,
)

training_config = dict(
    batch_size=args.batch_size,
    num_epochs=args.num_epochs,
    lr=args.lr,
    fwd_decay=args.weight_decay,
    lrn_decay=args.weight_decay,
    momentum=0.9,
    optimizer=OptimizerName.ENCRYPTED_NESTEROV_SGD.name,
    wandb=False,
    cpu_counts=args.cpu_counts
)

configs: dict[str, Any] = dataset_config | network_config | training_config


if __name__ == "__main__":
    print("Configuration:", configs)
    print("-"*79)
    for k, v in configs.items():
        print(f"-- {k}\t{v}")
    print("-"*79)

    # Setup reproducibility
    np.random.seed(configs["seed"])

    # W&B integration
    wandb_init(configs)
    # if cpu_counts > 0
    if configs["cpu_counts"] > 0:
        reboot_py.set_num_threads(configs["cpu_counts"])
    else:
        reboot_py.set_num_threads(os.cpu_count())
    print(f"\nDetected {reboot_py.get_num_threads()} threads")

    # Data
    np.random.seed(configs["seed"])
    X_train, _, X_test, y_train, _, y_test = load_float_dataset(
        configs, data_path=args.data_path, show_images=False
    )

    # Network
    np.random.seed(configs["seed"])
    plain_model = LocalLossMLP(
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
        instantiate_last_block=configs["instantiate_last_block"],
        subnet_trainable=configs["subnet_trainable"],
        output_non_linearity=configs["output_non_linearity"],
        encrypted=False,
    )

    # Encryption
    fname_yaml = f"fhe_config_mlp{args.num_layers - 1}.yaml"
    fpath_yaml = os.path.join(configs["config_dir"], fname_yaml)
    cc = CryptoContext(fpath_yaml, model=plain_model)

    np.random.seed(configs["seed"])
    encrypted_model = LocalLossMLP(
        cc=cc,
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
        instantiate_last_block=configs["instantiate_last_block"],
        subnet_trainable=configs["subnet_trainable"],
        row_size=cc.row_size,
        col_size=cc.col_size,
        output_non_linearity=configs["output_non_linearity"],
        encrypted=True,
    )
    print(encrypted_model.extra_repr(1))
    encrypted_model.encrypt(cc)

    # Training
    n_train_batches = len(X_train) // configs["batch_size"]
    loss_history_enc, acc_history_enc = [], []
    lr_scheduler = CosineLR(
        base_lr=configs["lr"], steps_before_restart=50, from_step=5 * n_train_batches
    )

    # Flatten the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    for epoch_id in range(configs["num_epochs"]):
        X_train, y_train = shuffle_dataset(X_train, y_train)
        loss_enc, acc_enc = 0, 0
        lr_scheduler.step()

        for batch_id in range(n_train_batches):
            # Generate the batch
            batch_start = batch_id * configs["batch_size"]
            batch_end = batch_start + configs["batch_size"]
            X = X_train[batch_start:batch_end]
            y = y_train[batch_start:batch_end]

            # Train encrypted model
            loss_enc_batch, acc_enc_batch, step_time, pred_dec = train_step_encrypted(
                lr=lr_scheduler.get_lr(),
                model=encrypted_model,
                X_train=X,
                y_train=y,
                cc=cc,
            )

            batch_log(
                epoch_id,
                configs["num_epochs"],
                batch_id,
                n_train_batches,
                loss_enc_batch,
                acc_enc_batch,
                step_time,
            )

            # Update epoch's stats
            loss_enc += loss_enc_batch
            acc_enc += acc_enc_batch

            # Log estimated epoch metrics
            partial_epoch_log(
                epoch_id,
                configs["num_epochs"],
                loss_enc,
                acc_enc,
                batch_id,
                n_train_batches,
            )
            partial_epoch_wandb_log(
                configs["wandb"], loss_enc, acc_enc, batch_id, n_train_batches, epoch_id
            )

            # Compute the mean absolute weights of the layers
            means_dict = compute_mean_weights(encrypted_model, cc)

            # Compute the precisions of the weights of the layers
            precisions_dict = compute_precisions(
                encrypted_model, cc, include_momentum=False
            )
            save_precisions_csv(epoch_id, batch_id, precisions_dict, configs["seed"])

            # Log statistics and differences
            batch_wandb_log(
                configs["wandb"],
                loss_enc_batch,
                acc_enc_batch,
                step_time,
                means_dict | precisions_dict,
            )

        # Update epoch's stats
        loss_enc /= n_train_batches
        acc_enc /= n_train_batches
        loss_history_enc.append(loss_enc)
        acc_history_enc.append(acc_enc)
        epoch_log(epoch_id, configs["num_epochs"], loss_enc, acc_enc)
        epoch_wandb_log(
            configs["wandb"], loss_enc, acc_enc, batch_id, n_train_batches, epoch_id
        )

    print(f"Final train loss: {loss_history_enc[-1]:.5f}")
    print(f"Final train accuracy: {acc_history_enc[-1]:.2%}")

    if configs["wandb"]:
        wandb.finish(quiet=True)
