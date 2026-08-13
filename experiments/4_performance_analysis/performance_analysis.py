import os
import sys
from time import perf_counter
from typing import Any

import numpy as np
import wandb

import reboot_py
from reboot.cryptocontext import CryptoContext
from reboot.models.local_loss_models import LocalLossMLP
from reboot.optim.schedulers import ConstantLR
from reboot.parser import get_parser_args
from reboot.utils.data import load_float_dataset, shuffle_dataset
from reboot.utils.enums import NonLinearity, OptimizerName
from reboot.utils.nn import l2_loss_grad_encrypted
from reboot.utils.train import batch_log, train_step_encrypted, wandb_init

args = get_parser_args()


# Experiment config
dataset_config = dict(
    dataset=args.dataset,
    subsample=args.subsample,
    resize=args.resize,
    input_dim=args.input_dim,
    output_dim=args.output_dim,
    normalize=False,
    pad_data=False,
    pad_labels=True,
)

network_config = dict(
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
)

configs: dict[str, Any] = dataset_config | network_config | training_config


def print_sep(text: str) -> None:
    print(f"{'-' * 139}\n{text}\n{'-' * 139}")


if __name__ == "__main__":
    # W&B integration
    wandb_init(configs)
    reboot_py.set_num_threads(os.cpu_count())
    print(f"\nDetected {reboot_py.get_num_threads()} threads")

    model_enc_times = []
    data_enc_times = []
    forward_times = []
    backward_times = []
    bootstrap_times = []

    # Data
    np.random.seed(args.seed)
    X_train, _, X_test, y_train, _, y_test = load_float_dataset(
        configs, show_images=False
    )

    # Network
    np.random.seed(args.seed)
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
    file_path = f"fhe_config_mlp{args.num_layers - 1}.yaml"
    cc = CryptoContext(f"../config/{file_path}", model=plain_model)

    ######################################################
    # Model encryption
    ######################################################
    for i in range(args.num_runs):
        np.random.seed(args.seed + i)
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
        model_enc_start = perf_counter()
        encrypted_model.encrypt(cc)
        model_enc_time = perf_counter() - model_enc_start
        print_sep(f"Model encryption time: {model_enc_time:.3f} s")
        model_enc_times.append(model_enc_time)

    ######################################################
    # Memory size
    ######################################################
    model_size = 0
    for layer in encrypted_model.get_trainable_layers():
        model_size += layer.weights.get_memory_size()
    print_sep(f"Model size: {model_size / (1024 ** 2):.3f} MB")

    ######################################################
    # Warmup
    ######################################################
    lr_scheduler = ConstantLR(configs["lr"])
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    n_train_batches = len(X_train) // configs["batch_size"]
    loss_history_enc, acc_history_enc = [], []
    X_train, y_train = shuffle_dataset(X_train, y_train)
    lr_scheduler.step()

    ######################################################
    # Warmup
    ######################################################
    print_sep("Running warmup...")
    warmup_batches = 5
    for batch_id in range(warmup_batches):
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
            0,
            configs["num_epochs"],
            batch_id,
            n_train_batches,
            loss_enc_batch,
            acc_enc_batch,
            step_time,
        )
    print_sep("End of warmup")
    
    print("Running performance analysis...")
    for i in range(args.num_runs):
        ######################################################
        # Data/labels encryption
        ######################################################
        X_train, y_train = shuffle_dataset(X_train, y_train, seed=args.seed + i)
        X = X_train[batch_start:batch_end]
        y = y_train[batch_start:batch_end]
        row_packing = encrypted_model.get_trainable_layers()[-1].row_packing

        data_enc_start = perf_counter()
        x_enc = cc.expand_and_encrypt(X)
        if row_packing:
            y_enc = cc.repeat_and_encrypt(y)
        else:
            y_enc = cc.expand_and_encrypt(y)
        data_enc_time = perf_counter() - data_enc_start
        data_enc_times.append(data_enc_time)
        print_sep(f"Data encryption time: {data_enc_time:.3f} s")

        ######################################################
        # Forward
        ######################################################
        forward_start = perf_counter()
        y_pred_enc = encrypted_model.forward(x_enc)
        forward_time = perf_counter() - forward_start
        forward_times.append(forward_time)
        print_sep(f"Forward pass time: {forward_time:.3f} s")

        ######################################################
        # Backward
        ######################################################
        backward_start = perf_counter()
        l2_grad_enc = l2_loss_grad_encrypted(y_enc, y_pred_enc)
        encrypted_model.backward(l2_grad_enc, y_true=y, lr=configs["lr"])
        backward_time = perf_counter() - backward_start
        backward_times.append(backward_time)
        print_sep(f"Backward pass time: {backward_time:.3f} s")

        ######################################################
        # Bootstrapping
        ######################################################
        bootstrap_start = perf_counter()
        encrypted_model.bootstrap(cc)
        bootstrap_time = perf_counter() - bootstrap_start
        bootstrap_times.append(bootstrap_time)
        print_sep(f"Bootstrapping time: {bootstrap_time:.3f} s")

    # Compute mean and standard deviation across runs
    model_enc_mean, model_std_mean = np.mean(model_enc_times), np.std(model_enc_times)
    data_enc_mean, data_std_mean = np.mean(data_enc_times), np.std(data_enc_times)
    forward_mean, forward_std = np.mean(forward_times), np.std(forward_times)
    backward_mean, backward_std = np.mean(backward_times), np.std(backward_times)
    bootstrap_mean, bootstrap_std = np.mean(bootstrap_times), np.std(bootstrap_times)
    print(f"{'-' * 139}")
    print(f"Mean model encryption time: {model_enc_mean:6.3f} ± {model_std_mean * 2:.3f} s")
    print(f" Mean data encryption time: {data_enc_mean:6.3f} ± {data_std_mean * 2:.3f} s")
    print(f"    Mean forward pass time: {forward_mean:6.3f} ± {forward_std * 2:.3f} s")
    print(f"   Mean backward pass time: {backward_mean:6.3f} ± {backward_std * 2:.3f} s")
    print(f"   Mean bootstrapping time: {bootstrap_mean:6.3f} ± {bootstrap_std * 2:.3f} s")
    print(f"{'-' * 139}")

    if configs["wandb"]:
        wandb.finish(quiet=True)
