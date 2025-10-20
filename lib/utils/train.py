import csv
import os
from time import time
from typing import Any

import numpy as np
import wandb

from lib.cryptocontext import CryptoContext
from lib.models.models import LocalLossModel, Model
from lib.types import Array
from lib.utils.nn import accuracy, l2_loss, l2_loss_grad, l2_loss_grad_encrypted


def train_step_plain(
    model: LocalLossModel,
    X_train: Array,
    y_train: Array,
    lr: float,
    cc: CryptoContext | None = None,
    **bw_args,
) -> tuple[float, float, float, Array]:
    return __train_step(model, X_train, y_train, lr, cc, encrypted=False, **bw_args)


def train_step_encrypted(
    model: LocalLossModel,
    X_train: Array,
    y_train: Array,
    lr: float,
    cc: CryptoContext,
    **bw_args,
) -> tuple[float, float, float, Array]:
    return __train_step(model, X_train, y_train, lr, cc, encrypted=True, **bw_args)


def __train_step(
    model: LocalLossModel,
    X_train: Array,
    y_train: Array,
    lr: float,
    cc: CryptoContext | None,
    encrypted: bool,
    **bw_args,
) -> tuple[float, float, float, Array]:
    model.train()
    loss, acc = 0, 0

    # Forward and backward pass
    step_start = time()
    if encrypted:
        row_packing = model.get_trainable_layers()[-1].row_packing
        assert cc is not None
        # Encrypted training
        x_enc = cc.expand_and_encrypt(X_train)
        if row_packing:
            y_enc = cc.repeat_and_encrypt(y_train)
        else:
            y_enc = cc.expand_and_encrypt(y_train)

        y_pred_enc = model.forward(x_enc)
        l2_grad_enc = l2_loss_grad_encrypted(y_enc, y_pred_enc)
        model.backward(
            l2_grad_enc,
            y_true=y_train,
            lr=lr,
        )
        y_pred_plain = cc.decrypt_array(y_pred_enc, num_slots=cc.valid_slots)

        if row_packing:
            y_pred_plain = y_pred_plain[:, : y_train.shape[1]]
        else:
            y_pred_plain = y_pred_plain[:, :: cc.col_size][:, : y_train.shape[1]]

        start = time()
        model.bootstrap(cc)
        print(
            f"{'-' * 139}\nBootstrapping the weights took {time() - start:.3f} s\n{'-' * 139}"
        )
    else:
        # Plaintext training
        y_pred_plain = model.forward(X_train)
        l2_grad_plain = l2_loss_grad(y_train, y_pred_plain)
        model.backward(l2_grad_plain, y_true=y_train, lr=lr, **bw_args)
    step_time = time() - step_start

    # Compute metrics
    batch_loss_enc = l2_loss(y_train, y_pred_plain).sum()
    batch_acc_enc = accuracy(y_train, y_pred_plain)
    loss += batch_loss_enc
    acc += batch_acc_enc

    return loss, acc, step_time, y_pred_plain


def validation_plain(
    model: Model,
    X_val: Array,
    y_val: Array,
    batch_size: int,
    cc: CryptoContext | None = None,
) -> tuple[float, float]:
    return __validation(model, X_val, y_val, batch_size, cc, encrypted=False)


def validation_encrypted(
    model: Model,
    X_val: Array,
    y_val: Array,
    batch_size: int,
    cc: CryptoContext,
) -> tuple[float, float]:
    return __validation(model, X_val, y_val, batch_size, cc, encrypted=True)


def __validation(
    model: Model,
    X_val: Array,
    y_val: Array,
    batch_size: int,
    cc: CryptoContext,
    encrypted: bool,
) -> tuple[float, float]:
    n_val_batches = int(np.ceil(len(X_val) / batch_size))
    n_val_batches = max(n_val_batches, 1)
    val_loss, val_acc = 0, 0
    model.eval()

    for batch_id in range(n_val_batches):
        # Generate the batch
        batch_start = batch_id * batch_size
        batch_end = batch_start + batch_size
        x = X_val[batch_start:batch_end]
        y = y_val[batch_start:batch_end]

        # Forward pass
        if encrypted:
            # Encrypted inference
            x_enc = cc.expand_and_encrypt(x)
            y_pred_enc = model.forward(x_enc)
            y_pred_plain = cc.decrypt_array(y_pred_enc, num_slots=cc.valid_slots)
            
            row_packing = model.get_trainable_layers()[-1].row_packing
            if row_packing:
                y_pred_plain = y_pred_plain[:, : y.shape[1]]
            else:
                y_pred_plain = y_pred_plain[:, :: cc.col_size][:, : y.shape[1]]

        else:
            # Plaintext inference
            y_pred_plain = model.forward(x)

        val_loss += l2_loss(y, y_pred_plain).sum()
        val_acc += accuracy(y, y_pred_plain)

    return float(val_loss / n_val_batches), float(val_acc / n_val_batches)


def wandb_init(config: dict[str, Any]) -> None:
    if config["wandb"]:
        try:
            with open("/workspaces/ReBoot/wandb_tags.txt", "r") as f:
                tags = f.read().splitlines()
        except FileNotFoundError:
            tags = []
        wandb.init(project="ReBoot", tags=tags)
        wandb.config.update(config)


def batch_log(
    epoch_id: int,
    n_epochs: int,
    batch_id: int,
    n_batches: int,
    batch_loss: float,
    batch_acc: float,
    step_time: float,
) -> None:
    """Log the stats of the current batch."""
    # Print step time in minutes and seconds
    minutes = int(step_time // 60)
    seconds = int(step_time % 60)
    stats = f"Epoch {epoch_id + 1:2d}/{n_epochs} - Step {batch_id + 1:2d}/{n_batches} - Batch loss: {batch_loss:.5f} - Batch accuracy: {batch_acc:2.2%} - Time: {minutes}m {seconds}s"
    print(f"{'-' * 139} \n{stats} \n{'-' * 139}")


def epoch_log(
    epoch_id: int,
    n_epochs: int,
    epoch_loss: float,
    epoch_acc: float,
    epoch_val_acc: float | None = None,
) -> None:
    """Log the stats of the current epoch."""
    if epoch_val_acc:
        print_string = f"Epoch {epoch_id + 1:3d}/{n_epochs} - Loss: {epoch_loss:.5f} - Accuracy: {epoch_acc:2.2%} - Val accuracy: {epoch_val_acc:2.2%}"
    else:
        print_string = f"Epoch {epoch_id + 1:3d}/{n_epochs} - Loss: {epoch_loss:.5f} - Accuracy: {epoch_acc:2.2%}"
    print(f"{'-' * 80} \n{print_string} \n{'-' * 80}")


def partial_epoch_log(
    epoch_id: int,
    n_epochs: int,
    epoch_loss: float,
    epoch_acc: float,
    batch_id: int,
    n_batches: int,
) -> None:
    """Log the estimated stats of the current epoch using the batches seen so far."""
    if batch_id != 0 and batch_id % (n_batches // 5) == 0:
        estimated_loss = epoch_loss / (batch_id + 1)
        estimated_acc = epoch_acc / (batch_id + 1)
        print_string = f"Epoch {epoch_id + 1:2d}/{n_epochs} - Estimated loss: {estimated_loss:.5f} - Estimated accuracy {estimated_acc:2.2%}"
        print(f"{'-' * 80} \n{print_string} \n{'-' * 80}")


def batch_wandb_log(
    log: bool,
    batch_loss: float,
    batch_acc: float,
    step_time: float | None = None,
    log_dict: dict[str, float] | None = None,
) -> None:
    """Log the stats to Weights & Biases."""
    if log:
        metrics = {
            "train/batch_loss": batch_loss,
            "train/batch_acc": batch_acc,
            "perf/step_time": step_time,
        }
        if log_dict is not None:
            wandb.log(metrics | log_dict)
        else:
            wandb.log(metrics)


def epoch_wandb_log(
    log: bool,
    epoch_loss: float,
    epoch_acc: float,
    batch_id: int,
    n_batches: int,
    epoch_id: int,
    epoch_val_loss: float | None = None,
    epoch_val_acc: float | None = None,
) -> None:
    """Log the stats to Weights & Biases."""
    if log:
        metrics = {
            "train/epoch_loss": epoch_loss,
            "train/epoch_acc": epoch_acc,
        }
        if epoch_val_loss is not None and epoch_val_acc is not None:
            metrics["val/epoch_loss"] = epoch_val_loss
            metrics["val/epoch_acc"] = epoch_val_acc
        step = epoch_id * n_batches + batch_id + 1
        wandb.log(metrics, step=step)


def partial_epoch_wandb_log(
    log: bool,
    epoch_loss: float,
    epoch_acc: float,
    batch_id: int,
    n_batches: int,
    epoch_id: int,
) -> None:
    """Log the estimated stats of the current epoch using the batches seen so far."""
    if log and batch_id != 0 and batch_id % (n_batches // 5) == 0:
        estimated_loss = epoch_loss / (batch_id + 1)
        estimated_acc = epoch_acc / (batch_id + 1)
        metrics = {
            "train/epoch_loss": estimated_loss,
            "train/epoch_acc": estimated_acc,
        }
        step = epoch_id * n_batches + batch_id + 1
        wandb.log(metrics, step=step)


def compute_zero_weights(model: Model, cc: CryptoContext) -> dict[str, float]:
    new_dict = {}
    layers = model.get_trainable_layers()
    for layer in layers:
        if layer.encrypted:
            weight_size = layer.in_features * layer.out_features
            weights = cc.decrypt_array(layer.weights, num_slots=cc.num_slots)
            # All weights after weight_size should be zeros
            zero_weights = weights[weight_size:]
            # Check their norm
            zero_norm = np.linalg.norm(zero_weights)
            new_dict[f"padding_norm/{layer.name}"] = zero_norm
    return new_dict


def check_repeated_values(model: Model, cc: CryptoContext) -> dict[str, float]:
    new_dict = {}
    layers = model.get_trainable_layers()
    for layer in layers:
        if layer.encrypted:
            last_input = cc.decrypt_array(layer.last_input[0], num_slots=cc.num_slots)
            if layer.row_packing:
                # The input is expanded. Compare the first value of each row with the others
                last_input = last_input.reshape(-1, cc.col_size)
                row_norms = []
                for i in range(last_input.shape[0]):
                    expected = np.full_like(last_input[i], last_input[i][0])
                    row_norms.append(np.linalg.norm(last_input[i] - expected))
                new_dict[f"repeated_values/{layer.name}"] = np.mean(row_norms)
            else:
                # The input is repeated. Compare the first row with the others
                last_input = last_input.reshape(-1, cc.col_size)
                row_norms = []
                expected = last_input[0]
                for i in range(last_input.shape[0]):
                    row_norms.append(np.linalg.norm(last_input[i] - expected))
                new_dict[f"repeated_values/{layer.name}"] = np.mean(row_norms)
    return new_dict


def compute_differences(
    encrypted_model: Model,
    plain_model: Model,
    y_pred_plain: Array,
    y_pred_dec: Array,
    cc: CryptoContext,
) -> dict[str, float]:
    """Compute the differences between the weights of the plaintext and the encrypted model."""
    differences_dict = {}
    plain_layers = plain_model.get_trainable_layers()
    enc_layers = encrypted_model.get_trainable_layers()

    for plain_layer, enc_layer in zip(plain_layers, enc_layers):
        dec_weights = cc.decrypt_array(enc_layer.weights, num_slots=cc.valid_slots)
        encoded_weights = enc_layer.encode_array(plain_layer.weights, debug=False)

        key_name = f"diff/{plain_layer.name}"
        difference_norm = np.linalg.norm(encoded_weights - dec_weights)
        differences_dict[key_name] = difference_norm / encoded_weights.size
        print(
            f"| {enc_layer.name:16} | Mean difference: {differences_dict[key_name]:.5e} |"
        )

    key_name = "diff/preds"
    differences_dict[key_name] = (
        np.linalg.norm(y_pred_plain - y_pred_dec) / y_pred_plain.size
    )
    print(f"| {'Preds':16} | Mean difference: {differences_dict[key_name]:.5e} |")
    print(f"{'-' * 139}")
    return differences_dict


def compute_precisions(
    model: Model, cc: CryptoContext, include_momentum: bool = True
) -> dict[str, float]:
    """Compute the precision of the weights of all layers of the model."""
    precisions_dict = {}
    layers = model.get_layers_with_parameters()
    get_mean = lambda x: float(x[1 : x.find("|")])

    for layer in layers:
        # Weights
        key_name = f"precision/{layer.name}"
        precisions = cc.get_precision(np.array([layer.weights]))
        mean_precision = get_mean(precisions)
        precisions_dict[key_name] = mean_precision

        # Momentums
        if include_momentum and hasattr(layer.optimizer, "velocity_w"):
            key_name = f"precision/{layer.name}_momentum"
            precisions = cc.get_precision(np.array([layer.optimizer.velocity_w]))
            mean_precision = get_mean(precisions)
            precisions_dict[key_name] = mean_precision

    return precisions_dict


def compute_mean_weights(
    model: Model, cc: CryptoContext | None = None
) -> dict[str, float]:
    """Compute the mean of weights (in absolute value) of all layers of the model."""
    stats_dict = {}
    layers = model.get_trainable_layers()

    for layer in layers:
        # Weights
        if layer.encrypted:
            weights = cc.decrypt_array(layer.weights, num_slots=cc.valid_slots)
        else:
            weights = layer.weights
        mean = np.mean(weights)
        std = np.std(weights)
        stats_dict[f"mean/{layer.name}"] = mean
        stats_dict[f"std/{layer.name}"] = std

        # Momentums
        if (
            hasattr(layer.optimizer, "velocity_w")
            and layer.optimizer.velocity_w is not None
        ):
            key_name = f"mean/{layer.name}_momentum"
            if layer.encrypted:
                velocity = cc.decrypt_array(
                    layer.optimizer.velocity_w, num_slots=cc.num_slots
                )
            else:
                velocity = layer.optimizer.velocity_w
            mean = np.mean(velocity)
            std = np.std(velocity)
            stats_dict[f"mean/{layer.name}_momentum"] = mean
            stats_dict[f"std/{layer.name}_momentum"] = std

    return stats_dict


def save_differences_csv(
    epoch_id: int, batch_id: int, differences: dict[str, float], step_time: float
):
    """Save the differences between the weights of the plaintext and the encrypted model to a CSV file."""
    os.makedirs("res", exist_ok=True)
    with open("res/precision.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch_id, batch_id] + list(differences.values()) + [step_time])


def save_precisions_csv(
    epoch_id: int, batch_id: int, precisions: dict[str, float], seed: int
):
    """Save the precisions of the layers' weights to a CSV file."""
    os.makedirs("res", exist_ok=True)
    with open(f"res/precisions_{seed}.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch_id, batch_id] + list(precisions.values()))


def save_accuracies_csv(
    epoch_id: int, batch_id: int, acc_plain: float, acc_enc: float, seed: int
):
    """Save the accuracies of the plaintext and encrypted models to a CSV file."""
    os.makedirs("res", exist_ok=True)
    with open(f"res/accuracies_{seed}.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch_id, batch_id, acc_plain, acc_enc])


def clear_csv():
    """Delete the precision.csv and accuracies.csv files if they exist."""
    if os.path.exists("res/precisions.csv"):
        os.remove("res/precisions.csv")
    if os.path.exists("res/accuracies.csv"):
        os.remove("res/accuracies.csv")
