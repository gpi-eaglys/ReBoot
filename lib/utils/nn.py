import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.typing import NDArray

import reboot_cpp
from lib.layers.activations import PolyReLU, Square
from lib.layers.modules import Module
from lib.optim.optimizers import SGD, EncryptedNesterovSGD, Optimizer, PlainNesterovSGD
from lib.types import Array
from lib.utils.enums import NonLinearity, OptimizerName

sns.set_theme()


def l2_loss(
    y_true: NDArray[np.float32], y_pred: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Compute the RSS loss function"""
    return np.sum((y_true - y_pred) * (y_true - y_pred), axis=1)


def l2_loss_encrypted(y_true: Array, y_pred: Array, n_cols: int) -> Array:
    return reboot_cpp.enc_l2_loss(y_true, y_pred, n_cols)


def l2_loss_grad(
    y_true: NDArray[np.float32], y_pred: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Computes the gradient of the RSS loss function"""
    return y_pred - y_true


def l2_loss_grad_encrypted(y_true: Array, y_pred: Array) -> Array:
    return reboot_cpp.enc_l2_loss_grad(y_true, y_pred)


def accuracy(y_true: NDArray[np.float32], y_pred: NDArray[np.float32]) -> float:
    return (
        np.equal(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)).sum()
        / y_true.shape[0]
    )


def create_optimizer(
    optimizer: OptimizerName,
    layer: Module,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    debug: bool = False,
) -> Optimizer:
    match optimizer:
        case OptimizerName.SGD:
            return SGD(layer, weight_decay=weight_decay, debug=debug)
        case OptimizerName.PLAIN_NESTEROV_SGD:
            return PlainNesterovSGD(
                layer, weight_decay=weight_decay, momentum=momentum, debug=debug
            )
        case OptimizerName.ENCRYPTED_NESTEROV_SGD:
            return EncryptedNesterovSGD(
                layer, weight_decay=weight_decay, momentum=momentum, debug=debug
            )
        case _:
            raise ValueError(f"Invalid optimizer: {optimizer}")


def create_non_linearity(non_linearity: NonLinearity, **kwargs) -> Module:
    match non_linearity:
        case NonLinearity.SQUARE:
            return Square(**kwargs)
        case NonLinearity.POLY_RELU:
            return PolyReLU(**kwargs)
        case _:
            raise ValueError(f"Invalid non-linearity: {non_linearity}")


def plot_history(
    loss_history: list[float],
    val_loss_history: list[float],
    acc_history: list[float],
    val_acc_history: list[float],
    figsize: tuple[int, int] = (16, 5),
    log_scale: bool = False,
) -> None:
    _, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].set_title("Loss")
    if log_scale:
        axs[0].semilogy(loss_history, label="Train")
        axs[0].semilogy(val_loss_history, label="Validation")
    else:
        axs[0].plot(loss_history, label="Train")
        axs[0].plot(val_loss_history, label="Validation")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title("Accuracy")
    axs[1].plot(acc_history, label="Train")
    axs[1].plot(val_acc_history, label="Validation")
    axs[1].grid(True)
    axs[1].legend()
    plt.show()


def plot_plain_enc_history(
    loss_history_plain: list[float],
    loss_history_enc: list[float],
    acc_history_plain: list[float],
    acc_history_enc: list[float],
    figsize: tuple[int, int] = (16, 5),
    log_scale: bool = False,
) -> None:
    _, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].set_title("Loss")
    if log_scale:
        axs[0].semilogy(loss_history_plain, label="Plain")
        axs[0].semilogy(loss_history_enc, label="Encrypted")
    else:
        axs[0].plot(loss_history_plain, label="Plain")
        axs[0].plot(loss_history_enc, label="Encrypted")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title("Accuracy")
    axs[1].plot(acc_history_plain, label="Plain")
    axs[1].plot(acc_history_enc, label="Encrypted")
    axs[1].grid(True)
    axs[1].legend()
    plt.show()


class EarlyStopping:
    def __init__(
        self, min_delta: float = 0.01, patience: int = 10, from_epoch: int = 0
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.from_epoch = from_epoch
        self.max_val_acc = 0
        self.counter = 0
        self.epoch = 0

    def early_stop(self, val_acc: float) -> bool:
        self.epoch += 1
        if self.epoch < self.from_epoch:
            return False

        if val_acc < (self.max_val_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                # Early stopping condition met
                return True

        else:
            # Reset the counter and update the maximum validation accuracy
            self.max_val_acc = val_acc
            self.counter = 0
            return False

        return False


class ReduceLROnPlateau:
    def __init__(
        self,
        factor: int = 2,
        min_delta: float = 0.01,
        patience: int = 10,
        from_epoch: int = 0,
    ) -> None:
        self.factor = factor
        self.min_delta = min_delta
        self.patience = patience
        self.from_epoch = from_epoch
        self.max_val_acc = 0.0
        self.counter = 0
        self.epoch = 0

    def reduce_lr(self, val_acc: float) -> bool:
        self.epoch += 1
        if self.epoch < self.from_epoch:
            return False

        if val_acc < (self.max_val_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                # Plateau detected: reduce the learning rate
                self.counter = 0
                return True

        else:
            # Reset the counter and update the maximum validation accuracy
            self.max_val_acc = val_acc
            self.counter = 0
            return False

        return False
