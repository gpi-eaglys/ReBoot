from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

import reboot_cpp
from lib.types import Array


class Optimizer(ABC):
    def __init__(
        self, layer: "Module", weight_decay: float = 0.0, debug: bool = False
    ) -> None:
        self.layer: "Module" = layer
        self.weight_decay: float = weight_decay
        self.debug: bool = debug

    @abstractmethod
    def compute_updates(
        self, lr: float, weight_gradient: Array, bias_gradient: Array
    ) -> tuple[Array, Array]:
        pass

    def _compute_weight_decay(
        self, weight_gradient: Array, bias_gradient
    ) -> tuple[Array, Array]:
        weight_gradient = weight_gradient + self.weight_decay * self.layer.weights
        if self.layer.bias is not None:
            bias_gradient = bias_gradient + self.weight_decay * self.layer.bias
        return weight_gradient, bias_gradient

    def _compute_updates(
        self, lr: float, weight_gradient: Array, bias_gradient: Array
    ) -> tuple[Array, Array]:
        weight_gradient = weight_gradient * lr
        if self.layer.bias is not None:
            bias_gradient = bias_gradient * lr
        return weight_gradient, bias_gradient


class SGD(Optimizer):
    def compute_updates(
        self, lr: float, weight_gradient: Array, bias_gradient: Array
    ) -> tuple[Array, Array]:
        if self.weight_decay > 0.0:
            # Add weight decay term if needed
            weight_gradient, bias_gradient = self._compute_weight_decay(
                weight_gradient, bias_gradient
            )

        # Compute the updates
        weight_update, bias_update = self._compute_updates(
            lr, weight_gradient, bias_gradient
        )

        # Return the updates
        return weight_update, bias_update


class PlainNesterovSGD(Optimizer):
    def __init__(
        self,
        layer: "Module",
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        debug: bool = False,
    ) -> None:
        super().__init__(layer, weight_decay, debug)
        self.momentum: float = momentum
        self.velocity_w: NDArray[np.float32] | None = None
        if layer.bias is not None:
            self.velocity_b: NDArray[np.float32] | None = None

    def compute_updates(
        self,
        lr: float,
        weight_gradient: NDArray[np.float32],
        bias_gradient: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if self.weight_decay > 0.0:
            # Add weight decay term if needed
            weight_gradient, bias_gradient = self._compute_weight_decay(
                weight_gradient, bias_gradient
            )

        if self.velocity_w is None:
            # Initialize the velocity term
            self.velocity_w = weight_gradient
            # Compute the gradient with momentum
            weight_gradient = weight_gradient + self.momentum * weight_gradient

            if self.layer.bias is not None:
                self.velocity_b = bias_gradient
                bias_gradient = bias_gradient + self.momentum * bias_gradient
        else:
            original_weight_gradient = weight_gradient
            # Compute the gradient with momentum
            weight_gradient = (
                original_weight_gradient
                + self.momentum * original_weight_gradient
                + (self.momentum**2) * self.velocity_w
            )
            # Update the velocity term
            self.velocity_w = self.momentum * self.velocity_w + original_weight_gradient

            if self.layer.bias is not None:
                original_bias_gradient = bias_gradient
                bias_gradient = (
                    original_bias_gradient
                    + self.momentum * original_bias_gradient
                    + (self.momentum**2) * self.velocity_b
                )
                self.velocity_b = (
                    self.momentum * self.velocity_b + original_bias_gradient
                )

        # Compute the updates
        weight_update, bias_update = self._compute_updates(
            lr, weight_gradient, bias_gradient
        )

        # Return the updates
        return weight_update, bias_update


class EncryptedNesterovSGD(Optimizer):
    def __init__(
        self,
        layer: "Module",
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        debug: bool = False,
    ) -> None:
        super().__init__(layer, weight_decay, debug)
        self.momentum: float = momentum
        self.velocity_w: Array | None = None
        if layer.bias is not None:
            self.velocity_b: Array | None = None

    def compute_updates(
        self, lr: float, weight_gradient: Array, bias_gradient: Array
    ) -> tuple[Array, Array]:
        if self.velocity_w is None:
            # Initialize the velocity term
            weight_update, self.velocity_w = reboot_cpp.init_nesterov(
                weight_gradient=weight_gradient,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                layer_weights=self.layer.weights,
                lr=lr,
            )
        else:
            # Compute the gradient with momentum
            weight_update, self.velocity_w = reboot_cpp.compute_nesterov(
                weight_gradient=weight_gradient,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                layer_weights=self.layer.weights,
                lr=lr,
                velocity=self.velocity_w,
            )

        # Return the updates
        return weight_update, None
