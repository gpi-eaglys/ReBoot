from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

import reboot_py
from reboot.cryptocontext import CryptoContext
from reboot.layers.modules import Module
from reboot.types import Array


class Activation(Module, ABC):
    """
    Abstract class for activation functions.
    """

    def __init__(
        self,
        name: str | None = None,
        debug: bool = False,
        encrypted: bool = False,
        cc: CryptoContext | None = None,
    ) -> None:
        super().__init__(name=name, debug=debug)
        self.last_input: Array | None = None
        self.encrypted: bool = encrypted
        self.cc: CryptoContext | None = cc

    @abstractmethod
    def forward(self, x: Array) -> Array:
        if self.training:
            self.last_input = x.copy()

    @abstractmethod
    def backward(self, delta: Array, lr: float) -> Array:
        pass

    def log_stats(self, value: Array, value_name: str) -> None:
        if self.debug:
            if self.cc is not None:
                level = self.cc.get_level(value)
                precision = self.cc.get_precision(value)
                value_dec = self.cc.decrypt_array(value, num_slots=self.cc.valid_slots)
                print(
                    f"| {self.name:16} | {value_name:10} | Min: {np.min(value_dec):7.3f} | Max: {np.max(value_dec):7.3f} | Mean: {np.mean(value_dec):7.3f} | Std: {np.std(value_dec):7.3f} | Level: {level:2} | Precision: {precision} |"
                )
            else:
                print(
                    f"| {self.name:16} | {value_name:10} | Min: {np.min(value):7.3f} | Max: {np.max(value):7.3f} | Mean: {np.mean(value):7.3f} | Std: {np.std(value):7.3f} |"
                )

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        if 0 <= logging_level <= 1:
            return super().extra_repr(logging_level) + ")"
        else:
            return (
                super().extra_repr(logging_level)
                + f"name={self.name}, encrypted={self.encrypted})"
            )


class Square(Activation):
    """
    Element-wise square function.
    HE-compatible. Forward and backward passes have a depth of 1.
    """

    def forward(self, x: Array) -> Array:
        super().forward(x)
        if self.encrypted:
            out = reboot_py.square_forward(x)
        else:
            out = x * x
        self.log_stats(out, "Forward")
        return out

    def backward(self, delta: Array, lr: float) -> Array:
        if self.encrypted:
            out = reboot_py.square_backward(delta, self.last_input)
        else:
            out = delta * (2 * self.last_input)
        self.log_stats(out, "Backward")
        return out


class PolyReLU(Activation):
    """
    Polynomial approximation of the ReLU function from Ali et al. (2022).
    HE-compatible. Forward and backward passes have a depth of 1.
    """

    def forward(self, x: Array) -> Array:
        super().forward(x)
        if self.encrypted:
            out = reboot_py.poly_relu_forward(x)
        else:
            out = x * x + x
        self.log_stats(out, "Forward")
        return out

    def backward(self, delta: Array, lr: float) -> Array:
        if self.encrypted:
            out = reboot_py.poly_relu_backward(delta, self.last_input)
        else:
            out = delta * (2 * self.last_input + 1)
        self.log_stats(out, "Backward")
        return out
