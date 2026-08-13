from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

import reboot_py
from reboot.cryptocontext import CryptoContext
from reboot.layers.modules import Module
from reboot.optim.optimizers import Optimizer
from reboot.types import Array
from reboot.utils.enums import OptimizerName
from reboot.utils.nn import create_optimizer


class Linear(Module, ABC):
    """
    Base class for all linear layers, which apply a linear transformation to the input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        optimizer: OptimizerName,
        decay_rate: float = 0.0,
        momentum: float = 0.0,
        bias: bool = True,
        trainable: bool = True,
        propagate_backward: bool = True,
        debug: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(trainable=trainable, name=name, debug=debug)
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.decay_rate: float = decay_rate
        self.momentum: float = momentum
        self.propagate_backward: bool = propagate_backward

        # Initialize the optimizer using OptimizerName
        self.optimizer: Optimizer = create_optimizer(
            optimizer, self, weight_decay=decay_rate, momentum=momentum, debug=debug
        )

        # Weight initialization
        weight_shape = (in_features, out_features)
        bias_shape = (1, out_features)

        dtype = np.float32
        # bound = np.sqrt(1 / in_features)  # Kaiming uniform (PyTorch's default)
        bound = np.sqrt(1 / (in_features + out_features))  # Xavier uniform

        self.weights = np.random.uniform(-bound, bound, size=weight_shape)
        self.weights = self.weights.astype(dtype)
        self.bias = np.random.uniform(-bound, bound, size=bias_shape) if bias else None
        self.bias = self.bias.astype(dtype) if bias else None

    @abstractmethod
    def forward(self, x: Array) -> Array:
        pass

    @abstractmethod
    def backward(self, delta: Array, lr: float) -> Array:
        pass

    @abstractmethod
    def compute_gradients(self, delta: Array) -> tuple[Array, Array, Array]:
        """
        The gradient to be propagated to the previous layer is given by a matrix multiplication.

        Args:
            delta (Array): Gradient of the loss w.r.t. the output of the layer (the gradient coming from the next layer)

        Returns:
            gradients (tuple[Array, Array, Array]): A tuple of three elements containing the gradient of the loss w.r.t. the input of the layer
                (i.e., the gradient to be propagated to the previous layer), the gradient of the loss w.r.t. the weights (i.e., the weight update),
                and the gradient of the loss w.r.t. the bias (i.e., the bias update)
        """
        pass


class PlainLinear(Linear):
    """
    Regular linear layer. The weights and bias are not encrypted.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        optimizer (OptimizerName): The optimizer to use for updating the weights and bias.
        decay_rate (float, optional): The amount of weight decay to apply during parameter updates. Float in the range [0, 1].
        momentum (float, optional): The momentum factor to apply during parameter updates. Float in the range [0, 1].
        weights (np.array): Learnable weights of the layer of shape (out_features, in_features).
        bias (bool, optional): Whether to learn an additive bias term or not. If True, learnable bias of shape (1, out_features) will be instantiated.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        optimizer: OptimizerName,
        decay_rate: float = 0.0,
        momentum: float = 0.0,
        bias: bool = True,
        trainable: bool = True,
        propagate_backward: bool = True,
        debug: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            optimizer=optimizer,
            decay_rate=decay_rate,
            momentum=momentum,
            bias=bias,
            trainable=trainable,
            propagate_backward=propagate_backward,
            debug=debug,
            name=name,
        )

    def forward(self, x: Array) -> Array:
        if self.training:
            # Save the input for backprop
            self.last_input = x.copy()

        # Compute the linear transformation
        if self.bias is not None:
            out = x @ self.weights + self.bias
        else:
            out = x @ self.weights
        self.log_stats(out, "Forward")
        return out

    def backward(self, delta: Array, lr: float) -> Array:
        new_delta, weight_gradient, bias_gradient = self.compute_gradients(delta)
        if new_delta is not None:
            self.log_stats(new_delta, "Grad X")
        self.log_stats(weight_gradient, "Grad W")

        if self.trainable:
            weight_update, bias_update = self.optimizer.compute_updates(
                lr, weight_gradient, bias_gradient
            )
            self.log_stats(weight_update, "Update W")

            # Perform the update
            self.weights = self.weights - weight_update
            if self.bias is not None:
                self.bias = self.bias - bias_update
            self.log_stats(self.weights, "Weights")

        # Log the velocity if a momentum-based optimizer is used
        if (
            hasattr(self.optimizer, "velocity_w")
            and self.optimizer.velocity_w is not None
        ):
            self.log_stats(self.optimizer.velocity_w, "Velocity")

        # Back-propagate the error to the next layer
        return new_delta

    def compute_gradients(self, delta: Array) -> tuple[Array, Array, Array]:
        new_delta = None
        if self.propagate_backward:
            # Compute the new delta, i.e., the gradient towards the input
            new_delta = delta @ self.weights.T

        weight_gradient, bias_gradient = None, None
        if self.trainable:
            # Compute the updates to the trainable parameters
            weight_gradient = self.last_input.T @ delta
            if self.bias is not None:
                bias_gradient = np.sum(delta, axis=0, keepdims=True)

        return new_delta, weight_gradient, bias_gradient

    def log_stats(self, value: Array, value_name: str) -> None:
        if self.debug:
            print(
                f"| {self.name:16} | {value_name:10} | Min: {np.min(value):7.3f} | Max: {np.max(value):7.3f} | Mean: {np.mean(value):7.3f} | Std: {np.std(value):7.3f} |"
            )

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layer_repr = ""
        if logging_level >= 0:
            layer_repr += super().extra_repr()
        if logging_level >= 1:
            layer_repr += (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}"
            )
        if logging_level >= 2:
            layer_repr += f", name={self.name}" if self.name else ""
            layer_repr += f", decay_rate={self.decay_rate}, momentum={self.momentum}"
            layer_repr += f", trainable={self.trainable}, debug={self.debug}"
            layer_repr += f", optimizer={self.optimizer.__class__.__name__}"
        return layer_repr + ")"


class CKKSPackedLinear(Linear):
    """
    Encrypted linear layer compatible with CKKS. Weights are encrypted with full-matrix packing.
    There are two versions of this layer, given by the attribute `row_packing`.
    1) Row-encoded (flatten + encode):              2) Column-encoded (transpose + flatten + encode):
    - Forward pass:                                 - Forward pass:
        - Input expanded (aaa|bbb|ccc)                  - Input repeated (abc|abc|abc)
        - EvalSumRows(Mult)                             - EvalSumCols(Mult)
        - Output repeated (abc|abc|abc)                 - Output expanded (aaa|bbb|ccc)
    - Backward pass:                                - Backward pass:
        - Input repeated (abc|abc|abc)                  - Input expanded (aaa|bbb|ccc)
        - EvalSumCols(Mult)                             - EvalSumRows(Mult)
        - Output expanded (aaa|bbb|ccc)                 - Output repeated (abc|abc|abc)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        optimizer: OptimizerName,
        cc: CryptoContext,
        row_packing: bool,
        row_size: int,
        col_size: int,
        decay_rate: float = 0.0,
        momentum: float = 0.0,
        bias: bool = False,
        trainable: bool = True,
        propagate_backward: bool = True,
        debug: bool = False,
        name: str | None = None,
    ) -> None:
        if bias:
            raise NotImplementedError("Bias not yet supported for encrypted layers")
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            optimizer=optimizer,
            decay_rate=decay_rate,
            momentum=momentum,
            bias=bias,
            trainable=trainable,
            propagate_backward=propagate_backward,
            debug=debug,
            name=name,
        )
        self.cc = cc
        self.row_packing = row_packing
        self.row_size = row_size
        self.col_size = col_size

    def add_padding(self, array: np.ndarray, debug: bool = True) -> np.ndarray:
        if self.row_packing:
            bottom_padding = (0, max(0, self.row_size - self.in_features))
            right_padding = (0, max(0, self.col_size - self.out_features))
        else:
            bottom_padding = (0, max(0, self.col_size - self.in_features))
            right_padding = (0, max(0, self.row_size - self.out_features))
        array_padded = np.pad(array, (bottom_padding, right_padding))
        if debug:
            print(f"{self.name:>16} weights: {array.shape} -> {array_padded.shape}")
        return array_padded

    def encode_array(self, array: np.ndarray, debug: bool = True) -> np.ndarray:
        array_padded = self.add_padding(array, debug)
        if not self.row_packing:
            array_padded = array_padded.T
        return array_padded.flatten()

    def encrypt_layer(self, cc) -> None:
        w_encoded = self.encode_array(self.weights)
        w_encoded = self.cc.cc.MakeCKKSPackedPlaintext(w_encoded)
        self.weights = reboot_py.EncryptedValue(
            cc.cc.Encrypt(cc.kp.publicKey, w_encoded)
        )

    def forward(self, x: Array) -> Array:
        if self.training:
            # Save the input for backprop
            self.last_input = x.copy()
        if self.row_packing:
            out = reboot_py.row_packing_forward(x, self.weights, self.col_size)
        else:
            out = reboot_py.col_packing_forward(x, self.weights, self.col_size)
        self.log_all(out, "Forward")
        return out

    def compute_gradients(self, delta: Array) -> tuple[Array, Array, Array]:
        new_delta = None
        if self.propagate_backward:
            # Compute the new delta, i.e., the gradient towards the input
            if self.row_packing:
                new_delta = reboot_py.row_packing_backward(
                    delta, self.weights, self.col_size
                )
            else:
                new_delta = reboot_py.col_packing_backward(
                    delta, self.weights, self.col_size
                )

        weight_gradient, _ = None, None
        if self.trainable:
            weight_gradient = reboot_py.packing_weight_update(self.last_input, delta)
        return new_delta, weight_gradient, _

    def backward(self, delta: Array, lr: float) -> Array:
        new_delta, weight_gradient, bias_gradient = self.compute_gradients(delta)
        if new_delta is not None:
            self.log_all(new_delta, "Grad X")
        self.log_all(weight_gradient, "Grad W")

        if self.trainable:
            weight_update, _ = self.optimizer.compute_updates(
                lr, weight_gradient, bias_gradient
            )
            self.log_all(weight_update, "Update W")
            # Perform the update
            self.weights = self.weights - weight_update

        self.log_all(self.weights, "Weights")
        # Log the velocity if a momentum-based optimizer is used
        if (
            hasattr(self.optimizer, "velocity_w")
            and self.optimizer.velocity_w is not None
        ):
            self.log_all(self.optimizer.velocity_w, "Velocity")

        # Back-propagate the error to the next layer
        return new_delta

    def log_all(self, value: Array, value_name: str) -> None:
        if self.debug and self.encrypted:
            if not isinstance(value, np.ndarray):
                value = np.array([value])
            level = self.cc.get_level(value)
            precision = self.cc.get_precision(value)
            value_dec = self.cc.decrypt_array(value, num_slots=self.cc.valid_slots)
            print(
                f"| {self.name:16} | {value_name:10} | Min: {np.min(value_dec):7.3f} | Max: {np.max(value_dec):7.3f} | Mean: {np.mean(value_dec):7.3f} | Std: {np.std(value_dec):7.3f} | Level: {level:2} | Precision: {precision} |"
            )

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layer_repr = ""
        if logging_level >= 0:
            layer_repr += super().extra_repr()
        if logging_level >= 1:
            layer_repr += (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"row_packing={self.row_packing}"
            )
        if logging_level >= 2:
            layer_repr += f", name={self.name}" if self.name else ""
            layer_repr += f", decay_rate={self.decay_rate}, momentum={self.momentum}"
            layer_repr += f", trainable={self.trainable}, debug={self.debug}"
            layer_repr += f", optimizer={self.optimizer.__class__.__name__}"
            layer_repr += f", row_packing={self.row_packing}"
        return layer_repr + ")"
