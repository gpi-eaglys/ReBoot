from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from lib.optim.optimizers import Optimizer
from lib.types import Array, Parameter


class Module(ABC):
    """
    Base class for all neural network modules.

    Args:
        weights (Parameter): The learnable weights of the module.
        bias (Parameter): The learnable bias of the module.
        optimizer(Optimizer): The optimizer used to update the parameters of the module.
        training (bool): Flag to indicate if the module is currently in training mode or not.
        trainable (bool, optional): Flag to indicate if the parameters of the module (if any) will be updated during training or not. Defaults to True.
        debug (bool, optional): Flag to indicate if the module is in debug mode or not. In debug mode, the module will save the activations and gradients of the forward and backward passes, respectively. Defaults to False.
        name (str, optional): Name of the module, used for debugging purposes. Defaults to None.
        encrypted (bool): Flag to indicate if the parameters of the module are encrypted or not. Defaults to False.
    """

    def __init__(
        self, trainable: bool = True, debug: bool = False, name: str | None = None
    ) -> None:
        self.weights: Parameter = None
        self.bias: Parameter = None
        self.optimizer: Optimizer | None = None
        self.last_input: Array | None = None

        self.debug: bool = debug
        self.training: bool = False
        self.name: str | None = name
        self.trainable: bool = trainable
        self.encrypted: bool = False

    @abstractmethod
    def forward(self, x: Array) -> Array:
        """
        Computes the forward pass of the layer.

        Args:
            x (Array): Input to the layer

        Returns:
            Array: Output of the layer
        """
        pass

    @abstractmethod
    def backward(self, delta: Array, lr: float) -> Array:
        """
        Computes the backward pass of the layer.

        Args:
            delta (Array): Gradient of the loss w.r.t. the output of the layer (the gradient coming from the next layer)
            lr (float): The learning rate used to compute the parameters' updates. If the layer works with integer arithmetics,

        Returns:
            Array: Gradient of the loss w.r.t. the input of the layer (the gradient to be propagated to the previous layer)
        """
        pass

    def train(self) -> None:
        """
        Sets the module in training mode.

        Returns:
            None
        """
        self.training = True

    def eval(self) -> None:
        """
        Sets the module in evaluation mode.

        Returns:
            None
        """
        self.training = False

    def freeze(self) -> None:
        """
        Freezes the module so that its parameters (if any) will no longer be updated.

        Returns:
            None
        """
        self.trainable = False

    def unfreeze(self) -> None:
        """
        Unfreezes the module so that its parameters (if any) will be updated.

        Returns:
            None
        """
        self.trainable = True

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        """
        Defines the extra representation of the module, used for debugging purposes.
        It is used to print the module in a human-readable format.

        Args:
            logging_level (int, optional): Level of logging to use. If 0, only the class name of the module is printed.
                If 1, the class name and the most relevant attributes are printed. If 2, all attributes are printed.

        Returns:
            str: A human-readable representation of the module
        """
        return self.__class__.__name__ + "("

    def __call__(self, *args, **kwargs) -> Array:
        """
        Shortcut to call the forward pass of the layer.

        Args:
            *args (list): List of positional arguments.
            **kwargs (dict): Dictionary of keyword arguments.

        Returns:
            Array: Output of the layer
        """
        return self.forward(*args, **kwargs)

    def __str__(self):
        return self.extra_repr(logging_level=1)

    def encrypt_layer(self) -> None:
        """
        Encrypts the parameters of the layer.

        Returns:
            None
        """
        pass
