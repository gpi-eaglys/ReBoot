from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import reboot_cpp
from lib.cryptocontext import CryptoContext
from lib.layers.modules import Module
from lib.types import Array
from lib.utils.enums import NonLinearity


class Model(Module, ABC):
    """
    Base class for all models.
    Forward and backward passes can be freely overridden.
    In debug mode, the activations and gradients of the forward and backward passes, respectively, will be saved.

    Attributes:
        layers (Sequential): List of references to the layers of the model.
    """

    def __init__(
        self, debug: bool = False, name: str | None = None, trainable: bool = True
    ) -> None:
        super().__init__(trainable=trainable, name=name, debug=debug)
        self.layers = None

    @abstractmethod
    def get_layers_with_parameters(self) -> list[Module]:
        """
        Returns a list of references to the layers in the module that have parameters (i.e., weights and biases).

        Returns:
            list[Module]: List of references to the layers with parameters
        """
        pass

    def get_trainable_layers(self) -> list[Module]:
        """
        Returns a list of references to the layers in the module that are trainable.

        Returns:
            list[Module]: List of references to the trainable layers
        """
        return [layer for layer in self.get_layers_with_parameters() if layer.trainable]

    def print_layers_parameters(self) -> None:
        """
        Prints the parameters of the layers of the model.

        Returns:
        -------
        None
        """
        param_layers = self.get_layers_with_parameters()
        for layer in param_layers:
            print(f"Layer: {layer.name}")
            if layer.weights is not None:
                print(f"- Weights:\n{layer.weights}")
            if layer.bias is not None:
                print(f"- Bias:\n{layer.bias}\n")

    def predict(
        self, x: Array, batch_size: int = 128, progress_bar: bool = True
    ) -> Array:
        """
        Generates output predictions for the input samples.
        Computation is done in batches, which are automatically generated.

        Parameters
        ----------
        x : Array
            Input data
        batch_size : int, default=128
            Batch size to be used for the prediction
        progress_bar : bool, default=True
            Whether to show a progress bar or not

        Returns
        -------
        Array
            Predictions of the model for the provided input data
        """
        predictions = []
        self.eval()
        for i in range(0, len(x), batch_size):
            x_batch = x[i : i + batch_size]
            predictions.append(self.forward(x_batch))
        return np.concatenate(predictions)

    def encrypt(self, cc: CryptoContext) -> None:
        """
        Returns a copy of the model with encrypted parameters.

        Args:
            cc (CryptoContext): The crypto context used to encrypt the model.

        Returns:
            Model: A copy of the model with encrypted parameters.
        """
        for layer in self.get_layers_with_parameters():
            layer.encrypted = True
            layer.encrypt_layer(cc)
        self.encrypted = True

    def bootstrap(self, cc: CryptoContext) -> None:
        """
        Apply CKKS bootstrapping to the the model's parameters.
        If a momentum-based optimizer is used, the velocities are also bootstrapped.
        Does not return anything as the operation happens in-place.
        The parameters num_iterations and precision are used in iterative bootstrapping only.
        OpenFHE currently supports only one or two iterations of bootstrapping.

        Args:
            cc (CryptoContext): The crypto context used to bootstrap the parameters of the model.
        Returns
            None
        """
        trainable_layers = self.get_trainable_layers()
        has_momentum = (
            hasattr(trainable_layers[0].optimizer, "velocity_w")
            and trainable_layers[0].optimizer.velocity_w is not None
        )

        if has_momentum:
            # Bootstrap weights and velocities
            weights = [layer.weights for layer in trainable_layers]
            velocities = [layer.optimizer.velocity_w for layer in trainable_layers]
            to_bootstrap = weights + velocities
            bootstrapped = reboot_cpp.bootstrap_array(
                to_bootstrap, num_iterations=cc.bs_iterations, precision=cc.bs_precision
            )
            # Assign the bootstrapped values to the layers
            for i, layer in enumerate(trainable_layers):
                layer.weights = bootstrapped[i]
                layer.optimizer.velocity_w = bootstrapped[i + len(weights)]

        else:
            # Bootstrap only the weights
            weights = [layer.weights for layer in trainable_layers]
            bootstrapped = reboot_cpp.bootstrap_array(
                weights, num_iterations=cc.bs_iterations, precision=cc.bs_precision
            )
            # Assign the bootstrapped values to the layers
            for i, layer in enumerate(trainable_layers):
                layer.weights = bootstrapped[i]


class LocalLossModel(Model, ABC):
    """
    Base class for models that can be trained by locally generated error signals.

    Attributes:
        blocks (list[LocalLossBlock]): List of the local loss blocks of the model, trained with local BP.
        layers (Sequential): Forward layers of the model, wrapped in a Sequential module.
        num_classes (int): Number of classes of the classification task.
        non_linearity (num): Non-linear activation function to be used in the model.
        subnet_trainable (bool): Whether the learning layers of the local loss blocks are trainable or not.
    """

    def __init__(
        self,
        num_classes: int,
        non_linearity: NonLinearity,
        subnet_trainable: bool = True,
        debug: bool = False,
        trainable: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(trainable=trainable, name=name, debug=debug)
        self.num_classes: int = num_classes
        self.non_linearity: NonLinearity = non_linearity
        self.subnet_trainable: bool = subnet_trainable

        self.blocks = None
        self.layers = None

    def forward(self, x: Array) -> Array:
        return self.layers(x)

    @abstractmethod
    def backward(self, delta: Array, lr: float, y_true: Array | None = None) -> Array:
        pass

    def train(self) -> None:
        self.training = True
        self.layers.train()

    def eval(self) -> None:
        self.training = False
        self.layers.eval()

    def freeze(self) -> None:
        self.trainable = False
        self.subnet_trainable = False
        self.layers.freeze()

    def unfreeze(self) -> None:
        self.trainable = True
        self.subnet_trainable = True
        self.layers.unfreeze()

    def get_layers_with_parameters(self) -> list:
        return self.layers.get_layers_with_parameters()

    def freeze_subnets(self) -> None:
        self.subnet_trainable = False
        for block in self.blocks:
            block.freeze_subnet()

    def unfreeze_subnets(self) -> None:
        self.subnet_trainable = True
        for block in self.blocks:
            block.unfreeze_subnet()
