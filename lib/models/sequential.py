from typing import Literal

from lib.layers.modules import Module
from lib.models.models import Model
from lib.types import Array


class Sequential(Model):
    """
    A sequential container. Modules will be added to it in the order they are passed in the constructor.
    The forward pass of Sequential will call the forward pass of each module in the correct order.
    The same applies for the backward pass.

    Attributes:
        layers (list[Module]): List of arbitrary length of modules to be executed sequentially.
    """

    def __init__(
        self,
        layers: list[Module],
        trainable: bool = True,
        debug: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(trainable=trainable, debug=debug, name=name)
        self.layers: list[Module] = layers

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, delta: Array, lr: float) -> Array:
        for layer in reversed(list(self.layers)):
            delta = layer.backward(delta, lr)
        return delta

    def train(self) -> None:
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        self.training = False
        for layer in self.layers:
            layer.eval()

    def freeze(self) -> None:
        self.trainable = False
        for layer in self.layers:
            layer.freeze()

    def unfreeze(self) -> None:
        self.trainable = True
        for layer in self.layers:
            layer.unfreeze()

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layers = ""
        for i, layer in enumerate(self.layers):
            layers += f"\t\t\t({i}): {layer.extra_repr(logging_level)}\n"
        return f"{super().extra_repr(logging_level)}\n{layers}\t\t)"

    def get_layers_with_parameters(self) -> list[Module]:
        return [layer for layer in self.layers if layer.weights is not None]

    def __getitem__(self, item: int) -> Module:
        return self.layers[item]
