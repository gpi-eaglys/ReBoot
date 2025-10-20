from abc import ABC
from time import time
from typing import Literal

import reboot_cpp
from lib.cryptocontext import CryptoContext
from lib.layers.modules import Module
from lib.models.models import Model
from lib.types import Array
from lib.utils.nn import l2_loss_grad, l2_loss_grad_encrypted


class LocalLossBlock(Model, ABC):
    """
    Base class for all blocks that use local loss.
    The block is trained by locally generated error signal based on MSE loss.
    In debug mode, the activations and gradients of the forward and backward passes, respectively, will be saved.

    Attributes:
        subnet_trainable (bool): If True, the parameters of the learning layers are updated during training
        layers (Sequential): The forward layers of the block
        last_activation (Array): The output of the block at the previous training step, used to compute the local loss
        pred_loss_net (Sequential): The learning layers used to compute the local loss
    """

    def __init__(
        self,
        subnet_trainable: bool = True,
        trainable: bool = True,
        debug: bool = False,
        name: str | None = None,
        cc: CryptoContext | None = None,
    ) -> None:
        super().__init__(debug=debug, trainable=trainable, name=name)
        self.subnet_trainable: bool = subnet_trainable

        self.layers = None
        self.last_activation: Array | None = None
        self.pred_loss_net = None
        self.cc = cc

    def compute_local_loss(self, h: Array, y_onehot: Array) -> Array:
        """
        Compute the local loss and the gradient of the loss with respect to the output of the block.
        It also updates the internal statistics of the block.

        Args:
            h (Array): The output of the block at the previous training step.
            y_onehot (Array): Ground truth labels of the input batch.

        Returns:
            loss_grad (Array): The gradient of the loss with respect to the output of the block.
        """
        # Forward pass of the learning layers
        y_hat_local = self.pred_loss_net(h)
        learning_layer = self.pred_loss_net[-1]
        if learning_layer.encrypted:
            # Encrypt y_onehot using repetition or encryption according to the learning layer's packing
            if learning_layer.row_packing:
                y_onehot_enc = self.pred_loss_net[0].cc.repeat_and_encrypt(y_onehot)
            else:
                y_onehot_enc = self.pred_loss_net[0].cc.expand_and_encrypt(y_onehot)
            grad_pred = l2_loss_grad_encrypted(y_onehot_enc, y_hat_local)
        else:
            grad_pred = l2_loss_grad(y_onehot, y_hat_local)
        return grad_pred

    def forward(self, x: Array) -> Array:
        x = self.layers(x)
        self.last_activation = x.copy()
        return x

    def backward(self, y_true: Array, lr: float) -> Array:
        """
        Performs the backward pass of the block, computing the local loss and back-propagating it through the block.

        Args:
            y_true (Array): Ground truth labels of the input batch.
            lr (float): The inverse of the learning rate used to compute the parameters' updates.

        Returns:
            local_loss (Array): The local loss of the block, i.e., the gradient of the loss w.r.t. the input of the block.
        """
        # Forward pass of the learning layers and compute the local loss
        delta = self.compute_local_loss(self.last_activation, y_true)

        # Back-propagate through the learning layers
        new_delta = self.pred_loss_net.backward(delta, lr)

        # Back-propagate through the forward layers
        new_delta = self.layers.backward(new_delta, lr)

        return new_delta

    def train(self) -> None:
        self.training = True
        self.layers.train()
        if self.pred_loss_net is not None:
            self.pred_loss_net.train()

    def eval(self) -> None:
        self.training = False
        self.layers.eval()
        if self.pred_loss_net is not None:
            self.pred_loss_net.eval()

    def freeze(self) -> None:
        self.trainable = False
        self.subnet_trainable = False
        self.layers.freeze()
        if self.pred_loss_net is not None:
            self.pred_loss_net.freeze()

    def unfreeze(self) -> None:
        self.trainable = True
        self.subnet_trainable = True
        self.layers.unfreeze()
        if self.pred_loss_net is not None:
            self.pred_loss_net.unfreeze()

    def get_layers_with_parameters(self) -> list[Module]:
        layers = self.layers.get_layers_with_parameters()
        if self.pred_loss_net is not None:
            layers = layers + self.pred_loss_net.get_layers_with_parameters()
        return layers

    def get_saved_activations(self) -> dict[str, Array]:
        return self.layers.get_saved_activations()

    def get_saved_gradients(self) -> dict[str, Array]:
        return self.layers.get_saved_gradients()

    def freeze_subnet(self) -> None:
        """
        Freeze the parameters of the learning layers, so that they are not updated during training.

        Returns:
            None
        """
        self.subnet_trainable = False
        if self.pred_loss_net is not None:
            self.pred_loss_net.freeze()

    def unfreeze_subnet(self) -> None:
        """
        Unfreeze the parameters of the learning layers, so that they are updated during training.

        Returns:
            None
        """
        self.subnet_trainable = True
        if self.pred_loss_net is not None:
            self.pred_loss_net.unfreeze()

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layers = ""
        for i, layer in enumerate(self.layers.layers):
            layers += f"\t\t\t({i}): {layer.extra_repr(logging_level)}\n"
        if self.pred_loss_net is not None:
            layers += f"\t\t\t(learning_layers): {self.pred_loss_net.extra_repr(logging_level)}\n"

        return f"{self.__class__.__name__}(\n{layers}\t)"
