from typing import Literal

from lib.blocks.local_linear_block import LocalLinearBlock
from lib.cryptocontext import CryptoContext
from lib.layers.linear import CKKSPackedLinear, PlainLinear
from lib.models.models import LocalLossModel
from lib.models.sequential import Sequential
from lib.types import Array
from lib.utils.enums import NonLinearity, OptimizerName
from lib.utils.nn import create_non_linearity


class LocalLossMLP(LocalLossModel):
    """
    Multi-Layer Perceptron (MLP) that can be trained by locally generated error signals.
    It is composed of num_layers integer local-loss blocks and by a regular Integer Linear layer which is the output.

    Attributes:
        num_classes (int): Number of classes of the classification task
        non_linearity (Enum): Non-linear activation function to be used in the model
        num_layers (int): Number of layers of the model: (num_layers - 1) blocks + 1 regular output layer
        num_hidden_neurons (tuple[int]): A tuple of integers representing the number of neurons in each hidden layer
        cc (CryptoContext): The cryptographic context of the OpenFHE library
        fwd_decay_inv (int, default=0): The amount of weight decay to apply during parameter updates in the forward part of the blocks
            When equal to 0, no weight decay is applied.
            With larger numbers, the weight decay is smaller
        subnet_decay_inv (int, default=0): The amount of weight decay to apply during parameter updates in the learning layers of the blocks
            When equal to 0, no weight decay is applied.
            With larger numbers, the weight decay is smaller
        layers (Sequential): Forward layers of the model, wrapped in a Sequential module
        input (Flatten): Input layer of the model which flattens the input image into a vector
        blocks (List[LocalLossBlock]): List of the local loss blocks of the model, trained with local BP
        output_layers (Sequential): Output layers of the model wrapped in a Sequential module, trained with regular BP
        instantiate_last_block (bool): Whether to instantiate the last block of the model or replace it with two regular linear layers
        output_non_linearity (bool): Whether to apply a non-linearity after the output layer or not
    """

    def __init__(
        self,
        num_fc_layers: int,
        num_fc_hidden: tuple[int, ...],
        num_classes: int,
        input_dim: int,
        dropout_rate: float,
        non_linearity: NonLinearity,
        optimizer: OptimizerName,
        cc: CryptoContext | None,
        row_size: int = 0,
        col_size: int = 0,
        name_index: int = 0,
        debug: bool = False,
        bias: bool = True,
        fwd_decay_inv: int = 0,
        subnet_decay_inv: int = 0,
        encrypted: bool = True,
        name: str | None = None,
        momentum: float = 0.0,
        trainable: bool = True,
        subnet_trainable: bool = True,
        instantiate_last_block: bool = False,
        output_non_linearity: bool = False,
    ) -> None:
        if (num_fc_layers - 1) != len(num_fc_hidden):
            raise ValueError("The length of num_hidden_neurons must be num_layers - 1")
        if encrypted and any([row_size == 0, col_size == 0]):
            raise ValueError(
                "When encrypted is True, row_size and col_size must be provided"
            )
        super().__init__(
            num_classes,
            non_linearity,
            subnet_trainable,
            name=name,
            debug=debug,
            trainable=trainable,
        )
        self.num_layers = num_fc_layers
        self.num_hidden_neurons = num_fc_hidden
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.fwd_decay_inv = fwd_decay_inv
        self.subnet_decay_inv = subnet_decay_inv
        self.optimizer = optimizer
        self.encrypted = encrypted
        self.cc = cc

        # Instantiate layers
        self.hidden_layers = []
        self.blocks = []

        # If three or more layers are requested, instantiate local-loss blocks
        num_blocks = num_fc_layers - 1 if instantiate_last_block else num_fc_layers - 2
        layer_index = name_index
        row_packing = True
        for i in range(0, num_blocks):
            in_features = input_dim if i == 0 else num_fc_hidden[i - 1]
            self.hidden_layers.append(
                LocalLinearBlock(
                    in_features=in_features,
                    out_features=num_fc_hidden[i],
                    cc=cc,
                    num_classes=num_classes,
                    non_linearity=non_linearity,
                    optimizer=optimizer,
                    name=f"linear_block_{layer_index:02d}",
                    fwd_decay_inv=fwd_decay_inv,
                    subnet_decay_inv=subnet_decay_inv,
                    debug=debug,
                    bias=bias,
                    name_index=i + name_index,
                    trainable=trainable,
                    momentum=momentum,
                    subnet_trainable=subnet_trainable,
                    encrypted=encrypted,
                    row_packing=row_packing,
                    row_size=row_size,
                    col_size=col_size,
                )
            )
            row_packing = not row_packing
            layer_index += 1

            # Keep track of a reference to the block
            self.blocks.append(self.hidden_layers[-1])

        output_layers = []
        if not instantiate_last_block and num_fc_layers > 1:
            # Instantiate a regular hidden linear layer if needed
            in_features = input_dim if num_fc_layers == 2 else num_fc_hidden[-2]
            args = dict(
                in_features=in_features,
                out_features=num_fc_hidden[-1],
                bias=bias,
                decay_rate=fwd_decay_inv,
                name=f"linear_{layer_index:02d}",
                momentum=momentum,
                propagate_backward=False,
                debug=debug,
                optimizer=optimizer,
                trainable=trainable,
            )
            if self.encrypted:
                hidden_linear = CKKSPackedLinear(
                    cc=cc,
                    row_packing=row_packing,
                    row_size=row_size,
                    col_size=col_size,
                    **args,
                )
            else:
                hidden_linear = PlainLinear(**args)
            output_layers.append(hidden_linear)
            output_layers.append(
                create_non_linearity(
                    non_linearity,
                    name=f"non_linearity_{layer_index:02d}",
                    encrypted=encrypted,
                    debug=debug,
                    cc=cc if encrypted else None,
                )
            )
            layer_index += 1

        # The output is always given a by a regular linear layer
        in_features = input_dim if num_fc_layers == 1 else num_fc_hidden[-1]
        propagate_backward = True if num_fc_layers >= 2 else False
        row_packing = True if num_fc_layers == 1 else not row_packing

        args = dict(
            in_features=in_features,
            out_features=num_classes,
            bias=bias,
            decay_rate=subnet_decay_inv,
            name=f"linear_{layer_index:02d}",
            momentum=momentum,
            propagate_backward=propagate_backward,
            debug=debug,
            optimizer=optimizer,
            trainable=trainable,
        )
        if self.encrypted:
            output_linear = CKKSPackedLinear(
                cc=cc,
                row_packing=row_packing,
                row_size=row_size,
                col_size=col_size,
                **args,
            )
        else:
            output_linear = PlainLinear(**args)
        output_layers.append(output_linear)

        if output_non_linearity:
            output_layers.append(
                create_non_linearity(
                    non_linearity,
                    name=f"non_linearity_{layer_index:02d}",
                    encrypted=encrypted,
                    debug=debug,
                    cc=cc if encrypted else None,
                )
            )

        # Connect all the layers through a Sequential module for convenience
        self.output_layers = Sequential(output_layers, debug=debug, trainable=trainable)
        self.layers = Sequential(
            self.hidden_layers + output_layers, debug=debug, trainable=trainable
        )

    def backward(self, delta: Array, lr: float, y_true: Array | None = None) -> Array:
        # Backward pass of the output layers, with regular BP
        new_delta = self.output_layers.backward(delta, lr)

        # Backward pass of the local loss blocks, with local BP
        for block in reversed(self.blocks):
            new_delta = block.backward(y_true, lr)
        # Return the gradient of the input for debugging purposes
        return new_delta

    def get_layers_with_parameters(self) -> list:
        param_layers = []
        for block in self.blocks:
            param_layers = param_layers + block.get_layers_with_parameters()
        return param_layers + self.output_layers.get_layers_with_parameters()

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        return f"{self.__class__.__name__}(\n\t(0): {self.layers.extra_repr(logging_level)}\n\t)"
