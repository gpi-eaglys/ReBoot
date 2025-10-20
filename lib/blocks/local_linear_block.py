from lib.blocks.local_loss_block import LocalLossBlock
from lib.cryptocontext import CryptoContext
from lib.layers.linear import CKKSPackedLinear, PlainLinear
from lib.models.sequential import Sequential
from lib.utils.enums import NonLinearity, OptimizerName
from lib.utils.nn import create_non_linearity


class LocalLinearBlock(LocalLossBlock):
    """
    A local-loss block containing the following layers:
    - Integer Linear layer
    - Scaling layer
    - Non-linearity
    - Dropout layer (optional)

    Attributes:
        in_features (int): Number of input features of the block
        out_features (int): Number of output features of the block
        cc (CryptoContext): The cryptographic context of the OpenFHE library
        num_classes (int): Number of classes of the classification task, used in the local prediction loss
        non_linearity (Enum): Non-linear activation function to be used in the block
        fwd_decay_inv (int, default=0): Inverse of the decay rate to be used in the Integer Linear forward layer
        subnet_decay_inv (int, default=0): Inverse of the decay rate to be used in the Integer Linear learning layer
        bias (bool): Whether to use a bias in the Integer layers
        layers (Sequential): The forward layers of the block
        pred_loss_net (Sequential): The learning layers used to compute the local loss
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cc: CryptoContext,
        num_classes: int,
        non_linearity: NonLinearity,
        optimizer: OptimizerName,
        row_packing: bool = True,
        row_size: int = 0,
        col_size: int = 0,
        name: str | None = None,
        fwd_decay_inv: int = 0,
        momentum: float = 0.0,
        subnet_decay_inv: int = 0,
        debug: bool = False,
        bias: bool = True,
        name_index: int = 0,
        trainable: bool = True,
        encrypted: bool = True,
        subnet_trainable: bool = True,
    ) -> None:
        super().__init__(
            subnet_trainable, debug=debug, trainable=trainable, name=name, cc=cc
        )
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.num_classes: int = num_classes
        self.non_linearity: NonLinearity = non_linearity
        self.fwd_decay_inv: float = fwd_decay_inv
        self.subnet_decay_inv: float = subnet_decay_inv
        self.encrypted: bool = encrypted
        self.optimizer = optimizer

        # Instantiate the forward layers
        args = dict(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            decay_rate=fwd_decay_inv,
            name=f"linear_fwd_{name_index:02d}",
            propagate_backward=False,
            debug=debug,
            optimizer=optimizer,
            trainable=trainable,
            momentum=momentum,
        )
        if encrypted:
            fwd_linear = CKKSPackedLinear(
                cc=cc,
                row_packing=row_packing,
                row_size=row_size,
                col_size=col_size,
                **args,
            )
        else:
            fwd_linear = PlainLinear(**args)

        layers = [
            fwd_linear,
            create_non_linearity(
                non_linearity,
                name=f"non_linearity_{name_index:02d}",
                encrypted=encrypted,
                debug=debug,
                cc=cc if encrypted else None,
            ),
        ]

        self.layers = Sequential(layers, debug=debug, trainable=trainable)

        # Build the learning layers
        args = dict(
            in_features=out_features,
            out_features=num_classes,
            bias=bias,
            decay_rate=subnet_decay_inv,
            name=f"linear_lrn_{name_index:02d}",
            debug=debug,
            optimizer=optimizer,
            trainable=subnet_trainable,
            momentum=momentum,
        )
        if self.encrypted:
            pred_linear = CKKSPackedLinear(
                cc=cc,
                row_packing=not row_packing,
                row_size=row_size,
                col_size=col_size,
                **args,
            )
        else:
            pred_linear = PlainLinear(**args)

        subnet_layers = [pred_linear]
        self.pred_loss_net = Sequential(subnet_layers, debug=debug, trainable=trainable)
