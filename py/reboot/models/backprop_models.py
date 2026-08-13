from typing import Literal

import torch
from torch import nn


def create_non_linearity(
    non_linearity: Literal["relu", "leaky_relu", "gelu"] = "leaky_relu"
) -> nn.Module:
    match non_linearity:
        case "relu":
            return nn.ReLU(inplace=True)
        case "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        case "gelu":
            return nn.GELU()


class BackpropMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) network that can be trained by regular backpropagation.
    The network is instantiated and trained with PyTorch.
    """

    def __init__(
        self,
        num_layers: int,
        num_hidden_neurons: list[int],
        num_classes: int,
        input_dim: int,
        non_linearity: Literal["relu", "leaky_relu", "gelu"] = "gelu",
    ) -> None:
        if (num_layers - 1) != len(num_hidden_neurons):
            raise ValueError("The length of num_hidden_neurons must be num_layers - 1")
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.num_classes = num_classes
        self.non_linearity = non_linearity

        # Instantiate hidden layers with a list comprehension
        self.layers = nn.ModuleList(
            [
                self._make_linear_block(
                    input_dim=(input_dim if i == 0 else num_hidden_neurons[i - 1]),
                    output_dim=num_hidden_neurons[i],
                )
                for i in range(0, num_layers - 1)
            ]
        )

        # Instantiate the last layer
        self.layer_out = nn.Linear(
            in_features=(input_dim if num_layers == 1 else num_hidden_neurons[-1]),
            out_features=num_classes,
        )

    def _make_linear_block(self, input_dim: int, output_dim: int) -> nn.Sequential:
        layers = []
        linear = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.kaiming_uniform_(linear.weight, mode="fan_in")
        layers.append(linear)
        layers.append(create_non_linearity(self.non_linearity))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the layers
        for layer in self.layers:
            x = layer(x)
        x = self.layer_out(x)
        return x
