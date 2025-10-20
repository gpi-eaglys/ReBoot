from abc import ABC, abstractmethod

import numpy as np
import torch


class LRScheduler(ABC):
    def __init__(
        self, base_lr: float, from_epoch: int = 0, verbose: bool = False
    ) -> None:
        self.last_epoch: int = -1
        self.base_lr: float = base_lr
        self.from_epoch: int = from_epoch
        self.verbose: bool = verbose
        self.lr: float = base_lr

    def get_lr(self) -> float:
        return self.lr

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    @abstractmethod
    def _compute_lr(self) -> float:
        """Compute learning rate using the scheduler"""
        pass

    def print_lr(self) -> None:
        print(f"Epoch {self.last_epoch:>2}: adjusting learning rate to {self.lr}")

    def step(self) -> None:
        self.last_epoch += 1
        old_lr = self.lr
        self.lr = self._compute_lr()

        # Log the learning rate only when it changes
        if self.verbose and old_lr != self.lr:
            self.print_lr()


class ConstantLR(LRScheduler):
    def _compute_lr(self) -> float:
        return self.lr


class LinearLR(LRScheduler):
    """
    Decay the learning rate by a fixed percentage of the base learning rate at each epoch,
    until a minimum learning rate is reached.

    Args:
        base_lr (float): Base learning rate
        factor (float): Percentage of the base learning rate to be reduced at each epoch
        minimum_lr (float): Maximum learning rate
    """

    def __init__(
        self,
        base_lr: float,
        factor: float,
        minimum_lr: float,
        from_epoch: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(base_lr, from_epoch=from_epoch, verbose=verbose)
        self.factor = factor
        self.minimum_lr = minimum_lr

    def _compute_lr(self) -> float:
        if self.last_epoch <= self.from_epoch:
            return self.base_lr
        else:
            return np.maximum(self.lr + self.base_lr * self.factor, self.minimum_lr)


class ExponentialLR(LRScheduler):
    """
    Decay the learning rate by a factor of the current learning rate at each epoch,
    until a minimum learning rate is reached.

    Args:
        base_lr (float): Base learning rate
        factor (int): Percentage of the current learning rate to be reduced at each epoch
        minimum_lr (float): Maximum learning rate
    """

    def __init__(
        self,
        base_lr: float,
        factor: float,
        minimum_lr: float,
        from_epoch: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(base_lr, from_epoch=from_epoch, verbose=verbose)
        self.factor = factor
        self.minimum_lr = minimum_lr

    def _compute_lr(self) -> float:
        if self.last_epoch <= self.from_epoch:
            return self.base_lr
        else:
            return np.maximum(self.lr * self.factor, self.minimum_lr)


class HalvingLR(LRScheduler):
    """
    Decay the learning rate by a specified factor after a specified number of epochs, until a minimum learning rate is reached.

    Args:
        base_lr (float): Base learning rate
        factor (float): Factor to decay the learning rate by
        num_epochs (int): Number of epochs after which to decay the learning rate
        minimum_lr (float): Maximum learning rate
    """

    def __init__(
        self,
        base_lr: float,
        num_epochs: int,
        minimum_lr: float,
        from_epoch: int = 0,
        factor: float = 0.5,
        verbose: bool = False,
    ) -> None:
        super().__init__(base_lr, from_epoch=from_epoch, verbose=verbose)
        self.num_epochs = num_epochs
        self.minimum_lr = minimum_lr
        self.factor = factor

    def _compute_lr(self) -> float:
        if self.last_epoch <= self.from_epoch:
            return self.base_lr
        else:
            if (self.last_epoch - self.from_epoch) % self.num_epochs == 0:
                return np.maximum(self.lr * self.factor, self.minimum_lr)
            else:
                return self.lr


class CosineLR(LRScheduler):
    """
    Decay the learning rate using a cosine annealing scheduling as proposed in Loshchilov & Hutter, (2017).
    Warm restarts are used to escape local minima.

    Args:
        base_lr (float): Base learning rate
        factor (float): Factor to decay the learning rate by
        num_epochs (int): Number of epochs after which to decay the learning rate
        minimum_lr (float): Maximum learning rate
    """

    def __init__(
        self,
        base_lr: float,
        steps_before_restart: int,
        minimum_lr: float = 1e-7,
        T_mult: int = 1,
        from_step: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(base_lr, from_epoch=from_step, verbose=verbose)
        self.steps_before_restart = steps_before_restart
        self.minimum_lr = minimum_lr
        self.T_mult = T_mult

        self.optimizer = torch.optim.SGD(
            [torch.nn.Parameter(torch.randn(1, 1))], lr=base_lr
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=self.steps_before_restart,
            T_mult=self.T_mult,
            eta_min=self.minimum_lr,
        )

    def _compute_lr(self) -> float:
        if self.last_epoch <= self.from_epoch:
            return self.base_lr
        else:
            self.scheduler.step(self.last_epoch)
            return self.scheduler.get_last_lr()[0]
