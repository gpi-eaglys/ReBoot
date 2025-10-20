from abc import ABC
from enum import Enum, auto


class NonLinearity(Enum):
    SQUARE = auto()
    POLY_RELU = auto()


class OptimizerName(Enum):
    SGD = auto()
    PLAIN_NESTEROV_SGD = auto()
    ENCRYPTED_NESTEROV_SGD = auto()


class Dataset(Enum):
    MNIST = auto()
    CROPPED_MNIST = auto()
    FASHION_MNIST = auto()
    KUZUSHIJI_MNIST = auto()
    TMNIST = auto()

    BREAST_CANCER = auto()
    LETTER_RECOGNITION = auto()
    HEART_DISEASE = auto()
    IRIS = auto()
    PENGUINS = auto()
    ONE_CLASS = auto()
    CUSTOM = auto()
    
    def is_image(self) -> bool:
        return self in {Dataset.MNIST, Dataset.FASHION_MNIST, Dataset.KUZUSHIJI_MNIST, Dataset.TMNIST, Dataset.CROPPED_MNIST}

    def is_tabular(self) -> bool:
        return self in {
            Dataset.BREAST_CANCER,
            Dataset.LETTER_RECOGNITION,
            Dataset.HEART_DISEASE,
            Dataset.IRIS,
            Dataset.PENGUINS,
            Dataset.ONE_CLASS,
            Dataset.CUSTOM,
        }
