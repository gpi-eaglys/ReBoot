from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import InterpolationMode
from ucimlrepo import fetch_ucirepo

from reboot.utils.enums import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification


def download_dataset(
    config: dict, dataset: Dataset, data_path: str, pad_data: bool = True
) -> tuple[
    tuple[NDArray[np.float32], NDArray[np.float32]],
    tuple[NDArray[np.float32], NDArray[np.float32]],
]:
    """
    Helper function which downloads the dataset using torchvision.
    The dataset is pre-processed so that it is compatible with NumPy, and it is of data type uint8.

    Parameters:
        config (dict): A dictionary with the configuration parameters.
        dataset (Dataset): Enum representing dataset to be downloaded.
        data_path (str): Path to the data folder.
        pad_data (bool, optional): Whether to add padding so that datasets with 28x28 images become 32x32. Defaults to True.

    Returns:
        tuple[tuple[NDArray[np.float32], NDArray[np.float32]], tuple[NDArray[np.float32], NDArray[np.float32]]]: A tuple containing the train dataset and the test dataset, further split into data and labels.
    """

    def process_grayscale(
        train_split, test_split
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        train_split = train_split.data.numpy()
        test_split = test_split.data.numpy()
        train_split = np.expand_dims(train_split, axis=1)
        test_split = np.expand_dims(test_split, axis=1)
        pad_width = np.array(((0, 0), (0, 0), (2, 2), (2, 2)))
        if pad_data:
            train_split = np.pad(
                train_split, pad_width=pad_width, mode="constant", constant_values=0
            )
            test_split = np.pad(
                test_split, pad_width=pad_width, mode="constant", constant_values=0
            )
        return train_split, test_split

    train_args: dict[str, Any] = dict(root=data_path, train=True, download=True)
    test_args: dict[str, Any] = dict(root=data_path, train=False, download=True)

    if dataset.is_image():
        match dataset:
            case Dataset.MNIST:
                train_set = torchvision.datasets.MNIST(**train_args)
                test_set = torchvision.datasets.MNIST(**test_args)
                train_data, test_data = process_grayscale(train_set.data, test_set.data)
                train_targets = np.array(train_set.targets)
                test_targets = np.array(test_set.targets)
                
            case Dataset.CROPPED_MNIST:
                train_set = torchvision.datasets.MNIST(**train_args)
                test_set = torchvision.datasets.MNIST(**test_args)
                
                # Crop the images to 24x24
                train_set.data = train_set.data[:, 6:22, 6:22]
                test_set.data = test_set.data[:, 6:22, 6:22]
                
                # Resize to 8x8 using bilinear interpolation
                train_data_resized = F.interpolate(train_set.data.unsqueeze(1).float(), size=(8, 8), mode='bilinear').squeeze(1)
                test_data_resized = F.interpolate(test_set.data.unsqueeze(1).float(), size=(8, 8), mode='bilinear').squeeze(1)


                train_data, test_data = process_grayscale(train_data_resized, test_data_resized)
                train_targets = np.array(train_set.targets)
                test_targets = np.array(test_set.targets)

            case Dataset.TMNIST:
                train_set = torchvision.datasets.MNIST(**train_args)
                test_set = torchvision.datasets.MNIST(**test_args)
                train_data, test_data = process_grayscale(train_set.data, test_set.data)
                train_targets = np.array(train_set.targets)
                test_targets = np.array(test_set.targets)

                # Create Ternary classification dataset
                train_indexes, test_indexes = [], []
                for i in range(len(train_data)):
                    if train_targets[i] == 0 or train_targets[i] == 1 or train_targets[i] == 2:
                        train_indexes.append(i)
                for i in range(len(test_data)):
                    if test_targets[i] == 0 or test_targets[i] == 1 or test_targets[i] == 2:
                        test_indexes.append(i)
                train_data, train_targets = train_data[train_indexes], train_targets[train_indexes]
                test_data, test_targets = test_data[test_indexes], test_targets[test_indexes]

                # Crop the images
                train_data = train_data[:, :, 6:22, 6:22]
                test_data = test_data[:, :, 6:22, 6:22]
                
                # Remove the last 5000 samples for validation
                val_images = 5000
                idx_train = len(train_data) - val_images
                train_data, train_targets = train_data[:idx_train], train_targets[:idx_train]
                
                # Randomly select 50 samples for training, with a seed for reproducibility
                rng = np.random.default_rng(0)
                indices = rng.choice(len(train_data), size=50, replace=False)
                train_data, train_targets = train_data[indices], train_targets[indices]
                
                # Shuffle the train data and targets
                train_data, train_targets = shuffle_dataset(train_data, train_targets)

                # Apply MaxPooling to reduce to 4x4 images
                pool = torch.nn.MaxPool2d(kernel_size=4, stride=4)
                train_data = pool(torch.tensor(train_data)).numpy()
                test_data = pool(torch.tensor(test_data)).numpy()

            case Dataset.FASHION_MNIST:
                train_set = torchvision.datasets.FashionMNIST(**train_args)
                test_set = torchvision.datasets.FashionMNIST(**test_args)
                train_data, test_data = process_grayscale(train_set.data, test_set.data)
                train_targets = np.array(train_set.targets)
                test_targets = np.array(test_set.targets)

            case Dataset.KUZUSHIJI_MNIST:
                train_set = torchvision.datasets.KMNIST(**train_args)
                test_set = torchvision.datasets.KMNIST(**test_args)
                train_data, test_data = process_grayscale(train_set.data, test_set.data)
                train_targets = np.array(train_set.targets)
                test_targets = np.array(test_set.targets)

            case _:
                raise ValueError(f"Invalid dataset: {dataset}")

    elif dataset.is_tabular():
        match dataset:
            case Dataset.BREAST_CANCER:
                df = fetch_ucirepo(id=17)
                X = df.data.features
                y = df.data.targets

                # Balance the classes by resampling
                ros = RandomOverSampler()
                X, y = ros.fit_resample(X, y)
                # Perform train-test split
                train_data, test_data, train_targets, test_targets = train_test_split(
                    X, y, test_size=0.2, stratify=y
                )
                # Standardize the data
                scaler = StandardScaler()
                train_data = scaler.fit_transform(train_data)
                test_data = scaler.transform(test_data)

            case Dataset.LETTER_RECOGNITION:
                df = fetch_ucirepo(id=59)
                X = df.data.features
                y = df.data.targets

                # # Balance the classes by resampling
                # ros = RandomOverSampler()
                # X, y = ros.fit_resample(X, y)
                # Perform train-test split
                train_data, test_data, train_targets, test_targets = train_test_split(
                    X, y, test_size=0.2, stratify=y
                )
                # Standardize the data
                scaler = StandardScaler()
                train_data = scaler.fit_transform(train_data)
                test_data = scaler.transform(test_data)

            case Dataset.HEART_DISEASE:
                df = fetch_ucirepo(id=45)
                X = df.data.features
                y = df.data.targets

                # Drop NaN values
                nan_index = X[X.isna().any(axis=1)].index
                X = X.drop(nan_index)
                y = y.drop(nan_index)
                # Balance the classes by resampling
                ros = RandomOverSampler()
                X, y = ros.fit_resample(X, y)
                cat_features = [
                    "sex",
                    "cp",
                    "fbs",
                    "restecg",
                    "exang",
                    "slope",
                    "ca",
                    "thal",
                ]
                num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
                # One-hot encode categorical features
                X[cat_features] = X[cat_features].astype("category")
                X = pd.get_dummies(X, columns=cat_features, drop_first=False)
                # Perform train-test split
                train_data, test_data, train_targets, test_targets = train_test_split(
                    X, y, test_size=0.2, stratify=y
                )
                # Standardize continuous features
                scaler = StandardScaler()
                train_data[num_features] = scaler.fit_transform(
                    train_data[num_features]
                )
                test_data[num_features] = scaler.transform(test_data[num_features])
                train_data = train_data.to_numpy()
                test_data = test_data.to_numpy()
                train_targets = train_targets.astype("category")
                test_targets = test_targets.astype("category")

            case Dataset.IRIS:
                df = fetch_ucirepo(id=53)
                X = df.data.features
                y = df.data.targets

                # Perform train-test split
                train_data, test_data, train_targets, test_targets = train_test_split(
                    X, y, test_size=0.2, stratify=y
                )
                # Standardize the data
                scaler = StandardScaler()
                train_data = scaler.fit_transform(train_data)
                test_data = scaler.transform(test_data)
                
            case Dataset.PENGUINS:
                df = sns.load_dataset("penguins")
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                X = df.drop(columns=["species"])
                y = df["species"]
                
                # Drop NaN values
                nan_index = X[X.isna().any(axis=1)].index
                X = X.drop(nan_index)
                y = y.drop(nan_index)
                # Feature selection
                cat_features = ["island"]
                num_features = ["bill_length_mm", "flipper_length_mm", "body_mass_g"]
                X = X[cat_features + num_features]
                # Label encode categorical features
                for col in cat_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                # Perform train-test split
                train_dim, val_dim = 150, 64 
                train_data, train_targets = X[:train_dim], y[:train_dim]
                test_data, test_targets = X[train_dim+val_dim:], y[train_dim+val_dim:]
                # Standardize continuous features
                scaler = StandardScaler()
                train_data[num_features] = scaler.fit_transform(train_data[num_features])
                test_data[num_features] = scaler.transform(test_data[num_features])
                # Convert to numpy arrays
                train_data = train_data.to_numpy()
                test_data = test_data.to_numpy()
                
            case Dataset.ONE_CLASS:
                # Create a toy dataset with 1 input feature and 1 output feature
                X, y = make_classification(n_samples=5000, n_features=1, n_classes=1, n_informative=9, random_state=42)
                train_data, test_data, train_targets, test_targets = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                # Normalize the input features
                scaler = StandardScaler()
                train_data = scaler.fit_transform(train_data)
                test_data = scaler.transform(test_data)
                
            case Dataset.CUSTOM:
                # Create a toy dataset with the specified number of input and output features
                X, y = make_classification(n_samples=5000, n_features=config["input_dim"], n_classes=config["output_dim"], n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42)
                train_data, test_data, train_targets, test_targets = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                # Normalize the input features
                scaler = StandardScaler()
                train_data = scaler.fit_transform(train_data)
                test_data = scaler.transform(test_data)
                
            case _:
                raise ValueError(f"Invalid dataset: {dataset}")

        # One-hot encode the targets
        train_targets = pd.get_dummies(train_targets).to_numpy()
        test_targets = pd.get_dummies(test_targets).to_numpy()

    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    return (
        (train_data.astype(np.float32), train_targets.astype(np.uint8)),
        (test_data.astype(np.float32), test_targets.astype(np.uint8)),
    )


def shuffle_dataset(
    X: NDArray[np.float32], y: NDArray[np.float32], seed: int | None = None
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Shuffle the dataset using a fixed seed if needed.

    Args:
        X (NDArray[np.float32]): Data to be shuffled
        y (NDArray[np.float32]): Labels to be shuffled
        seed (int, optional): Random seed to be used for shuffling. Defaults to None.

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32]]: The shuffled data and labels
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate and shuffle the set of indices
    idx_list = np.arange(len(X))
    np.random.shuffle(idx_list)

    # Apply the new ordering to the dataset
    X = X[idx_list]
    y = y[idx_list]
    return X, y


def plot_images(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    num_img: int = 6,
    seed: int | None = None,
) -> None:
    """
    Display a sample of images from the training-validation dataset.

    Args:
        x_train (NDArray[np.float32]): Training data
        y_train (NDArray[np.float32]): Training labels
        num_img (int, optional): Number of images to display. Defaults to 6.
        seed (int, optional): Random seed to be used for shuffling. Defaults to None.
    """
    _, axes = plt.subplots(1, num_img, figsize=(24, 20))

    if seed is not None:
        np.random.seed(seed)

    # Iterate through the selected number of images
    for i in range(num_img):
        sample_id = np.random.randint(0, len(x_train))
        ax = axes[i % num_img]
        image = x_train[sample_id]
        ax.imshow(image.transpose(1, 2, 0), cmap="gray")
        ax.set_title(y_train[sample_id])
        ax.axis("off")

    # Adjust layout and display the images
    plt.tight_layout()
    plt.show()


def normalize_channels(
    x_train, x_val, x_test
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Normalize all the data using statistics computed on the training set.
    The data is normalized so that it has a zero-mean and unitary standard deviation.

    Args:
        x_train (NDArray[np.float32]): Training data
        x_val (NDArray[np.float32]): Validation data
        x_test (NDArray[np.float32]): Test data
        print_stats (bool, optional): Whether to print the new statistics after normalization. Defaults to True.

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: The normalized training, validation, and test data
    """
    # To avoid overflow, use a subset of the data to compute statistics
    train_mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    train_std = x_train.std(axis=(0, 2, 3), keepdims=True)

    def normalize(x):
        # Apply standardization
        return (x - train_mean) / (train_std + 1e-7)

    x_train = normalize(x_train)
    if x_val is not None:
        x_val = normalize(x_val)
    x_test = normalize(x_test)

    return x_train, x_val, x_test


def load_float_dataset(
    config,
    data_path: str,
    val_dim: int = 0,
    test_dim: int = 10_000,
    show_images: bool = True,
    show_log: bool = True,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """
    Load, subsample, normalize, and move data to the GPU if required.

    Args:
        config (dict): A dictionary with the configuration parameters.
        data_path (str): Path to the data folder.
        val_dim (int, optional): Desired size of the validation set. Defaults to 0.
        test_dim (int, optional): Desired size of the test set. Defaults to 10_000.
        show_images (bool, optional): Whether to display a sample of images from the training dataset. Defaults to True.
        show_log (bool, optional): Whether to print information about the dataset. Defaults to True.

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        Tuple containing the training, validation, and test data and labels.
    """
    return _load_dataset(
        config,
        val_dim=val_dim,
        test_dim=test_dim,
        show_images=show_images,
        show_log=show_log,
        data_path=data_path,
    )


def _load_dataset(
    config: dict,
    data_path: str,
    val_dim: int = 0,
    test_dim: int = 10_000,
    show_images: bool = True,
    show_log: bool = True,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """
    Load, subsample, normalize, and move data to the GPU if required.

    Args:
        config (dict): A dictionary with the configuration parameters.
        data_path (str): Path to the data folder.
        val_dim (int, optional): Desired size of the validation set. Defaults to 0.
        test_dim (int, optional): Desired size of the test set. Defaults to 10_000.
        show_images (bool, optional): Whether to display a sample of images from the training dataset. Defaults to True.
        show_log (bool, optional): Whether to print information about the dataset. Defaults to True.

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        Tuple containing the training, validation, and test data and labels.
    """
    (train_data, train_labels), (test_data, test_labels) = download_dataset(
        config,
        Dataset[config["dataset"]],
        pad_data=config["pad_data"],
        data_path=data_path,
    )

    is_image_dataset = Dataset[config["dataset"]].is_image()

    if config["subsample"] is not None:
        # Subsample if required
        train_data = train_data[: config["subsample"]]
        train_labels = train_labels[: config["subsample"]]

        if show_log:
            print(f"Subsampling applied: training on {len(train_data)} samples")
        else:
            print(f"Training on {len(train_data)} samples")

    if is_image_dataset and config["resize"] is not None:
        resize = torchvision.transforms.Resize(
            config["resize"], interpolation=InterpolationMode.BICUBIC
        )
        train_data = resize(torch.tensor(train_data)).numpy()
        test_data = resize(torch.tensor(test_data)).numpy()

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _preprocess_dataset(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        val_dim=val_dim,
        test_dim=test_dim,
    )

    if config["normalize"]:
        # Normalize the data if required
        x_train, x_val, x_test = normalize_channels(x_train, x_val, x_test)

    if config["pad_labels"]:
        y_train, y_val, y_test = pad_labels(y_train, y_val, y_test)

    # Cast data and labels to float32
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    if x_val is not None:
        x_val = x_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    if show_log:
        print(f"Loaded dataset: {config['dataset']}")
        print(f"Train set:  {x_train.shape}, {y_train.shape}")
        if x_val is not None:
            print(f"Val set:    {x_val.shape}, {y_val.shape}")
        print(f"Test set:   {x_test.shape}, {y_test.shape}")
        print(f"Data types: ({x_train.dtype}, {y_train.dtype})")

        if is_image_dataset:
            train_min = x_train.min(axis=(0, 2, 3), keepdims=False).squeeze()
            train_max = x_train.max(axis=(0, 2, 3), keepdims=False).squeeze()
            train_mean = x_train.mean(axis=(0, 2, 3), keepdims=False).squeeze()
            train_std = x_train.std(axis=(0, 2, 3), keepdims=False).squeeze()

            print(
                f"\nTrain data statistics:\n"
                f" Min: {train_min: .5f} - Max: {train_max: .5f}\n"
                f"Mean: {train_mean: .5f} - Std: {train_std: .5f}"
            )

    if is_image_dataset and show_images:
        images = x_train
        plot_images(images, y_train, num_img=6)

    # Reshape the data so that it has two dimensions
    x_train = x_train.reshape(x_train.shape[0], -1)
    if x_val is not None:
        x_val = x_val.reshape(x_val.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, x_val, x_test, y_train, y_val, y_test


def _preprocess_dataset(
    train_data,
    train_labels,
    test_data,
    test_labels,
    val_dim: int = 0,
    test_dim: int = 10_000,
) -> tuple[tuple[NDArray[np.float32], NDArray[np.float32]], ...]:
    """
    Prepare a train-validation-test split, scale the data in the correct range, and apply one-hot encoding.

    Args:
        train_data (NDArray[np.float32]): The training data
        train_labels (NDArray[np.float32]): The training labels
        test_data (NDArray[np.float32]): The test data
        test_labels (NDArray[np.float32]): The test labels
        val_dim (int, optional): Number of samples to be used in the validation set. Defaults to 0
        test_dim (int, optional): Number of samples to be used in the test set. Defaults to 10_000

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32]], tuple[NDArray[np.float32], NDArray[np.float32]], tuple[NDArray[np.float32], NDArray[np.float32]]:
        Tuple containing the training, validation, and test data and labels
    """
    # Shuffle train set
    train_data, train_labels = shuffle_dataset(train_data, train_labels, seed=42)

    if val_dim != 0:
        # Train-validation-test split
        train_data, val_data = train_data[:-val_dim], train_data[-val_dim:]
        train_labels, val_labels = train_labels[:-val_dim], train_labels[-val_dim:]
    else:
        # Train-test split
        val_data, val_labels = None, None

    test_data, test_labels = test_data[:test_dim], test_labels[:test_dim]

    return _preprocess_float_dataset(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
    )


def _preprocess_float_dataset(
    train_data, train_labels, val_data, val_labels, test_data, test_labels
) -> tuple[tuple[NDArray[np.float32], NDArray[np.float32]], ...]:
    """
    Scale the data in the correct range and apply one-hot encoding.

    Args:
        train_data (NDArray[np.float32]): Training data.
        train_labels (NDArray[np.float32]): Training labels.
        val_data (NDArray[np.float32]): Validation data.
        val_labels (NDArray[np.float32]): Validation labels.
        test_data (NDArray[np.float32]): Test data.
        test_labels (NDArray[np.float32]): Test labels.

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32]], tuple[NDArray[np.float32], NDArray[np.float32]], tuple[NDArray[np.float32], NDArray[np.float32]]:
        Tuple containing the training, validation, and test data and labels.
    """
    # Train set (scaled to [0, 1])
    train_min, train_max = train_data.min(), train_data.max()
    train_data = (train_data - train_min) / (train_max - train_min)
    if val_data is not None:
        val_data = (val_data - train_min) / (train_max - train_min)
    test_data = (test_data - train_min) / (train_max - train_min)

    # One-hot encoding of labels
    if train_labels.ndim == 1:
        num_classes = len(np.unique(train_labels))
        train_labels = np.eye(num_classes)[train_labels]
        if val_data is not None:
            val_labels = np.eye(num_classes)[val_labels]
        test_labels = np.eye(num_classes)[test_labels]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def pad_labels(
    y_train: NDArray[np.float32],
    y_val: NDArray[np.float32],
    y_test: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Pad the labels with zeros to the next power of 2.

    Args:
        y_train (NDArray[np.float32]): Training labels
        y_val (NDArray[np.float32]): Validation labels
        y_test (NDArray[np.float32]): Test labels

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: Padded training, validation, and test labels
    """
    num_classes = y_train.shape[1]
    next_power_of_2 = 2 ** (num_classes - 1).bit_length()
    pad_width = next_power_of_2 - num_classes

    y_train = np.pad(y_train, ((0, 0), (0, pad_width)))
    if y_val is not None:
        y_val = np.pad(y_val, ((0, 0), (0, pad_width)))
    y_test = np.pad(y_test, ((0, 0), (0, pad_width)))

    return y_train, y_val, y_test
