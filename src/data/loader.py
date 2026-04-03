import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

LABEL_DICT = {"PP": 0, "PE": 1}


class SpectraDataset(Dataset):
    """PyTorch Dataset for spectral data."""

    def __init__(self, features, labels):
        self.data = []
        for i in range(len(features)):
            self.data.append([features[i], labels[i]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feature_tensor, label_tensor


def load_data(path_x, path_y, sep=","):
    data_x = pd.read_csv(path_x, header=0, dtype=float, sep=sep).to_numpy()
    data_y = pd.read_csv(path_y, header=0).to_numpy()
    return data_x, data_y


def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff


def get_norder_diff(dataset, order, diff_val):
    """Apply n-th order differencing to dataset."""
    differences = []
    for i in range(dataset.shape[0]):
        diff = dataset[i]
        for _ in range(order):
            diff = difference(diff, interval=diff_val)
        differences.append(diff)
    return np.asarray(differences)


def preprocess_data(data_x, data_y, shuffle=True, diff_order=6, diff_interval=4):
    # Apply differencing
    data_x = get_norder_diff(data_x, diff_order, diff_interval)

    # Reshape to (n_samples, n_features, 1)
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1], 1))
    data_y = data_y.reshape((data_y.shape[0]))

    # Shuffle
    if shuffle:
        idx = np.random.permutation(len(data_x))
        data_x = data_x[idx]
        data_y = data_y[idx]

    # Convert string labels to integers
    data_y = np.array([LABEL_DICT[label] for label in data_y])
    data_x = np.asarray(data_x).astype("float32")

    return data_x, data_y


def get_dataloader(
    path_x, path_y, batch_size, shuffle=True, diff_order=6, diff_interval=4, sep=","
):
    data_x, data_y = load_data(path_x, path_y, sep=sep)
    data_x, data_y = preprocess_data(
        data_x,
        data_y,
        shuffle=shuffle,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )

    shape = data_x.shape[1]
    dataset = SpectraDataset(data_x, data_y)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, shape


def get_test_dataloader(
    path_x, path_y, batch_size, diff_order=6, diff_interval=4, sep=","
):
    data_x, data_y = load_data(path_x, path_y, sep=sep)
    data_x, data_y = preprocess_data(
        data_x,
        data_y,
        shuffle=False,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )

    dataset = SpectraDataset(data_x, data_y)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader


class DataConfig:
    """Configuration for data paths."""

    def __init__(self, data_dir="data/no_fp"):
        self.train_x = f"{data_dir}/train/train-x.csv"
        self.train_y = f"{data_dir}/train/train-y.csv"
        self.val_x = f"{data_dir}/val/val-x.csv"
        self.val_y = f"{data_dir}/val/val-y.csv"
        self.test_x = f"{data_dir}/test/test-x.csv"
        self.test_y = f"{data_dir}/test/test-y.csv"
        self.std_x = f"{data_dir}/std/std-x.csv"
        self.std_y = f"{data_dir}/std/std-y.csv"


def create_dataloaders(
    data_dir="data/no_fp", batch_size=64, diff_order=6, diff_interval=4
):
    config = DataConfig(data_dir)

    train_loader, shape = get_dataloader(
        config.train_x,
        config.train_y,
        batch_size,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )

    val_loader, _ = get_dataloader(
        config.val_x,
        config.val_y,
        batch_size,
        shuffle=False,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )

    test_loader = get_test_dataloader(
        config.test_x,
        config.test_y,
        batch_size,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )

    std_loader = get_test_dataloader(
        config.std_x,
        config.std_y,
        batch_size,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "std": std_loader,
        "input_shape": shape,
    }
