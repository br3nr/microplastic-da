import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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


def load_raw_csv(path):
    df = pd.read_csv(path, header=0)
    labels = df.iloc[:, 0].to_numpy()
    features = df.iloc[:, 1:].to_numpy(dtype=float)
    return features, labels


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


def make_dataloader(features, labels, batch_size, shuffle, diff_order, diff_interval):
    """Preprocess and wrap arrays into a DataLoader."""
    features, labels = preprocess_data(
        features,
        labels,
        shuffle=shuffle,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )
    shape = features.shape[1]
    dataset = SpectraDataset(features, labels)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, shape


def create_dataloaders(
    data_dir="data",
    batch_size=64,
    diff_order=6,
    diff_interval=4,
    seed=42,
):
    """Load raw CSVs, split marine into train/val/test, and return dataloaders."""
    marine_x, marine_y = load_raw_csv(f"{data_dir}/marine_polymers.csv")
    std_x, std_y = load_raw_csv(f"{data_dir}/std_polymers.csv")

    # Split marine into train/val/test (70/15/15), stratified by label
    train_x, temp_x, train_y, temp_y = train_test_split(
        marine_x,
        marine_y,
        test_size=0.3,
        stratify=marine_y,
        random_state=seed,
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x,
        temp_y,
        test_size=0.5,
        stratify=temp_y,
        random_state=seed,
    )

    print(
        f"Data: {len(train_x)} train, {len(val_x)} val, {len(test_x)} test, {len(std_x)} std"
    )

    train_loader, shape = make_dataloader(
        train_x,
        train_y,
        batch_size,
        shuffle=True,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )
    val_loader, _ = make_dataloader(
        val_x,
        val_y,
        batch_size,
        shuffle=False,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )
    test_loader, _ = make_dataloader(
        test_x,
        test_y,
        batch_size,
        shuffle=False,
        diff_order=diff_order,
        diff_interval=diff_interval,
    )
    std_loader, _ = make_dataloader(
        std_x,
        std_y,
        batch_size,
        shuffle=False,
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
