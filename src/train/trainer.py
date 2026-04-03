"""Training utilities for domain adaptation methods."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

from ..models import Transformer, DeepCORAL, DSAN, DANN, compute_lambda
from ..data import create_dataloaders
from ..utils import wandb_utils


# Tuned via experiments/*_stability.py scripts
# CORAL: ~98.6% test, ~93.6% std
# DSAN: ~97.7% test, ~93.8% std
# DANN: ~98.6% test, ~94.8% std
METHOD_DEFAULTS = {
    "dann": {
        "epochs": 10,
        "batch_size": 64,
        "lr": 1e-5,
        "d_model": 64,
        "q": 8,
        "v": 8,
        "h": 4,
        "N": 2,
        "dropout": 0.2,
        "pe": False,
        "num_classes": 2,
        "diff_order": 6,
        "diff_interval": 4,
        "bottleneck_dim": 512,
    },
    "dsan": {
        "epochs": 10,
        "batch_size": 64,
        "lr": 5e-6,
        "d_model": 64,
        "q": 8,
        "v": 8,
        "h": 4,
        "N": 2,
        "dropout": 0.2,
        "pe": False,
        "num_classes": 2,
        "diff_order": 6,
        "diff_interval": 4,
        "bottleneck_dim": 512,
    },
    "coral": {
        "epochs": 10,
        "batch_size": 64,
        "lr": 1e-5,
        "d_model": 64,
        "q": 8,
        "v": 8,
        "h": 4,
        "N": 2,
        "dropout": 0.2,
        "pe": False,
        "num_classes": 2,
        "diff_order": 6,
        "diff_interval": 4,
        "bottleneck_dim": 512,
    },
    "vanilla": {
        "epochs": 25,
        "batch_size": 64,
        "lr": 5e-4,
        "d_model": 64,
        "q": 8,
        "v": 8,
        "h": 2,
        "N": 2,
        "dropout": 0.2,
        "pe": False,
        "num_classes": 2,
        "diff_order": 6,
        "diff_interval": 4,
        "bottleneck_dim": 256,
    },
}


def get_method_config(method, user_config=None):
    """Merge user config with method defaults."""
    defaults = METHOD_DEFAULTS.get(method, METHOD_DEFAULTS["vanilla"]).copy()
    if user_config:
        defaults.update(user_config)
    return defaults


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_transformer(input_shape, config):
    return Transformer(
        d_input=input_shape,
        d_channel=1,
        d_model=config.get("d_model", 64),
        d_output=config.get("num_classes", 2),
        q=config.get("q", 8),
        v=config.get("v", 8),
        h=config.get("h", 2),
        N=config.get("N", 2),
        dropout=config.get("dropout", 0.2),
        pe=config.get("pe", False),
        bottleneck_dim=config.get("bottleneck_dim", 256),
    )


def evaluate(model, dataloader, device, flag="test"):
    model.eval()
    predictions, actuals = [], []
    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model.predict(x)
            loss = F.nll_loss(F.log_softmax(pred, dim=1), y).item()
            total_loss += loss

            pred_labels = pred.data.max(1)[1]
            predictions.extend(pred_labels.tolist())
            actuals.extend(y.tolist())

            total += y.size(0)
            correct += (pred_labels == y).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)

    wandb_utils.log({f"{flag}_loss": avg_loss, f"{flag}_accuracy": accuracy})

    return accuracy, predictions, actuals


def train_coral(config, dataloaders, device):
    input_shape = dataloaders["input_shape"]

    transformer = create_transformer(input_shape, config)
    model = DeepCORAL(transformer, num_classes=config.get("num_classes", 2))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    wandb_utils.watch(model)
    wandb_utils.update_config(config)

    train_loader = dataloaders["train"]
    std_loader = dataloaders["std"]

    print(f"Training Deep CORAL for {config['epochs']} epochs...")

    for epoch in range(config["epochs"]):
        model.train()
        target_iter = iter(std_loader)

        for batch_idx, (src_x, src_y) in enumerate(train_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)

            try:
                trg_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(std_loader)
                trg_x, _ = next(target_iter)

            if src_x.shape[0] != trg_x.shape[0]:
                continue

            trg_x = trg_x.to(device)

            optimizer.zero_grad()
            src_pred, coral_loss = model(src_x, trg_x)

            cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_y)
            lambd = 2 / (1 + math.exp(-10 * epoch / config["epochs"])) - 1
            loss = cls_loss + lambd * coral_loss

            loss.backward()
            optimizer.step()

            wandb_utils.log({"loss": loss.item(), "coral_loss": coral_loss.item()})

        # Validation
        val_acc, _, _ = evaluate(model, dataloaders["val"], device, "val")
        std_acc, _, _ = evaluate(model, dataloaders["std"], device, "std")
        print(
            f"Epoch {epoch + 1}/{config['epochs']} - Val: {val_acc:.1f}%, Std: {std_acc:.1f}%"
        )

    # Final evaluation
    print("\n=== Final Evaluation ===")
    test_acc, test_pred, test_actual = evaluate(
        model, dataloaders["test"], device, "test"
    )
    std_acc, std_pred, std_actual = evaluate(model, dataloaders["std"], device, "std")

    print(f"\nTest Set Accuracy: {test_acc:.2f}%")
    print(classification_report(test_actual, test_pred, target_names=["PP", "PE"]))

    print(f"\nStandards Set Accuracy: {std_acc:.2f}%")
    print(classification_report(std_actual, std_pred, target_names=["PP", "PE"]))

    return model


def train_dsan(config, dataloaders, device):
    input_shape = dataloaders["input_shape"]

    transformer = create_transformer(input_shape, config)
    model = DSAN(transformer, num_classes=config.get("num_classes", 2))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    wandb_utils.watch(model)
    wandb_utils.update_config(config)

    train_loader = dataloaders["train"]
    std_loader = dataloaders["std"]

    print(f"Training DSAN for {config['epochs']} epochs...")

    for epoch in range(config["epochs"]):
        model.train()
        target_iter = iter(std_loader)

        for batch_idx, (src_x, src_y) in enumerate(train_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)

            try:
                trg_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(std_loader)
                trg_x, _ = next(target_iter)

            if src_x.shape[0] != trg_x.shape[0]:
                continue

            trg_x = trg_x.to(device)

            optimizer.zero_grad()
            src_pred, loss_lmmd = model(src_x, trg_x, src_y)

            cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_y)
            lambd = 2 / (1 + math.exp(-10 * epoch / config["epochs"])) - 1
            loss = cls_loss + 0.5 * lambd * loss_lmmd

            loss.backward()
            optimizer.step()

            wandb_utils.log({"loss": loss.item(), "lmmd_loss": loss_lmmd.item()})

        # Validation
        val_acc, _, _ = evaluate(model, dataloaders["val"], device, "val")
        std_acc, _, _ = evaluate(model, dataloaders["std"], device, "std")
        print(
            f"Epoch {epoch + 1}/{config['epochs']} - Val: {val_acc:.1f}%, Std: {std_acc:.1f}%"
        )

    # Final evaluation
    print("\n=== Final Evaluation ===")
    test_acc, test_pred, test_actual = evaluate(
        model, dataloaders["test"], device, "test"
    )
    std_acc, std_pred, std_actual = evaluate(model, dataloaders["std"], device, "std")

    print(f"\nTest Set Accuracy: {test_acc:.2f}%")
    print(classification_report(test_actual, test_pred, target_names=["PP", "PE"]))

    print(f"\nStandards Set Accuracy: {std_acc:.2f}%")
    print(classification_report(std_actual, std_pred, target_names=["PP", "PE"]))

    return model


def train_dann(config, dataloaders, device):
    input_shape = dataloaders["input_shape"]

    transformer = create_transformer(input_shape, config)
    model = DANN(transformer, num_classes=config.get("num_classes", 2))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    domain_criterion = nn.CrossEntropyLoss()

    wandb_utils.watch(model)
    wandb_utils.update_config(config)

    train_loader = dataloaders["train"]
    std_loader = dataloaders["std"]

    print(f"Training DANN for {config['epochs']} epochs...")

    for epoch in range(config["epochs"]):
        model.train()
        target_iter = iter(std_loader)

        alpha = compute_lambda(epoch, config["epochs"])

        for batch_idx, (src_x, src_y) in enumerate(train_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)

            try:
                trg_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(std_loader)
                trg_x, _ = next(target_iter)

            if src_x.shape[0] != trg_x.shape[0]:
                continue

            trg_x = trg_x.to(device)
            batch_size = src_x.shape[0]

            # Domain labels: 0 for source, 1 for target
            source_domain_label = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            target_domain_label = torch.ones(
                batch_size, dtype=torch.long, device=device
            )

            optimizer.zero_grad()

            src_pred, src_domain, tgt_domain = model(src_x, trg_x, alpha=alpha)

            cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_y)

            domain_loss_src = domain_criterion(src_domain, source_domain_label)
            domain_loss_tgt = domain_criterion(tgt_domain, target_domain_label)
            domain_loss = domain_loss_src + domain_loss_tgt

            loss = cls_loss + domain_loss

            loss.backward()
            optimizer.step()

            wandb_utils.log(
                {
                    "loss": loss.item(),
                    "cls_loss": cls_loss.item(),
                    "domain_loss": domain_loss.item(),
                    "alpha": alpha,
                }
            )

        # Validation
        val_acc, _, _ = evaluate(model, dataloaders["val"], device, "val")
        std_acc, _, _ = evaluate(model, dataloaders["std"], device, "std")
        print(
            f"Epoch {epoch + 1}/{config['epochs']} - Val: {val_acc:.1f}%, Std: {std_acc:.1f}%, alpha: {alpha:.3f}"
        )

    # Final evaluation
    print("\n=== Final Evaluation ===")
    test_acc, test_pred, test_actual = evaluate(
        model, dataloaders["test"], device, "test"
    )
    std_acc, std_pred, std_actual = evaluate(model, dataloaders["std"], device, "std")

    print(f"\nTest Set Accuracy: {test_acc:.2f}%")
    print(classification_report(test_actual, test_pred, target_names=["PP", "PE"]))

    print(f"\nStandards Set Accuracy: {std_acc:.2f}%")
    print(classification_report(std_actual, std_pred, target_names=["PP", "PE"]))

    return model


def train_vanilla(config, dataloaders, device):
    input_shape = dataloaders["input_shape"]

    model = create_transformer(input_shape, config)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    wandb_utils.watch(model)
    wandb_utils.update_config(config)

    train_loader = dataloaders["train"]

    print(f"Training Vanilla Transformer for {config['epochs']} epochs...")

    for epoch in range(config["epochs"]):
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            features = model(x)
            pred = model._linear(features)

            loss = F.nll_loss(F.log_softmax(pred, dim=1), y)
            loss.backward()
            optimizer.step()

            wandb_utils.log({"loss": loss.item()})

        # Validation
        val_acc = evaluate_vanilla(model, dataloaders["val"], device, "val")
        std_acc = evaluate_vanilla(model, dataloaders["std"], device, "std")
        print(
            f"Epoch {epoch + 1}/{config['epochs']} - Val: {val_acc:.1f}%, Std: {std_acc:.1f}%"
        )

    # Final evaluation
    print("\n=== Final Evaluation ===")
    test_acc, test_pred, test_actual = evaluate_vanilla_full(
        model, dataloaders["test"], device
    )
    std_acc, std_pred, std_actual = evaluate_vanilla_full(
        model, dataloaders["std"], device
    )

    print(f"\nTest Set Accuracy: {test_acc:.2f}%")
    print(classification_report(test_actual, test_pred, target_names=["PP", "PE"]))

    print(f"\nStandards Set Accuracy: {std_acc:.2f}%")
    print(classification_report(std_actual, std_pred, target_names=["PP", "PE"]))

    return model


def evaluate_vanilla(model, dataloader, device, flag="test"):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            features = model(x)
            pred = model._linear(features)
            pred_labels = pred.data.max(1)[1]
            total += y.size(0)
            correct += (pred_labels == y).sum().item()

    accuracy = 100 * correct / total
    wandb_utils.log({f"{flag}_accuracy": accuracy})
    return accuracy


def evaluate_vanilla_full(model, dataloader, device):
    model.eval()
    predictions, actuals = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            features = model(x)
            pred = model._linear(features)
            pred_labels = pred.data.max(1)[1]
            predictions.extend(pred_labels.tolist())
            actuals.extend(y.tolist())
            total += y.size(0)
            correct += (pred_labels == y).sum().item()

    return 100 * correct / total, predictions, actuals


TRAINERS = {
    "coral": train_coral,
    "dsan": train_dsan,
    "dann": train_dann,
    "vanilla": train_vanilla,
}


def train(method, config=None, data_dir="data"):
    if method not in TRAINERS:
        raise ValueError(
            f"Unknown method: {method}. Choose from {list(TRAINERS.keys())}"
        )

    merged_config = get_method_config(method, config)

    device = get_device()
    print(f"Using device: {device}")

    dataloaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=merged_config.get("batch_size", 64),
        diff_order=merged_config.get("diff_order", 6),
        diff_interval=merged_config.get("diff_interval", 4),
    )

    trainer = TRAINERS[method]
    return trainer(merged_config, dataloaders, device)
