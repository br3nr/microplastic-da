"""
Microplastic Domain Adaptation - Training CLI

Usage:
    python main.py --method coral
    python main.py --method dann --wandb
    python main.py --method dsan --epochs 20
"""

import argparse
import sys

from src.train import train, TRAINERS, METHOD_DEFAULTS
from src.utils import init_wandb, finish


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train domain adaptation models for microplastic classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=list(TRAINERS.keys()),
        default="coral",
        help="Domain adaptation method",
    )

    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="data",
        help="Path to data directory (must contain marine_polymers.csv and std_polymers.csv)",
    )

    # Training hyperparameters (None = use method default)
    parser.add_argument("--epochs", "-e", type=int, default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    # Model hyperparameters
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--heads", "-H", type=int, default=None)
    parser.add_argument("--layers", "-N", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--pe", action="store_true", help="Use positional encoding")
    parser.add_argument("--bottleneck-dim", type=int, default=None)

    # Preprocessing
    parser.add_argument("--diff-order", type=int, default=None)
    parser.add_argument("--diff-interval", type=int, default=None)

    # W&B logging
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="microplastic-da")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config from args (None values filled by method defaults)
    config = {}

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.d_model is not None:
        config["d_model"] = args.d_model
    if args.heads is not None:
        config["h"] = args.heads
    if args.layers is not None:
        config["N"] = args.layers
    if args.dropout is not None:
        config["dropout"] = args.dropout
    if args.pe:
        config["pe"] = True
    if args.bottleneck_dim is not None:
        config["bottleneck_dim"] = args.bottleneck_dim
    if args.diff_order is not None:
        config["diff_order"] = args.diff_order
    if args.diff_interval is not None:
        config["diff_interval"] = args.diff_interval

    method_defaults = METHOD_DEFAULTS.get(args.method, {})
    effective_config = {**method_defaults, **config}

    run_name = args.run_name or f"{args.method}-{effective_config['epochs']}ep"
    init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=effective_config,
    )

    print(f"\n{'=' * 60}")
    print(f"Microplastic Domain Adaptation Training")
    print(f"{'=' * 60}")
    print(f"Method: {args.method.upper()}")
    print(f"Data: {args.data_dir}")
    print(
        f"Epochs: {effective_config['epochs']}, Batch: {effective_config['batch_size']}, LR: {effective_config['lr']}"
    )
    print(
        f"Model: d_model={effective_config['d_model']}, heads={effective_config['h']}, layers={effective_config['N']}, bottleneck={effective_config['bottleneck_dim']}"
    )
    print(f"W&B: {'enabled' if args.wandb else 'disabled'}")
    print(f"{'=' * 60}\n")

    try:
        model = train(method=args.method, config=config, data_dir=args.data_dir)
        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        finish()

    return model


if __name__ == "__main__":
    main()
