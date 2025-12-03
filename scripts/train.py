#!/usr/bin/env python
import argparse
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ckan_specnet import (
    CKANSpecNet,
    stratified_split,
    create_dataloaders,
    Trainer,
    set_seed,
    get_device,
    count_parameters,
    save_results,
    SMARTS_PATTERNS,
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path, target, upper_bound):
    df = pl.read_parquet(data_path)
    if "components" in df.columns:
        df = df.filter(pl.col("components") == 1)
    
    if "X" in df.columns:
        X = np.vstack(df["X"].list.slice(1).to_list()).astype(np.float32)
        X = X[:, 76:-78]
    elif "x" in df.columns:
        X = np.vstack(df["x"].to_list()).astype(np.float32)
    else:
        raise ValueError("Cannot find spectrum column")
    
    Y = df[target].clip(upper_bound=upper_bound).to_numpy().astype(np.int64)
    return X, Y


def main():
    parser = argparse.ArgumentParser(description="Train CKAN-SpecNet")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default="data/data.parquet")
    parser.add_argument("--target", type=str, default="alcohols")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    upper_bound = args.num_classes - 1

    print(f"Loading data: {args.data}")
    print(f"Target: {args.target}, Classes: {args.num_classes}")
    X, Y = load_data(args.data, args.target, upper_bound)
    print(f"Dataset size: {len(X)}")

    X_train, X_test, Y_train, Y_test, class_weights = stratified_split(
        X, Y, args.num_classes,
        test_size=config["data"]["test_size"],
        random_state=args.seed
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Class weights: {class_weights.round(2)}")

    train_loader, test_loader = create_dataloaders(
        X_train, Y_train, X_test, Y_test, batch_size
    )

    model = CKANSpecNet(
        input_size=X_train.shape[1],
        num_classes=args.num_classes,
        conv_channels=config["model"]["conv_channels"],
        conv_kernels=config["model"]["conv_kernels"],
        pool_sizes=config["model"]["pool_sizes"],
        eca_positions=config["model"]["eca_positions"],
        adaptive_pool_size=config["model"]["adaptive_pool_size"],
        fc_hidden=config["model"]["fc_hidden"],
        kan_hidden=config["model"]["kan_hidden"],
        dropout_fc=config["model"]["dropout_fc"],
        dropout_head=config["model"]["dropout_head"],
    )
    print(f"Model parameters: {count_parameters(model):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_classes=args.num_classes,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        class_weights=class_weights,
    )

    print(f"\nTraining for {epochs} epochs...")
    best_acc = trainer.train(
        epochs=epochs,
        patience=config["training"]["patience"],
        use_mixup=config["training"]["use_mixup"],
        mixup_alpha=config["training"]["mixup_alpha"],
        mixup_p=config["training"]["mixup_p"],
        use_smoothness=config["training"]["use_smoothness"],
        lambda_smooth=config["training"]["lambda_smooth"],
    )
    print(f"Best accuracy: {best_acc:.2f}%")

    metrics, y_true, y_pred, y_probs = trainer.evaluate()
    print(f"\nFinal Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    save_dir = Path(args.output) / f"{args.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_results(save_dir, metrics, (y_true, y_pred, y_probs), vars(args))
    trainer.save(save_dir / "model.pth")
    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
