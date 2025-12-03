#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ckan_specnet import CKANSpecNet, get_device


def load_spectrum(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "spectrum" in df.columns:
            return df["spectrum"].values.astype(np.float32)
        return df.iloc[:, 0].values.astype(np.float32)
    elif path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {path}")


def main():
    parser = argparse.ArgumentParser(description="Predict with CKAN-SpecNet")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input spectrum file")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    spectrum = load_spectrum(args.input)
    print(f"Loaded spectrum: shape={spectrum.shape}")

    model = CKANSpecNet(
        input_size=len(spectrum),
        num_classes=args.num_classes,
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    x = torch.tensor(spectrum).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred = output.argmax(dim=1).item()

    print(f"\nPrediction: Class {pred}")
    print(f"Probabilities:")
    for i, p in enumerate(probs):
        print(f"  Class {i}: {p:.4f}")

    if args.output:
        result = {
            "predicted_class": pred,
            **{f"prob_class_{i}": probs[i] for i in range(len(probs))}
        }
        pd.DataFrame([result]).to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
