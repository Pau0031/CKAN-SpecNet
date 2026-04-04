# CKAN-SpecNet

**Intelligent Decoding of Functional Group Multiplicity and Substructural Isomerism from Infrared Spectroscopy via Deep Neural Networks**

A deep learning architecture combining Kolmogorov-Arnold Networks (KAN) with CNNs for functional group enumeration and isomer identification from mid-infrared spectra.

## Overview

CKAN-SpecNet integrates CNN feature extraction with KAN decision-making to tackle challenging MIR analysis tasks including functional group multiplicity and substructural isomerism. The framework leverages hierarchical spectral features and interpretable activation functions to achieve superior classification performance on organic compound spectra.

## Quick Start

### Requirements

- Python >= 3.12
- PyTorch >= 2.2

### Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas polars scikit-learn matplotlib seaborn
```

### Running

```bash
python scripts/train.py --config configs/default.yaml
```

Or use Jupyter notebooks:

```bash
jupyter-lab
```

## Project Structure

```
CKAN-SpecNet/
├── ckan_specnet/
│   ├── models.py          # Model architectures
│   ├── dataset.py         # Data loading
│   ├── losses.py          # Loss functions
│   └── utils.py           # Helper functions
├── configs/
│   └── default.yaml       # Configuration
├── scripts/
│   ├── train.py           # Training script
│   └── predict.py         # Inference script
├── notebooks/
│   └── train_demo.ipynb   # Demo notebook
├── data/                  # Dataset files
```