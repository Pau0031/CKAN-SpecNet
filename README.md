# CKAN-SpecNet

**KAN-Enhanced CNN Framework for Multi-Task MIR Spectral Analysis**

A deep learning architecture combining Kolmogorov-Arnold Networks with CNNs for functional group enumeration and isomer identification from mid-infrared spectra.

## Overview

CKAN-SpecNet integrates CNN feature extraction with KAN decision-making to tackle challenging MIR analysis tasks including functional group multiplicity and substructural isomerism. The framework leverages hierarchical spectral features and interpretable activation functions to achieve superior classification performance on organic compound spectra.

## Quick Start

Install required dependencies: `python >=3.12` && `pytorch>=2.2`

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn matplotlib seaborn kan-torch grad-cam
```

## Running the Framework

1. **Launch Jupyter Lab:**
```bash
jupyter-lab
```

2. **Execute notebooks in sequence:**
```bash
*.ipynb
```