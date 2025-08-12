import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network Layer"""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Initialize B-spline parameters
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size))
        self.coef = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.1)
        
        # Learnable scaling and bias
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def b_splines(self, x):
        """Compute B-spline basis functions"""
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)  # [batch, in_features, 1]
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # [1, 1, grid_size]
        
        # Simple cubic B-spline approximation
        dist = torch.abs(x - grid)
        basis = torch.where(dist < 1, 1 - dist**3, torch.zeros_like(dist))
        basis = basis / (basis.sum(dim=-1, keepdim=True) + 1e-8)
        
        return basis
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Normalize input to [-1, 1]
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1
        
        # Compute B-spline basis
        basis = self.b_splines(x_normalized)  # [batch, in_features, grid_size]
        
        # Apply learnable coefficients
        # Reshape for matrix multiplication
        basis = basis.unsqueeze(1)  # [batch, 1, in_features, grid_size]
        coef = self.coef.unsqueeze(0)  # [1, out_features, in_features, grid_size]
        
        # Compute weighted sum
        output = (basis * coef).sum(dim=-1)  # [batch, out_features, in_features]
        
        # Apply scaling
        output = output * self.scale.unsqueeze(0)
        
        # Sum over input features and add bias
        output = output.sum(dim=-1) + self.bias
        
        return output


class KANBlock(nn.Module):
    """KAN block with residual connection and normalization"""
    def __init__(self, in_features, out_features, grid_size=5):
        super(KANBlock, self).__init__()
        self.kan = KANLayer(in_features, out_features, grid_size)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        
        # Residual connection if dimensions match
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        out = self.kan(x)
        out = self.norm(out)
        out = self.activation(out + residual)
        return out


class ECA(nn.Module):
    """Efficient Channel Attention module"""
    def __init__(self, in_channel, b=1, gama=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv = nn.Conv1d(in_channels=in_channel,
                              out_channels=1,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
    
    def forward(self, x):
        x_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x_pool = x_pool.unsqueeze(-1)
        x_pool = self.conv(x_pool)
        x_pool = torch.sigmoid(x_pool)
        x_pool = x_pool.squeeze(-1)
        x_pool = x_pool.unsqueeze(-1)
        return x * x_pool


class CKAN_SpecNet(nn.Module):
    """CKAN-SpecNet: CNN feature extractor with KAN classifier"""
    def __init__(self, num_classes=4, use_kan=True, grid_size=5):
        super().__init__()
        self.use_kan = use_kan
        
        # CNN Feature Extractor (unchanged)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3),
            
            nn.Conv1d(32, 64, kernel_size=13, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=11, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ECA(128),
            
            nn.Conv1d(128, 256, kernel_size=7, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ECA(256),
            
            nn.AdaptiveAvgPool1d(64),
        )
        
        flattened_size = 256 * 64
        
        if self.use_kan:
            # KAN-based classifier
            self.classifier = nn.Sequential(
                nn.Linear(flattened_size, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(0.6),
                
                # Replace second linear layer with KAN
                KANBlock(768, 256, grid_size=grid_size),
                nn.Dropout(0.2),
                
                # Final KAN layer for classification
                KANLayer(256, num_classes, grid_size=grid_size)
            )
        else:
            # Original MLP classifier (for comparison)
            self.classifier = nn.Sequential(
                nn.Linear(flattened_size, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(0.6),
                
                nn.Linear(768, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Extract features
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Extract CNN features for visualization"""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        return x


# Example usage and comparison
if __name__ == "__main__":
    # Create sample data
    batch_size = 16
    sequence_length = 3600
    num_classes = 4
    
    # Generate random input
    x = torch.randn(batch_size, sequence_length)
    
    # Initialize models
    model_kan = CKAN_SpecNet(num_classes=num_classes, use_kan=True)
    model_mlp = CKAN_SpecNet(num_classes=num_classes, use_kan=False)
    
    # Forward pass
    output_kan = model_kan(x)
    output_mlp = model_mlp(x)
    
    print(f"Input shape: {x.shape}")
    print(f"KAN output shape: {output_kan.shape}")
    print(f"MLP output shape: {output_mlp.shape}")
    
    # Count parameters
    kan_params = sum(p.numel() for p in model_kan.parameters() if p.requires_grad)
    mlp_params = sum(p.numel() for p in model_mlp.parameters() if p.requires_grad)
    
    print(f"\nKAN model parameters: {kan_params:,}")
    print(f"MLP model parameters: {mlp_params:,}")
    print(f"Parameter reduction: {(1 - kan_params/mlp_params)*100:.1f}%")