import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight


SMARTS_PATTERNS = {
    "alkane": "[CX4]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#C",
    "aromatics": "[$([cX3](:*):*),$([cX2+](:*):*)]",
    "alkyl_halides": "[#6][F,Cl,Br,I]",
    "alcohols": "[#6][OX2H]",
    "esters": "[#6][CX3](=O)[OX2H0][#6]",
    "ketones": "[#6][CX3](=O)[#6]",
    "aldehydes": "[CX3H1](=O)[#6]",
    "carbonyl_oxygen": "[CX3](=O)[OX2]",
    "ether": "[OX2;!$(OC=O)]([#6])[#6]",
    "acyl_halides": "[CX3](=[OX1])[F,Cl,Br,I]",
    "amines": "[NX3;!$(NC=O)]",
    "amides": "[NX3][CX3](=[OX1])[#6]",
    "nitriles": "[NX1]#[CX2]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "isocyanate": "[#7]=[C]=[O]",
    "isothiocyanate": "[#7]=[C]=[S]",
    "ortho": "*-!:aa-!:*",
    "meta": "*-!:aaa-!:*",
    "para": "*-!:aaaa-!:*",
}


class SpectrumDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MultiTaskDataset(Dataset):
    def __init__(self, X, Y_dict):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = {task: torch.tensor(y, dtype=torch.long) for task, y in Y_dict.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {task: y[idx] for task, y in self.Y.items()}


def prepare_labels(df, tasks, upper_bounds, is_polars=True):
    labels_dict = {}
    for task, upper in zip(tasks, upper_bounds):
        if is_polars:
            labels_dict[task] = df[task].clip(upper_bound=upper).to_numpy().astype(np.int64)
        else:
            labels_dict[task] = df[task].clip(upper=upper).to_numpy().astype(np.int64)
    return labels_dict


def compute_task_weights(Y_dict, task_num_classes):
    weights_dict = {}
    for task, labels in Y_dict.items():
        unique = np.unique(labels)
        n_classes = task_num_classes[task]
        if len(unique) < n_classes:
            weights = np.ones(n_classes)
            present_weights = compute_class_weight("balanced", classes=unique, y=labels)
            for i, c in enumerate(unique):
                weights[c] = present_weights[i]
        else:
            weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=labels)
        weights_dict[task] = weights
    return weights_dict


def stratified_split(X, Y, n_classes, test_size=0.2, random_state=42):
    X_np = X if isinstance(X, np.ndarray) else np.array(X)
    Y_np = Y if isinstance(Y, np.ndarray) else np.array(Y)
    Y_np = Y_np.astype(np.int64)

    class_weights = compute_class_weight("balanced", classes=np.unique(Y_np), y=Y_np)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_np, Y_np, test_size=test_size, stratify=Y_np, random_state=random_state
    )

    return X_train, X_test, Y_train, Y_test, class_weights


def create_dataloaders(X_train, Y_train, X_test, Y_test, batch_size=1024, num_workers=0):
    train_dataset = SpectrumDataset(X_train, Y_train)
    test_dataset = SpectrumDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def create_kfold_loaders(X, Y, n_splits=5, batch_size=1024, random_state=42):
    dataset = SpectrumDataset(X, Y)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kfold.split(X):
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=batch_size, shuffle=False
        )
        yield train_loader, val_loader
