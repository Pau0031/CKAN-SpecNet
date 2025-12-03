import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score


class MixupAugmentation:
    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, x, y, num_classes=2):
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        if torch.rand(1).item() > self.p:
            return x, y_onehot, False
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]
        return mixed_x, mixed_y, True


def mixup_criterion(pred, y_mixed):
    return -torch.mean(torch.sum(y_mixed * F.log_softmax(pred, dim=1), dim=1))


def focal_loss(pred, target, weight=None, gamma=2.0):
    ce = F.cross_entropy(pred, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def get_smoothness_loss(model, neighbor_range=1):
    smooth_loss = 0
    count = 0

    for module in model.modules():
        if hasattr(module, "spline_weight"):
            weights = module.spline_weight
            for shift in range(1, neighbor_range + 1):
                if weights.dim() == 3 and weights.shape[1] > 2 * shift:
                    left = weights[:, :-2 * shift, :]
                    center = weights[:, shift:-shift, :]
                    right = weights[:, 2 * shift:, :]
                    laplacian = left - 2 * center + right
                    smooth_loss += torch.mean(laplacian ** 2)
                    count += 1
                elif weights.dim() == 2 and weights.shape[0] > 2 * shift:
                    left = weights[:-2 * shift, :]
                    center = weights[shift:-shift, :]
                    right = weights[2 * shift:, :]
                    laplacian = left - 2 * center + right
                    smooth_loss += torch.mean(laplacian ** 2)
                    count += 1

    return smooth_loss / max(count, 1) if count > 0 else torch.tensor(0.0)


def train_epoch(model, train_loader, optimizer, criterion, device,
                use_mixup=False, mixup_alpha=1.0, mixup_p=0.5, num_classes=2,
                use_smoothness=False, lambda_smooth=0.01):
    model.train()
    total_loss = 0
    augment = MixupAugmentation(mixup_alpha, mixup_p) if use_mixup else None

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        if use_mixup:
            data, target_mixed, is_mixed = augment(data, target, num_classes)
        else:
            is_mixed = False

        optimizer.zero_grad()
        output = model(data)

        if is_mixed:
            loss = mixup_criterion(output, target_mixed)
        else:
            loss = criterion(output, target)

        if use_smoothness:
            smooth_loss = get_smoothness_loss(model)
            loss = loss + lambda_smooth * smooth_loss.to(device)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), 100.0 * correct / total


def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(target.numpy())

    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)


def compute_metrics(y_true, y_pred, num_classes=2):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=list(range(num_classes))
    )
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if num_classes > 2:
        metrics["kappa"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return metrics


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device,
        num_classes=2,
        learning_rate=0.001,
        weight_decay=0.05,
        class_weights=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.best_acc = 0
        self.best_state = None

    def train(
        self,
        epochs=500,
        patience=50,
        use_mixup=False,
        mixup_alpha=1.0,
        mixup_p=0.5,
        use_smoothness=False,
        lambda_smooth=0.01,
        verbose=True,
    ):
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion,
                self.device, use_mixup, mixup_alpha, mixup_p, self.num_classes,
                use_smoothness, lambda_smooth
            )

            test_loss, test_acc = evaluate(
                self.model, self.test_loader, self.criterion, self.device
            )

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                      f"Test Loss={test_loss:.4f}, Acc={test_acc:.2f}%")

            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self.best_acc

    def evaluate(self):
        y_true, y_pred, y_probs = get_predictions(self.model, self.test_loader, self.device)
        metrics = compute_metrics(y_true, y_pred, self.num_classes)
        return metrics, y_true, y_pred, y_probs

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
