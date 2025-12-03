import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KAN, KANLinear


class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).squeeze(-1).unsqueeze(1)
        y = torch.sigmoid(self.conv(y)).squeeze(1).unsqueeze(-1)
        return x * y


class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1.0, weight=None, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, labels):
        if labels.ndim == 1:
            labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(
                device=logits.device, dtype=logits.dtype
            )
        else:
            labels_onehot = labels.to(device=logits.device, dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        poly1 = ce + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            return poly1.mean()
        elif self.reduction == "sum":
            return poly1.sum()
        return poly1


class CKANSpecNet(nn.Module):
    def __init__(
        self,
        input_size=1801,
        num_classes=2,
        conv_channels=(32, 64, 128, 256),
        conv_kernels=(15, 13, 11, 5),
        pool_sizes=(3, 2, None, None),
        eca_positions=(2, 3),
        adaptive_pool_size=64,
        fc_hidden=1024,
        kan_hidden=256,
        dropout_fc=0.7,
        dropout_head=0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        in_ch = 1
        for i, (out_ch, k) in enumerate(zip(conv_channels, conv_kernels)):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, k),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            if pool_sizes[i] is not None:
                layers.append(nn.AvgPool1d(pool_sizes[i], pool_sizes[i]))
            if i in eca_positions:
                layers.append(ECA(out_ch))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(adaptive_pool_size))
        self.backbone = nn.Sequential(*layers)

        fc_in = conv_channels[-1] * adaptive_pool_size
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
        )

        self.kan_layer = KAN([fc_hidden, kan_hidden])

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_head),
            KANLinear(kan_hidden, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.backbone(x).flatten(1)
        x = self.shared_fc(x)
        x = self.kan_layer(x)
        return self.head(x)

    def get_features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.backbone(x).flatten(1)
        x = self.shared_fc(x)
        return self.kan_layer(x)


class MultiTaskCKANSpecNet(nn.Module):
    def __init__(
        self,
        input_size=1801,
        task_num_classes=None,
        conv_channels=(32, 64, 128, 256),
        conv_kernels=(15, 13, 11, 5),
        pool_sizes=(3, 2, None, None),
        eca_positions=(2, 3),
        adaptive_pool_size=64,
        fc_hidden=1024,
        kan_hidden=256,
        dropout_fc=0.7,
        dropout_head=0.3,
    ):
        super().__init__()
        if task_num_classes is None:
            task_num_classes = {"default": 2}
        self.task_num_classes = task_num_classes

        layers = []
        in_ch = 1
        for i, (out_ch, k) in enumerate(zip(conv_channels, conv_kernels)):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, k),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            if pool_sizes[i] is not None:
                layers.append(nn.AvgPool1d(pool_sizes[i], pool_sizes[i]))
            if i in eca_positions:
                layers.append(ECA(out_ch))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(adaptive_pool_size))
        self.backbone = nn.Sequential(*layers)

        fc_in = conv_channels[-1] * adaptive_pool_size
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
        )

        self.kan_layer = KAN([fc_hidden, kan_hidden])

        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout_head),
                KANLinear(kan_hidden, n_classes),
            )
            for task, n_classes in task_num_classes.items()
        })

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.backbone(x).flatten(1)
        x = self.shared_fc(x)
        x = self.kan_layer(x)
        return {task: head(x) for task, head in self.heads.items()}


class CKANSpecNetMLP(nn.Module):
    def __init__(
        self,
        input_size=1801,
        num_classes=2,
        conv_channels=(32, 64, 128, 256),
        conv_kernels=(15, 13, 11, 5),
        pool_sizes=(3, 2, None, None),
        eca_positions=(2, 3),
        adaptive_pool_size=64,
        fc_hidden=1024,
        head_hidden=256,
        dropout_fc=0.7,
        dropout_head=0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        in_ch = 1
        for i, (out_ch, k) in enumerate(zip(conv_channels, conv_kernels)):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, k),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            if pool_sizes[i] is not None:
                layers.append(nn.AvgPool1d(pool_sizes[i], pool_sizes[i]))
            if i in eca_positions:
                layers.append(ECA(out_ch))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(adaptive_pool_size))
        self.backbone = nn.Sequential(*layers)

        fc_in = conv_channels[-1] * adaptive_pool_size
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
        )

        self.head = nn.Sequential(
            nn.Linear(fc_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.backbone(x).flatten(1)
        x = self.shared_fc(x)
        return self.head(x)
