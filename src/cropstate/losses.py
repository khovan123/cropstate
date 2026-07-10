"""Training losses for the vision baselines.

- CrossEntropy (optionally class-weighted) is the paper's default.
- FocalLoss down-weights easy examples to help severely under-represented stages
  (Tier C#6; the paper notes focal loss is a standard mitigation "not applied here").
- OrdinalExpectationLoss adds an expectation-regression penalty that explicitly
  punishes predictions far from the true stage index, exploiting the ordinal
  structure the paper only measures via MASD (Tier A#2). It keeps the ordinary
  six-way softmax head so existing checkpoints/pipeline stay compatible.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class OrdinalExpectationLoss(nn.Module):
    """CE + lambda * (E[stage] - true_stage)^2, with E[stage] = sum_i i * softmax_i.

    The squared term penalizes non-adjacent stage errors far more than adjacent
    ones, directly targeting MASD and cross-stage confusion.
    """

    def __init__(self, num_classes: int = 6, lam: float = 0.5, weight: torch.Tensor | None = None):
        super().__init__()
        self.lam = lam
        self.weight = weight
        self.register_buffer("indices", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight)
        probs = F.softmax(logits, dim=1)
        expected = (probs * self.indices.to(probs.device)).sum(dim=1)
        ordinal = ((expected - target.float()) ** 2).mean()
        return ce + self.lam * ordinal


def build_loss(name: str, class_weights: torch.Tensor | None = None,
               num_classes: int = 6, focal_gamma: float = 2.0, ordinal_lambda: float = 0.5) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    if name == "ordinal":
        return OrdinalExpectationLoss(num_classes=num_classes, lam=ordinal_lambda, weight=class_weights)
    raise ValueError(f"Unknown loss: {name!r} (expected ce|focal|ordinal)")
