"""Model implementation for A02 fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class FineTuner(nn.Module):
    """ResNet fine-tuning model for EuroSAT.

    Input shape:
        x: (B, 3, H, W)
    Output shape:
        logits: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10, freeze: bool = True) -> None:
        """Initialize backbone and classifier head.

        Configures freezing behavior and replaces the final classification head.
        """
        super().__init__()
        # Load the pretrained ResNet-18 model
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 1. Freeze backbone parameters if freeze=True
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2. Replace the classification head (model.fc) with the target number of classes
        # This new linear layer will have requires_grad=True by default.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model backbone."""
        return self.backbone(x)