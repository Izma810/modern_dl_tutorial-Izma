from __future__ import annotations

import torch
import torch.nn as nn

class MyCNN(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Flow: conv -> relu -> pool, repeated, linear head.
        self.num_classes = num_classes

        # Feature Extractor
        self.features=nn.Sequential(

            # Layer1->
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer2->
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=1, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        # Classifier Head
        self.classifier=nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=2048, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x=self.features(x)
        logits=self.classifier(x)
        return logits
      
        raise NotImplementedError(
            "Implement MyCNN.forward with the planned conv/classifier path."
        )
