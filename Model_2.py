
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FruitFreshnessModel(nn.Module):
    def __init__(
        self,
        num_produce_classes,
        pretrained=True,
        freeze_backbone=False
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.shared_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.3)
        )

        # self.produce_classifier = nn.Linear(
        #     256, num_produce_classes
        # )
        self.produce_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_produce_classes),

        )

        self.freshness_regressor = nn.Sequential(
            nn.Linear(256, 125),
            nn.SiLU(),
            nn.Linear(125, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.shared_features(x)

        produce_logits = self.produce_classifier(x)
        freshness_score = self.freshness_regressor(x).squeeze(1)

        return produce_logits, freshness_score
