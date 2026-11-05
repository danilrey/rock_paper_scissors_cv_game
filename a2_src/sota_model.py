import torch
import torch.nn as nn
from torchvision import models


class ConvNeXtSOTAModel(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)

        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.features[-1]


def build_sota_model(num_classes: int = 3, pretrained: bool = True) -> ConvNeXtSOTAModel:
    return ConvNeXtSOTAModel(num_classes=num_classes, pretrained=pretrained)
