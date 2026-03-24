import torch.nn as nn
from torchvision import models


class SkinCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # MobileNetV3 Small — fast + accurate for small datasets
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        # Unfreeze all layers — small dataset needs full fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Replace final classifier layer only
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)