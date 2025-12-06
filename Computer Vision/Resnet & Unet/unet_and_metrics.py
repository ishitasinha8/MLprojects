import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -------------------------
# UNetwithResnetFeatures
# -------------------------
class UNetwithResnetFeatures(nn.Module):
    def __init__(self, numclasses, layerid):
        super().__init__()
        # TODO: put your real UNet code here
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Encoder layers
        self.layer0 = nn.Sequential(
            resnet.conv1,   # /2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # /4
        )
        self.layer1 = resnet.layer1        # /4
        self.layer2 = resnet.layer2        # /8
        self.layer3 = resnet.layer3        # /16
        self.layer4 = resnet.layer4        # /32

        # Decoder Blocks

        # layer4 → layer3 scale (channels: 512 → 256)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.reduce3 = nn.Conv2d(256, 256, kernel_size=1)

        # layer3 → layer2 scale (256 → 128)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.reduce2 = nn.Conv2d(128, 128, kernel_size=1)

        # layer2 → layer1 scale (128 → 64)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.reduce1 = nn.Conv2d(64, 64, kernel_size=1)

        # Final classifier
        self.classifier = nn.Conv2d(64, numclasses, kernel_size=1)


    def forward(self, x):
        # TODO: put your real UNet forward code here
        B, C, H, W = x.shape

        # Encoder forward
        x0 = self.layer0(x)   # /4
        x1 = self.layer1(x0)  # /4   (64 channels)
        x2 = self.layer2(x1)  # /8   (128)
        x3 = self.layer3(x2)  # /16  (256)
        x4 = self.layer4(x3)  # /32  (512)

        # Decoder

        # layer4 → layer3 scale
        d4 = self.up4(x4)     # (B,256,H/16,W/16)
        d4 = d4 + x3          # skip connection
        d4 = self.reduce3(d4)

        # → layer2 scale
        d3 = self.up3(d4)     # (B,128,H/8,W/8)
        d3 = d3 + x2
        d3 = self.reduce2(d3)

        # → layer1 scale
        d2 = self.up2(d3)     # (B,64,H/4,W/4)
        d2 = d2 + x1
        d2 = self.reduce1(d2)

        # Final prediction
        logits = self.classifier(d2)

        # Upsample to original input size
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )

        return logits

# -------------------------
# compute_perclass_accuracy
# -------------------------
def compute_perclass_accuracy(predictions, labels):

    num_classes = max(predictions.max(), labels.max()) + 1
    per_class_acc = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            per_class_acc[c] = np.nan
        else:
            correct = (predictions[mask] == labels[mask]).sum()
            per_class_acc[c] = correct / mask.sum()

    return per_class_acc
