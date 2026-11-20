"""ResNet-50 teacher adapted to 32x32 CIFAR images.

The ImageNet stem (7x7 stride-2 conv + max-pool) would shrink a 32x32 image to
8x8 before the first residual stage, so it is replaced by a 3x3 stride-1 conv and
the max-pool is removed (standard CIFAR adaptation). Spatial sizes:
32 (layer1) -> 16 (layer2) -> 8 (layer3) -> 4 (layer4).
Optionally initialised from ImageNet weights (the new 3x3 stem is initialised from
the centre of the pretrained 7x7 kernels), then fully trained on CIFAR-100.
"""
import torch
import torch.nn as nn
import torchvision


class ResNet50Teacher(nn.Module):
    # channels of the intermediate representations used for feature distillation
    feature_channels = {"mid": 1024, "final": 2048}

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None
        try:
            net = torchvision.models.resnet50(weights=weights)
        except Exception as ex:  # no internet -> random init
            print(f"[teacher] could not load ImageNet weights ({ex}); training from scratch")
            net = torchvision.models.resnet50(weights=None)
            weights = None
        old_conv = net.conv1
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        if weights is not None:
            with torch.no_grad():
                net.conv1.weight.copy_(old_conv.weight[:, :, 2:5, 2:5] * (49.0 / 9.0) ** 0.5)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(2048, num_classes)
        self.pretrained = weights is not None
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = net.fc

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.layer2(self.layer1(x))
        mid = self.layer3(x)          # 1024 x 8 x 8
        final = self.layer4(mid)      # 2048 x 4 x 4
        logits = self.fc(torch.flatten(self.pool(final), 1))
        if return_features:
            return logits, {"mid": mid, "final": final}
        return logits


def build_teacher(num_classes=100, pretrained=True):
    return ResNet50Teacher(num_classes, pretrained)
