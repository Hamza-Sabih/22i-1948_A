"""Lightweight student networks, implemented from scratch and adapted to 32x32 inputs.

Every student exposes the same interface as the teacher:
    logits = model(x)
    logits, feats = model(x, return_features=True)   # feats = {"mid": 8x8 map, "final": 4x4 map}
and a `feature_channels` dict, which the feature-distillation module uses to build
1x1 regressors into the teacher's feature space.

CIFAR adaptation: the first strides are reduced so the total stride is 8
(32 -> 4), matching the teacher's layer3 (8x8) and layer4 (4x4) resolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=nn.ReLU6):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
        if act is not None:
            layers.append(act(inplace=True))
        super().__init__(*layers)


class DropPath(nn.Module):
    """Stochastic depth (used by EfficientNet-Lite)."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep)
        return x * mask / keep


class InvertedResidual(nn.Module):
    """MobileNetV2 / MBConv block: 1x1 expand -> kxk depthwise -> 1x1 linear projection."""

    def __init__(self, cin, cout, stride, expand, kernel=3, act=nn.ReLU6, drop_path=0.0):
        super().__init__()
        hidden = int(round(cin * expand))
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(cin, hidden, 1, act=act))
        layers += [ConvBNAct(hidden, hidden, kernel, stride, groups=hidden, act=act),
                   nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if self.use_res else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        return x + self.drop_path(out) if self.use_res else out


# --------------------------------------------------------------------------- MobileNetV2
class MobileNetV2(nn.Module):
    """MobileNetV2 (Sandler et al., 2018). CIFAR strides: stem 1, stages (1,1,2,2,1,2,1)."""

    # t (expansion), c (channels), n (repeats), s (stride)
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=100, width_mult=1.0, dropout=0.2):
        super().__init__()
        cin = _make_divisible(32 * width_mult)
        self.last_channels = _make_divisible(1280 * max(1.0, width_mult))
        self.stem = ConvBNAct(3, cin, 3, 1)
        stages = []
        for t, c, n, s in self.CFG:
            cout = _make_divisible(c * width_mult)
            blocks = []
            for i in range(n):
                blocks.append(InvertedResidual(cin, cout, s if i == 0 else 1, t))
                cin = cout
            stages.append(nn.Sequential(*blocks))
        self.stages_a = nn.Sequential(*stages[:5])   # -> 96 x 8 x 8  ("mid")
        self.stages_b = nn.Sequential(*stages[5:])   # -> 320 x 4 x 4
        self.head = ConvBNAct(cin, self.last_channels, 1)  # -> 1280 x 4 x 4 ("final")
        self.feature_channels = {"mid": _make_divisible(96 * width_mult), "final": self.last_channels}
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.last_channels, num_classes))
        _init_weights(self)

    def forward(self, x, return_features=False):
        mid = self.stages_a(self.stem(x))
        final = self.head(self.stages_b(mid))
        logits = self.classifier(torch.flatten(F.adaptive_avg_pool2d(final, 1), 1))
        return (logits, {"mid": mid, "final": final}) if return_features else logits


# --------------------------------------------------------------------------- SqueezeNet
class Fire(nn.Module):
    def __init__(self, cin, squeeze, e1, e3):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(cin, squeeze, 1, bias=False), nn.BatchNorm2d(squeeze), nn.ReLU(inplace=True))
        self.e1 = nn.Sequential(nn.Conv2d(squeeze, e1, 1, bias=False), nn.BatchNorm2d(e1), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(squeeze, e3, 3, padding=1, bias=False), nn.BatchNorm2d(e3), nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.e1(s), self.e3(s)], 1)


class SqueezeNet(nn.Module):
    """SqueezeNet v1.1 (Iandola et al., 2016) with BatchNorm, 3x3 stride-1 stem and
    max-pools placed at 32->16->8->4. Fire modules squeeze with 1x1 convs then expand
    with parallel 1x1 / 3x3 convs; the classifier is a 1x1 conv + global average pool."""

    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(nn.MaxPool2d(2), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64))           # 16x16
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128))     # 8x8 mid
        self.stage3 = nn.Sequential(nn.MaxPool2d(2), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
                                    Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))                       # 4x4 final
        self.feature_channels = {"mid": 256, "final": 512}
        # final ReLU of the original classifier is removed so logits can be negative (needed for KD)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1))
        _init_weights(self)
        nn.init.normal_(self.classifier[1].weight, 0, 0.01)

    def forward(self, x, return_features=False):
        mid = self.stage2(self.stage1(self.stem(x)))
        final = self.stage3(mid)
        logits = torch.flatten(F.adaptive_avg_pool2d(self.classifier(final), 1), 1)
        return (logits, {"mid": mid, "final": final}) if return_features else logits


# --------------------------------------------------------------------------- EfficientNet-Lite0
class EfficientNetLite0(nn.Module):
    """EfficientNet-Lite0 (Tan & Le, 2019; TF Lite variant): EfficientNet-B0 MBConv
    layout with the edge-friendly changes — no squeeze-and-excitation, ReLU6
    instead of Swish, and stem/head widths fixed (not scaled)."""

    # expansion, kernel, stride, out_channels, repeats   (B0 layout, CIFAR strides)
    CFG = [(1, 3, 1, 16, 1), (6, 3, 1, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3),
           (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]

    def __init__(self, num_classes=100, dropout=0.2, drop_path=0.1):
        super().__init__()
        self.stem = ConvBNAct(3, 32, 3, 1)
        total = sum(c[4] for c in self.CFG)
        cin, idx, stages = 32, 0, []
        for e, k, s, c, n in self.CFG:
            blocks = []
            for i in range(n):
                blocks.append(InvertedResidual(cin, c, s if i == 0 else 1, e, k, drop_path=drop_path * idx / total))
                cin, idx = c, idx + 1
            stages.append(nn.Sequential(*blocks))
        self.stages_a = nn.Sequential(*stages[:5])   # -> 112 x 8 x 8 ("mid")
        self.stages_b = nn.Sequential(*stages[5:])   # -> 320 x 4 x 4
        self.head = ConvBNAct(320, 1280, 1)          # -> 1280 x 4 x 4 ("final")
        self.feature_channels = {"mid": 112, "final": 1280}
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        _init_weights(self)

    def forward(self, x, return_features=False):
        mid = self.stages_a(self.stem(x))
        final = self.head(self.stages_b(mid))
        logits = self.classifier(torch.flatten(F.adaptive_avg_pool2d(final, 1), 1))
        return (logits, {"mid": mid, "final": final}) if return_features else logits


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


STUDENTS = {
    "mobilenetv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efficientnet_lite0": EfficientNetLite0,
}

PRETTY = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet",
          "efficientnet_lite0": "EfficientNet-Lite0", "resnet50": "ResNet-50 (teacher)"}


def build_student(name, num_classes=100):
    if name not in STUDENTS:
        raise ValueError(f"unknown student '{name}', choose from {list(STUDENTS)}")
    return STUDENTS[name](num_classes=num_classes)
