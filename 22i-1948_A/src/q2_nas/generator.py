"""Deliverable 5 - automated architecture generator: arch-config dict -> nn.Module."""
import torch
import torch.nn as nn
import torch.nn.functional as F

ACT = {"relu": nn.ReLU, "relu6": nn.ReLU6, "hardswish": nn.Hardswish, "silu": nn.SiLU, "gelu": nn.GELU}


def make_act(name):
    cls = ACT[name]
    return cls() if name == "gelu" else cls(inplace=True)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k, s=1, groups=1, act="relu"):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
        if act:
            layers.append(make_act(act))
        super().__init__(*layers)


class StandardBlock(nn.Module):
    """Plain kxk convolution block (residual when shapes match)."""

    def __init__(self, cin, cout, k, s, act, **_):
        super().__init__()
        self.conv = ConvBNAct(cin, cout, k, s, act=act)
        self.res = s == 1 and cin == cout

    def forward(self, x):
        y = self.conv(x)
        return x + y if self.res else y


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise kxk conv (groups=cin) + pointwise 1x1 conv (MobileNetV1-style)."""

    def __init__(self, cin, cout, k, s, act, **_):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, k, s, groups=cin, act=act)
        self.pw = ConvBNAct(cin, cout, 1, act=act)
        self.res = s == 1 and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return x + y if self.res else y


class MBConvBlock(nn.Module):
    """Inverted residual: 1x1 expand -> depthwise kxk -> 1x1 linear projection."""

    def __init__(self, cin, cout, k, s, act, expansion=4, **_):
        super().__init__()
        hid = cin * expansion
        self.body = nn.Sequential(ConvBNAct(cin, hid, 1, act=act), ConvBNAct(hid, hid, k, s, groups=hid, act=act),
                                  ConvBNAct(hid, cout, 1, act=None))
        self.res = s == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


BLOCKS = {"conv": StandardBlock, "dws": DepthwiseSeparableBlock, "mbconv": MBConvBlock}


class GeneratedNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()
        self.cfg = cfg
        first_act = cfg["stages"][0]["activation"]
        self.stem = ConvBNAct(3, cfg["stem_channels"], cfg["stem_kernel"], 1, act=first_act)
        cin, stages = cfg["stem_channels"], []
        for st in cfg["stages"]:
            blocks = []
            for i in range(st["depth"]):
                blocks.append(BLOCKS[st["block"]](cin, st["channels"], st["kernel"], st["stride"] if i == 0 else 1,
                                                  st["activation"], expansion=st["expansion"]))
                cin = st["channels"]
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        last_act = cfg["stages"][-1]["activation"]
        self.head = ConvBNAct(cin, cfg["head_channels"], 1, act=last_act) if cfg["head_channels"] else nn.Identity()
        feat = cfg["head_channels"] or cin
        self.classifier = nn.Sequential(nn.Dropout(cfg.get("dropout", 0.0)), nn.Linear(feat, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(self.stages(self.stem(x)))
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def build_model(cfg, num_classes=10):
    return GeneratedNet(cfg, num_classes)
