"""Knowledge-distillation framework.

Total student objective (per mini-batch):

    L = (1 - alpha) * CE(z_s, y)                                   # hard labels
      + alpha * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )   # soft targets (Hinton et al., 2015)
      + beta  * sum_h  L_feat( r_h(F_s^h), F_t^h )                 # feature hints (FitNets, Romero et al., 2015)

* `temperature_softmax`  : softened class distribution softmax(z / T).
* `SoftTargetKDLoss`     : soft-target KD; the T^2 factor keeps gradient magnitudes
                           comparable across temperatures.
* `FeatureDistillation`  : learnable 1x1-conv + BN regressors r_h map each student
                           hint layer to the teacher's channel dimension; spatial sizes
                           are aligned with adaptive average pooling. Two hint losses:
                           - "fitnet": MSE between L2-normalised feature maps
                           - "at"    : attention transfer (Zagoruyko & Komodakis, 2017)
* `DistillationWrapper`  : bundles student + regressors + frozen teacher into one
                           module so the generic trainer can optimise it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def temperature_softmax(logits, T):
    return F.softmax(logits / T, dim=1)


class SoftTargetKDLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits, teacher_logits):
        T = self.T
        log_p_s = F.log_softmax(student_logits.float() / T, dim=1)
        p_t = temperature_softmax(teacher_logits.float(), T)
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


class FeatureDistillation(nn.Module):
    def __init__(self, student_channels: dict, teacher_channels: dict, hints=("mid", "final"), loss_type="fitnet"):
        super().__init__()
        self.hints, self.loss_type = list(hints), loss_type
        self.regressors = nn.ModuleDict({
            h: nn.Sequential(nn.Conv2d(student_channels[h], teacher_channels[h], 1, bias=False),
                             nn.BatchNorm2d(teacher_channels[h]))
            for h in self.hints})

    @staticmethod
    def _attention(f):
        return F.normalize(f.float().pow(2).mean(1).flatten(1), dim=1)

    def forward(self, s_feats, t_feats):
        losses = {}
        for h in self.hints:
            s, t = s_feats[h], t_feats[h].detach()
            if self.loss_type == "at":
                if s.shape[-2:] != t.shape[-2:]:
                    s = F.adaptive_avg_pool2d(s, t.shape[-2:])
                losses[h] = (self._attention(s) - self._attention(t)).pow(2).sum(1).mean()
            else:  # fitnet hint with regressor
                s = self.regressors[h](s)
                if s.shape[-2:] != t.shape[-2:]:
                    s = F.adaptive_avg_pool2d(s, t.shape[-2:])
                s_n = F.normalize(s.float().flatten(1), dim=1)
                t_n = F.normalize(t.float().flatten(1), dim=1)
                losses[h] = (s_n - t_n).pow(2).sum(1).mean()
        return sum(losses.values()), losses


class DistillationWrapper(nn.Module):
    """Student + (frozen) teacher + feature regressors; forward returns the KD loss."""

    def __init__(self, student, teacher, temperature=4.0, alpha=0.7, beta=1.0,
                 hints=("mid", "final"), feat_loss="fitnet"):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.alpha, self.beta = alpha, beta
        self.kd = SoftTargetKDLoss(temperature)
        self.use_feat = beta > 0 and len(hints) > 0
        self.feat = (FeatureDistillation(student.feature_channels, teacher.feature_channels, hints, feat_loss)
                     if self.use_feat else None)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # teacher BN statistics stay frozen
        return self

    def forward(self, x):
        return self.student(x)

    def loss(self, x, y):
        with torch.no_grad():
            t_logits, t_feats = self.teacher(x, return_features=True)
        s_logits, s_feats = self.student(x, return_features=True)
        ce = F.cross_entropy(s_logits.float(), y)
        kd = self.kd(s_logits, t_logits)
        total = (1.0 - self.alpha) * ce + self.alpha * kd
        parts = {"ce": ce.item(), "kd": kd.item()}
        if self.use_feat:
            feat, per_hint = self.feat(s_feats, t_feats)
            total = total + self.beta * feat
            parts["feat"] = feat.item()
        return total, s_logits, parts


def kd_criterion(wrapper: DistillationWrapper, x, y):
    return wrapper.loss(x, y)
