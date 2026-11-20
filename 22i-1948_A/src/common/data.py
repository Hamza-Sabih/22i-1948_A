"""CIFAR-10 / CIFAR-100 loading and a fast on-device batch loader.

The whole dataset is held as uint8 on the training device; random-crop (pad 4),
horizontal flip, normalisation and optional resizing are done batch-wise on the
GPU. This removes the CPU data-loading bottleneck on Colab (2 vCPUs) and makes
both KD and NAS training several times faster than a torchvision DataLoader.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .utils import DATA_DIR

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def load_cifar(name="cifar100", root=None, smoke=False):
    """Return dict with uint8 NHWC arrays x_train/x_test, int64 y_train/y_test, classes."""
    name = name.lower()
    num_classes = 100 if name == "cifar100" else 10
    if smoke:  # synthetic data for a fast end-to-end functional test
        rng = np.random.default_rng(0)
        classes = CIFAR10_CLASSES if num_classes == 10 else [f"class_{i}" for i in range(100)]
        return dict(x_train=rng.integers(0, 256, (640, 32, 32, 3), dtype=np.uint8),
                    y_train=rng.integers(0, num_classes, 640).astype(np.int64),
                    x_test=rng.integers(0, 256, (256, 32, 32, 3), dtype=np.uint8),
                    y_test=rng.integers(0, num_classes, 256).astype(np.int64),
                    classes=classes, num_classes=num_classes, name=name)
    import torchvision
    root = str(root or DATA_DIR)
    cls = torchvision.datasets.CIFAR100 if name == "cifar100" else torchvision.datasets.CIFAR10
    tr = cls(root=root, train=True, download=True)
    te = cls(root=root, train=False, download=True)
    return dict(x_train=tr.data, y_train=np.array(tr.targets, dtype=np.int64),
                x_test=te.data, y_test=np.array(te.targets, dtype=np.int64),
                classes=list(tr.classes), num_classes=num_classes, name=name)


def train_val_split(x, y, n_val, seed=0):
    """Stratified hold-out split (used by NAS so that search never sees the test set)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, n_val // len(classes))
    val_idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:per_class] for c in classes])
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    return x[mask], y[mask], x[val_idx], y[val_idx]


class DeviceLoader:
    """Iterable over (images, labels) batches produced entirely on `device`."""

    def __init__(self, x, y, batch_size, device, dataset="cifar100", train=True,
                 resolution=32, drop_last=None, channels_last=True, seed=0):
        self.x = torch.as_tensor(x).permute(0, 3, 1, 2).contiguous().to(device)  # uint8 NCHW
        self.y = torch.as_tensor(y, dtype=torch.long).to(device)
        self.bs, self.device, self.train = batch_size, device, train
        self.resolution = resolution
        self.drop_last = train if drop_last is None else drop_last
        self.channels_last = channels_last and device.type == "cuda"
        mean, std = STATS[dataset]
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def __len__(self):
        n = len(self.y)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def _augment(self, xb):
        b, _, h, w = xb.shape
        pad = 4
        xp = F.pad(xb, (pad, pad, pad, pad))  # zero padding (standard CIFAR augmentation)
        oy = torch.randint(0, 2 * pad + 1, (b,), device=xb.device, generator=self.gen)
        ox = torch.randint(0, 2 * pad + 1, (b,), device=xb.device, generator=self.gen)
        ar_h = torch.arange(h, device=xb.device)
        ar_w = torch.arange(w, device=xb.device)
        rows = (oy[:, None] + ar_h)[:, None, :, None].expand(b, 3, h, w)
        cols = (ox[:, None] + ar_w)[:, None, None, :].expand(b, 3, h, w)
        out = xp[torch.arange(b, device=xb.device)[:, None, None, None],
                 torch.arange(3, device=xb.device)[None, :, None, None], rows, cols]
        flip = torch.rand(b, device=xb.device, generator=self.gen) < 0.5
        out = torch.where(flip[:, None, None, None], out.flip(3), out)
        return out

    def preprocess(self, xb_uint8):
        xb = xb_uint8.float().div_(255.0)
        if self.train:
            xb = self._augment(xb)
        xb = (xb - self.mean) / self.std
        if self.resolution != xb.shape[-1]:
            xb = F.interpolate(xb, size=(self.resolution, self.resolution), mode="bilinear",
                               align_corners=False, antialias=self.resolution < xb.shape[-1])
        if self.channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
        return xb

    def __iter__(self):
        n = len(self.y)
        idx = (torch.randperm(n, device=self.device, generator=self.gen) if self.train
               else torch.arange(n, device=self.device))
        for i in range(len(self)):
            bi = idx[i * self.bs:(i + 1) * self.bs]
            yield self.preprocess(self.x[bi]), self.y[bi]


def preprocess_numpy(images_uint8_nhwc, dataset, resolution=32):
    """CPU preprocessing used by inference / ONNX pipelines: uint8 NHWC -> float32 NCHW."""
    x = torch.as_tensor(np.ascontiguousarray(images_uint8_nhwc)).permute(0, 3, 1, 2).float() / 255.0
    mean, std = STATS[dataset]
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    if resolution != x.shape[-1]:
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear",
                          align_corners=False, antialias=resolution < x.shape[-1])
    return x.contiguous()
