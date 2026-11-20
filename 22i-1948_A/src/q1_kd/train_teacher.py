"""Deliverable 1 - train the ResNet-50 teacher on CIFAR-100.

Usage:  python -m src.q1_kd.train_teacher --epochs 30 --lr 0.05
"""
import time

import torch

from ..common.data import DeviceLoader, load_cifar
from ..common.trainer import ce_criterion, evaluate, fit
from ..common.utils import (count_params, fmt_time, get_device, save_checkpoint, save_json,
                            seed_everything, setup_logging, state_dict_size_mb)
from .common_args import Q1_CKPT, Q1_RES, base_parser, teacher_ckpt_path
from .models.teacher import build_teacher


def main():
    p = base_parser("Train ResNet-50 teacher on CIFAR-100")
    p.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet init")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.set_defaults(lr=0.05)
    args = p.parse_args()
    setup_logging("q1_train_teacher")
    print(vars(args))
    out = teacher_ckpt_path()
    if out.exists() and not args.force:
        print(f"[teacher] checkpoint exists at {out} - skipping (use --force to retrain)")
        return
    seed_everything(args.seed)
    device = get_device()
    data = load_cifar("cifar100", smoke=args.smoke)
    train = DeviceLoader(data["x_train"], data["y_train"], args.batch_size, device, "cifar100", True, seed=args.seed)
    test = DeviceLoader(data["x_test"], data["y_test"], 512, device, "cifar100", False)
    model = build_teacher(data["num_classes"], pretrained=not args.no_pretrained and not args.smoke).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"[teacher] ResNet-50 params={count_params(model) / 1e6:.2f}M  ImageNet-init={model.pretrained}  device={device}")
    t0 = time.time()
    history, best, _ = fit(model, train, test, device, epochs=args.epochs, lr=args.lr, wd=args.wd,
                           criterion=ce_criterion(args.label_smoothing), warmup_epochs=args.warmup_epochs,
                           amp=not args.no_amp, run_name="teacher",
                           resume_path=Q1_CKPT / "teacher" / "resnet50_teacher_last.pt",
                           history_csv=Q1_RES / "history" / "teacher_resnet50.csv")
    final = evaluate(model, test, device)
    meta = {"arch": "resnet50", "dataset": "cifar100", "num_classes": data["num_classes"],
            "classes": data["classes"], "resolution": 32, "test_top1": final["top1"],
            "test_top5": final["top5"], "best_epoch_top1": best["top1"], "epochs": args.epochs,
            "imagenet_init": model.pretrained, "args": vars(args)}
    save_checkpoint(out, model, meta)
    metrics = {"model": "resnet50", "variant": "teacher", "params": count_params(model),
               "size_mb": state_dict_size_mb(model), "test_top1": final["top1"], "test_top5": final["top5"],
               "best_epoch_top1": best["top1"], "epochs": args.epochs, "train_time_s": time.time() - t0}
    save_json(metrics, Q1_RES / "metrics" / "teacher_resnet50.json")
    print(f"[teacher] final test top1={final['top1']:.2f} top5={final['top5']:.2f} "
          f"(time {fmt_time(time.time() - t0)}) -> {out}")
    (Q1_CKPT / "teacher" / "resnet50_teacher_last.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
