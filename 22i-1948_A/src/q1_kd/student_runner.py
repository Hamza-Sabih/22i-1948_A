"""Shared logic for training a student (baseline or distilled) and recording results."""
import time

import torch

from ..common.data import DeviceLoader, load_cifar
from ..common.trainer import evaluate, fit
from ..common.utils import (count_params, fmt_time, get_device, save_checkpoint, save_json,
                            seed_everything, state_dict_size_mb)
from .common_args import Q1_CKPT, Q1_RES, student_ckpt_path
from .models.students import PRETTY, build_student


def run_student(name, variant, args, criterion_factory):
    """variant in {"baseline", "kd"}; criterion_factory(student, device, data) -> (trainable, criterion, extra_meta)."""
    out = student_ckpt_path(name, variant)
    if out.exists() and not args.force:
        print(f"[{name}/{variant}] checkpoint exists at {out} - skipping (use --force to retrain)")
        return
    seed_everything(args.seed)
    device = get_device()
    data = load_cifar("cifar100", smoke=args.smoke)
    train = DeviceLoader(data["x_train"], data["y_train"], args.batch_size, device, "cifar100", True, seed=args.seed)
    test = DeviceLoader(data["x_test"], data["y_test"], 512, device, "cifar100", False)
    student = build_student(name, data["num_classes"]).to(device)
    if device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)
    trainable, criterion, extra = criterion_factory(student, device, data)
    print(f"[{name}/{variant}] {PRETTY[name]} params={count_params(student) / 1e6:.3f}M device={device}")
    run = f"{name}_{variant}"
    t0 = time.time()
    history, best, _ = fit(student, train, test, device, epochs=args.epochs, lr=args.lr, wd=args.wd,
                           criterion=criterion, trainable=trainable, eval_model=student,
                           warmup_epochs=args.warmup_epochs, amp=not args.no_amp, run_name=run,
                           resume_path=Q1_CKPT / variant / f"{run}_last.pt",
                           history_csv=Q1_RES / "history" / f"{run}.csv")
    final = evaluate(student, test, device)
    train_time = time.time() - t0
    meta = {"arch": name, "variant": variant, "dataset": "cifar100", "num_classes": data["num_classes"],
            "classes": data["classes"], "resolution": 32, "test_top1": final["top1"], "test_top5": final["top5"],
            "best_epoch_top1": best["top1"], "epochs": args.epochs, "args": vars(args), **extra}
    save_checkpoint(out, student, meta)
    save_json({"model": name, "variant": variant, "params": count_params(student),
               "size_mb": state_dict_size_mb(student), "test_top1": final["top1"], "test_top5": final["top5"],
               "best_epoch_top1": best["top1"], "epochs": args.epochs, "train_time_s": train_time, **extra},
              Q1_RES / "metrics" / f"{run}.json")
    print(f"[{run}] final test top1={final['top1']:.2f} top5={final['top5']:.2f} "
          f"(time {fmt_time(train_time)}) -> {out}")
    (Q1_CKPT / variant / f"{run}_last.pt").unlink(missing_ok=True)
