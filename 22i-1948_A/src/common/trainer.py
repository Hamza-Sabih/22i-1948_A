"""Generic training / evaluation engine (AMP, cosine LR with warm-up, resumable)."""
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AverageMeter, fmt_time, topk_correct, write_csv


@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    loss_m, n, c1, c5 = AverageMeter(), 0, 0, 0
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits = model(x)
        loss_m.update(F.cross_entropy(logits.float(), y).item(), y.size(0))
        a, b = topk_correct(logits, y, (1, 5))
        c1, c5, n = c1 + a, c5 + b, n + y.size(0)
    return {"loss": loss_m.avg, "top1": 100.0 * c1 / max(n, 1), "top5": 100.0 * c5 / max(n, 1)}


def build_optimizer(params, lr, wd, momentum=0.9):
    """SGD-Nesterov; no weight decay on BatchNorm / bias parameters."""
    decay, no_decay = [], []
    for p in params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.SGD([{"params": decay, "weight_decay": wd},
                            {"params": no_decay, "weight_decay": 0.0}],
                           lr=lr, momentum=momentum, nesterov=True)


def cosine_with_warmup(optimizer, total_steps, warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def ce_criterion(label_smoothing=0.0):
    def fn(model, x, y):
        logits = model(x)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=label_smoothing)
        return loss, logits, {"ce": loss.item()}
    return fn


def fit(model, train_loader, val_loader, device, *, epochs, lr, wd, criterion,
        trainable: nn.Module = None, eval_model: nn.Module = None, warmup_epochs=1,
        amp=True, run_name="run", resume_path=None, history_csv=None,
        report_fn=None, log_every_epoch=True, grad_clip=None):
    """Train `trainable` (defaults to `model`) using `criterion(trainable, x, y)`.

    criterion returns (loss, logits, dict_of_scalar_parts). `eval_model` is the network
    evaluated on `val_loader` (the student when trainable is a KD wrapper).
    `report_fn(epoch, val_top1)` may raise (e.g. optuna.TrialPruned).
    Returns (history list, best metrics dict, best_state_dict).
    """
    trainable = trainable or model
    eval_model = eval_model or model
    use_amp = amp and device.type == "cuda"
    opt = build_optimizer(trainable.parameters(), lr, wd)
    steps = epochs * len(train_loader)
    sched = cosine_with_warmup(opt, steps, warmup_epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, history, best = 0, [], {"top1": -1.0}
    best_state = None

    if resume_path and Path(resume_path).exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        trainable.load_state_dict(ck["trainable"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, history, best = ck["epoch"], ck["history"], ck["best"]
        best_state = ck.get("best_state")
        print(f"[{run_name}] resumed from epoch {start_epoch}/{epochs}")

    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        trainable.train()
        te = time.time()
        meters, loss_m, c1, n = {}, AverageMeter(), 0, 0
        for x, y in train_loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, logits, parts = criterion(trainable, x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(trainable.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            bs = y.size(0)
            loss_m.update(loss.item(), bs)
            for k, v in parts.items():
                meters.setdefault(k, AverageMeter()).update(v, bs)
            c1 += topk_correct(logits.detach(), y, (1,))[0]
            n += bs
        val = evaluate(eval_model, val_loader, device, amp)
        rec = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "train_loss": loss_m.avg,
               "train_top1": 100.0 * c1 / max(n, 1),
               **{f"train_{k}": m.avg for k, m in meters.items()},
               "val_loss": val["loss"], "val_top1": val["top1"], "val_top5": val["top5"],
               "epoch_time_s": time.time() - te}
        history.append(rec)
        if val["top1"] > best["top1"]:
            best = {"epoch": epoch + 1, **val}
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        if log_every_epoch:
            parts_s = " ".join(f"{k}={m.avg:.3f}" for k, m in meters.items())
            print(f"[{run_name}] ep {epoch + 1:3d}/{epochs} loss={loss_m.avg:.4f} {parts_s} "
                  f"train@1={rec['train_top1']:.2f} val@1={val['top1']:.2f} val@5={val['top5']:.2f} "
                  f"({rec['epoch_time_s']:.1f}s, total {fmt_time(time.time() - t0)})", flush=True)
        if history_csv:
            write_csv(history, history_csv)
        if resume_path:
            Path(resume_path).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(resume_path) + ".tmp")
            torch.save({"trainable": trainable.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch + 1, "history": history, "best": best,
                        "best_state": best_state}, tmp)
            tmp.replace(resume_path)  # atomic: a Colab disconnect never leaves a corrupt file
        if report_fn is not None:
            report_fn(epoch, val["top1"])
    return history, best, best_state
