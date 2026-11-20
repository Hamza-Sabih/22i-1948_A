"""Shared CLI arguments and helpers for the Q1 scripts."""
import argparse

from ..common.utils import CKPT_DIR, RESULTS_DIR

Q1_CKPT = CKPT_DIR / "q1"
Q1_RES = RESULTS_DIR / "q1"


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--force", action="store_true", help="retrain even if a final checkpoint exists")
    p.add_argument("--smoke", action="store_true", help="synthetic data, for a quick functional test")
    return p


def teacher_ckpt_path():
    return Q1_CKPT / "teacher" / "resnet50_teacher.pt"


def student_ckpt_path(name, variant):
    return Q1_CKPT / variant / f"{name}_{variant}.pt"
