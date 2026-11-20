"""Deliverable 4 - knowledge-distillation training of the three students.

Loss = (1-alpha)*CE + alpha*T^2*KL(teacher_T || student_T) + beta*feature-hint loss
(see src/q1_kd/kd.py). The ResNet-50 teacher is loaded from its checkpoint and frozen.

Usage:  python -m src.q1_kd.train_kd --temperature 4 --alpha 0.9 --beta 1.0 --epochs 30
"""
import torch

from ..common.utils import load_checkpoint, setup_logging
from .common_args import base_parser, teacher_ckpt_path
from .kd import DistillationWrapper, kd_criterion
from .models.students import STUDENTS
from .models.teacher import build_teacher
from .student_runner import run_student


def load_teacher(device, num_classes=100):
    path = teacher_ckpt_path()
    if not path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found at {path}; run src.q1_kd.train_teacher first")
    ck = load_checkpoint(path)
    teacher = build_teacher(num_classes, pretrained=False)
    teacher.load_state_dict(ck["state_dict"])
    teacher = teacher.to(device).eval()
    if device.type == "cuda":
        teacher = teacher.to(memory_format=torch.channels_last)
    return teacher, ck["meta"]


def main():
    p = base_parser("Knowledge distillation ResNet-50 -> students on CIFAR-100")
    p.add_argument("--models", nargs="+", default=list(STUDENTS), choices=list(STUDENTS))
    p.add_argument("--temperature", type=float, default=4.0, help="softmax temperature T")
    p.add_argument("--alpha", type=float, default=0.9, help="weight of soft-target loss (1-alpha for hard labels)")
    p.add_argument("--beta", type=float, default=1.0, help="weight of feature-distillation loss (0 disables)")
    p.add_argument("--hints", nargs="+", default=["mid", "final"], choices=["mid", "final"])
    p.add_argument("--feat-loss", default="fitnet", choices=["fitnet", "at"])
    args = p.parse_args()
    setup_logging("q1_train_kd")
    print(vars(args))

    cache = {}

    def factory(student, device, data):
        if "teacher" not in cache:
            cache["teacher"], meta = load_teacher(device, data["num_classes"])
            print(f"[kd] teacher loaded (test top1={meta.get('test_top1', float('nan')):.2f})")
        wrapper = DistillationWrapper(student, cache["teacher"], args.temperature, args.alpha,
                                      args.beta, args.hints, args.feat_loss).to(device)
        extra = {"kd": True, "temperature": args.temperature, "alpha": args.alpha, "beta": args.beta,
                 "hints": "+".join(args.hints), "feat_loss": args.feat_loss}
        return wrapper, kd_criterion, extra

    for name in args.models:
        run_student(name, "kd", args, factory)


if __name__ == "__main__":
    main()
