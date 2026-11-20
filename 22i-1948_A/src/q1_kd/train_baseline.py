"""Deliverable 3 - baseline pipeline: train the three students WITHOUT distillation
(plain cross-entropy on hard labels, identical schedule/augmentation to the KD runs).

Usage:  python -m src.q1_kd.train_baseline --models mobilenetv2 squeezenet efficientnet_lite0 --epochs 30
"""
from ..common.trainer import ce_criterion
from ..common.utils import setup_logging
from .common_args import base_parser
from .models.students import STUDENTS
from .student_runner import run_student


def main():
    p = base_parser("Baseline (no-KD) training of student networks on CIFAR-100")
    p.add_argument("--models", nargs="+", default=list(STUDENTS), choices=list(STUDENTS))
    args = p.parse_args()
    setup_logging("q1_train_baseline")
    print(vars(args))

    def factory(student, device, data):
        return student, ce_criterion(0.0), {"kd": False}

    for name in args.models:
        run_student(name, "baseline", args, factory)


if __name__ == "__main__":
    main()
