"""Deliverable 6 (hardware-aware part) - deployment constraints and the
resource-constrained selection rule, based on MEASURED CPU latency, model storage
size and peak inference memory (see src/common/profiling.py)."""
from dataclasses import asdict, dataclass

from ..common.profiling import profile_model


@dataclass
class DeploymentConstraints:
    max_latency_ms: float = 5.0      # median single-image CPU latency (1 thread)
    max_size_mb: float = 1.0         # serialized fp32 weights on disk (~250k parameters)
    max_peak_mem_mb: float = 6.0     # peak inference memory (weights + activations)
    threads: int = 1                 # CPU threads used for measurement (edge-like setting)

    def to_dict(self):
        return asdict(self)


def hardware_profile(model, resolution, constraints: DeploymentConstraints, runs=30, warmup=5):
    return profile_model(model, (1, 3, resolution, resolution), threads=constraints.threads,
                         warmup=warmup, runs=runs, memory=True)


def check_constraints(hw: dict, c: DeploymentConstraints):
    """Return list of violated constraints (empty list -> deployable)."""
    v = []
    if hw["cpu_latency_ms"] > c.max_latency_ms:
        v.append(f"latency {hw['cpu_latency_ms']:.2f}ms > {c.max_latency_ms}ms")
    if hw["size_mb"] > c.max_size_mb:
        v.append(f"size {hw['size_mb']:.2f}MB > {c.max_size_mb}MB")
    if hw.get("peak_mem_mb") is not None and hw["peak_mem_mb"] > c.max_peak_mem_mb:
        v.append(f"peak_mem {hw['peak_mem_mb']:.2f}MB > {c.max_peak_mem_mb}MB")
    return v


def constrained_score(val_acc, hw, c: DeploymentConstraints, lat_weight=2.0, size_weight=1.0):
    """Resource-aware objective (higher is better), only defined for feasible models:

        score = val_acc(%) - lat_weight * latency/latency_budget - size_weight * size/size_budget

    Accuracy dominates; among similarly accurate models the faster / smaller one wins
    (e.g. halving latency is worth ~1 accuracy point with the default weights)."""
    return (val_acc - lat_weight * hw["cpu_latency_ms"] / c.max_latency_ms
            - size_weight * hw["size_mb"] / c.max_size_mb)
