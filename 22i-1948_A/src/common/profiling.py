"""Hardware measurements: CPU latency (PyTorch & ONNX Runtime), storage size,
peak inference memory, FLOPs. Used by both the NAS hardware-aware search and the
benchmarking scripts."""
import copy
import gc
import os
import statistics
import time

import numpy as np
import torch

from .utils import count_params, state_dict_size_mb


def _stats(times_ms):
    t = sorted(times_ms)
    return {"mean_ms": statistics.fmean(t), "median_ms": statistics.median(t),
            "p90_ms": t[int(0.9 * (len(t) - 1))], "std_ms": statistics.pstdev(t),
            "min_ms": t[0]}


@torch.inference_mode()
def torch_cpu_latency(model, input_shape=(1, 3, 32, 32), warmup=10, runs=50, threads=1):
    """Measured wall-clock CPU latency of a PyTorch model (eval mode, fp32)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(*input_shape)
    for _ in range(warmup):
        m(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m(x)
        times.append((time.perf_counter() - t0) * 1e3)
    torch.set_num_threads(prev)
    del m
    return _stats(times)


def ort_session(onnx_path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


def ort_cpu_latency(onnx_path_or_session, input_shape=(1, 3, 32, 32), warmup=10, runs=100, threads=1):
    sess = (onnx_path_or_session if not isinstance(onnx_path_or_session, (str, os.PathLike))
            else ort_session(onnx_path_or_session, threads))
    name = sess.get_inputs()[0].name
    x = np.random.randn(*input_shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1e3)
    return _stats(times)


@torch.inference_mode()
def peak_inference_memory_mb(model, input_shape=(1, 3, 32, 32)):
    """Peak CPU memory allocated by the PyTorch allocator during one forward pass.

    Uses torch.profiler memory events (allocations +, frees -) replayed in time
    order; the running maximum is the peak activation + workspace memory. The
    weights are counted separately (they are resident before the pass starts),
    so peak_total = weights + peak_activations. Returns None when unsupported.
    """
    try:
        from torch.profiler import ProfilerActivity, profile
        m = copy.deepcopy(model).cpu().eval()
        x = torch.randn(*input_shape)
        m(x)  # warm-up (lazy init)
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            m(x)
        # leaf events only (allocating ops such as aten::empty and '[memory]' free events),
        # so an allocation is not counted again by its parent op
        events = [(e.time_range.start, e.cpu_memory_usage) for e in prof.events()
                  if len(e.cpu_children) == 0 and (e.cpu_memory_usage or 0) != 0]
        events.sort(key=lambda t: t[0])
        cur = peak = 0
        for _, mem in events:
            cur += mem
            peak = max(peak, cur)
        weights_mb = sum(p.numel() * p.element_size() for p in m.state_dict().values()) / 2 ** 20
        del m
        gc.collect()
        return {"peak_activation_mb": peak / 2 ** 20, "weights_mb": weights_mb,
                "peak_total_mb": peak / 2 ** 20 + weights_mb}
    except Exception as ex:  # pragma: no cover - depends on build
        print(f"[profiling] peak memory not supported: {ex}")
        return None


def gpu_peak_memory_mb(model, input_shape=(1, 3, 32, 32)):
    if not torch.cuda.is_available():
        return None
    m = copy.deepcopy(model).cuda().eval()
    x = torch.randn(*input_shape, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        m(x)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    del m
    torch.cuda.empty_cache()
    return peak


def count_flops(model, input_shape=(1, 3, 32, 32)):
    """FLOPs of one forward pass (torch.utils.flop_counter; 1 MAC = 2 FLOPs)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        m = copy.deepcopy(model).cpu().eval()
        with FlopCounterMode(display=False) as fc, torch.inference_mode():
            m(torch.randn(*input_shape))
        return int(fc.get_total_flops())
    except Exception as ex:  # pragma: no cover
        print(f"[profiling] flop counting failed: {ex}")
        return None


def profile_model(model, input_shape=(1, 3, 32, 32), threads=1, warmup=10, runs=50, memory=True):
    """Full hardware profile used by NAS and benchmarking."""
    lat = torch_cpu_latency(model, input_shape, warmup, runs, threads)
    out = {"params": count_params(model), "size_mb": state_dict_size_mb(model),
           "flops": count_flops(model, input_shape),
           "cpu_latency_ms": lat["median_ms"], "cpu_latency_mean_ms": lat["mean_ms"],
           "cpu_latency_p90_ms": lat["p90_ms"]}
    if memory:
        mem = peak_inference_memory_mb(model, input_shape)
        out["peak_mem_mb"] = mem["peak_total_mb"] if mem else None
        out["peak_act_mem_mb"] = mem["peak_activation_mb"] if mem else None
    return out
