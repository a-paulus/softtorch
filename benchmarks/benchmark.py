import argparse
import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import softtorch
import torch


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------


@dataclass
class FunctionSpec:
    name: str
    fn: Callable
    kwargs: dict = field(default_factory=dict)
    methods: list[str] | None = None
    two_d: bool = False


AXISWISE_SPECS = [
    FunctionSpec("argmax", softtorch.argmax, {"dim": 1}, ["ot", "softsort", "neuralsort", "sorting_network"], two_d=True),
    FunctionSpec("argmin", softtorch.argmin, {"dim": 1}, ["ot", "softsort", "neuralsort", "sorting_network"], two_d=True),
    FunctionSpec("argsort", softtorch.argsort, {"dim": 1}, ["ot", "softsort", "neuralsort", "sorting_network"], two_d=True),
    FunctionSpec("max", softtorch.max, {"dim": 1}, ["ot", "softsort", "neuralsort", "fast_soft_sort", "sorting_network"], two_d=True),
    FunctionSpec("min", softtorch.min, {"dim": 1}, ["ot", "softsort", "neuralsort", "fast_soft_sort", "sorting_network"], two_d=True),
    FunctionSpec("sort", softtorch.sort, {"dim": 1}, ["ot", "softsort", "neuralsort", "fast_soft_sort", "sorting_network"], two_d=True),
    FunctionSpec("rank", softtorch.rank, {"dim": 1}, ["ot", "softsort", "neuralsort", "fast_soft_sort", "sorting_network"], two_d=True),
    FunctionSpec("topk", softtorch.topk, {"dim": 1, "k": 5}, ["ot", "softsort", "neuralsort", "fast_soft_sort", "sorting_network"], two_d=True),
    FunctionSpec("argquantile", softtorch.argquantile, {"dim": 1, "q": 0.25}, ["ot", "softsort", "neuralsort", "sorting_network"], two_d=True),
    FunctionSpec("quantile", softtorch.quantile, {"dim": 1, "q": 0.25}, ["ot", "softsort", "neuralsort", "fast_soft_sort", "sorting_network"], two_d=True),
    FunctionSpec("argmedian", softtorch.argmedian, {"dim": 1}, ["ot", "softsort", "neuralsort", "sorting_network"], two_d=True),
    FunctionSpec("median", softtorch.median, {"dim": 1}, ["ot", "softsort", "neuralsort", "fast_soft_sort", "sorting_network"], two_d=True),
]

ELEMENTWISE_SPECS = [
    FunctionSpec("heaviside", softtorch.heaviside),
    FunctionSpec("round", softtorch.round),
    FunctionSpec("sign", softtorch.sign),
    FunctionSpec("abs", softtorch.abs),
    FunctionSpec("relu", softtorch.relu),
    FunctionSpec("clamp", softtorch.clamp, {"a": -1.0, "b": 1.0}),
    FunctionSpec("greater", softtorch.greater, {"y": 0.0}),
    FunctionSpec("greater_equal", softtorch.greater_equal, {"y": 0.0}),
    FunctionSpec("less", softtorch.less, {"y": 0.0}),
    FunctionSpec("less_equal", softtorch.less_equal, {"y": 0.0}),
    FunctionSpec("eq", softtorch.eq, {"y": 0.0}),
    FunctionSpec("not_equal", softtorch.not_equal, {"y": 0.0}),
    FunctionSpec("isclose", softtorch.isclose, {"y": 0.0}),
]

ALL_SPECS = AXISWISE_SPECS + ELEMENTWISE_SPECS

AXISWISE_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
ELEMENTWISE_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
OT_MAX_SIZE = None  # No cap; OOM is caught by try/except

MODES = ["hard", "smooth", "c0", "c1", "c2"]


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def _input_tensor(size: int, dtype: torch.dtype, two_d: bool, device: str, batch_size: int = 1, seed: int = 42) -> torch.Tensor:
    shape = (batch_size, size) if two_d else (batch_size * size,)
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, dtype=dtype, device=device, generator=gen)


def _reduce_output(res):
    if isinstance(res, tuple):
        return res[0].sum()
    return res.sum()


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def _sync(device: str):
    if device != "cpu":
        torch.cuda.synchronize()


def _microtime(fn: Callable, num_times: int, warmup: int, device: str) -> float:
    for i in range(num_times + warmup):
        if i == warmup:
            _sync(device)
            start = time.perf_counter_ns()
        fn()
    _sync(device)
    end = time.perf_counter_ns()
    return (end - start) / (num_times * 1e6)


def measure_runtime(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    device: str,
    num_runs: int = 20,
    warmup: int = 3,
    no_grad: bool = True,
) -> tuple[float, float, float]:
    gc.disable()

    def run():
        if no_grad:
            with torch.no_grad():
                fn(*args, **kwargs)
        else:
            fn(*args, **kwargs)

    # Adaptive micro-batching
    first = _microtime(run, 1, warmup=0, device=device)
    num_micro = max(1, int(400 / first))

    times = []
    for i in range(num_runs + warmup):
        dur = _microtime(run, num_micro, warmup=warmup, device=device)
        if i >= warmup:
            times.append(dur)

    gc.enable()
    gc.collect(1)

    arr = np.array(times)
    return arr.mean().item(), arr.std().item(), arr.min().item()


# ---------------------------------------------------------------------------
# Memory measurement (GPU only)
# ---------------------------------------------------------------------------


def measure_peak_memory(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    device: str,
) -> int:
    if device == "cpu":
        return -1

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    inp = args[0].detach().requires_grad_(True)

    out = fn(inp, **kwargs)
    loss = _reduce_output(out)
    loss.backward()

    peak = torch.cuda.max_memory_allocated()
    return peak


# ---------------------------------------------------------------------------
# Core benchmark driver
# ---------------------------------------------------------------------------


def benchmark_single(
    spec: FunctionSpec,
    mode: str,
    method: str | None,
    size: int,
    dtype: torch.dtype,
    device: str,
    num_trials: int = 1,
    softness: float = 1.0,
    batch_size: int = 1,
) -> dict:
    call_kwargs = dict(spec.kwargs)
    call_kwargs["mode"] = mode
    if method is not None:
        call_kwargs["method"] = method
    call_kwargs["softness"] = softness

    fn = spec.fn

    # --- Runtime: fresh random input per trial (data-dependent convergence) ---
    # Seed is deterministic so all modes/methods see identical inputs per trial.
    fwd_times = []
    grad_times = []
    for trial in range(num_trials):
        inp = _input_tensor(size, dtype, spec.two_d, device, batch_size=batch_size, seed=trial)
        args = (inp,)

        fwd_mean, _, _ = measure_runtime(fn, args, call_kwargs, device, num_runs=1)
        fwd_times.append(fwd_mean)

        def grad_run():
            x = inp.detach().requires_grad_(True)
            out = fn(x, **call_kwargs)
            loss = _reduce_output(out)
            loss.backward()

        grad_mean, _, _ = measure_runtime(
            lambda *a, **kw: grad_run(), (), {}, device, num_runs=1, no_grad=False
        )
        grad_times.append(grad_mean)

    fwd_arr = np.array(fwd_times)
    grad_arr = np.array(grad_times)

    # --- Peak memory (shape-dependent) ---
    inp0 = _input_tensor(size, dtype, spec.two_d, device, batch_size=batch_size, seed=0)
    peak_mem = measure_peak_memory(fn, (inp0,), call_kwargs, device)

    category = "axiswise" if spec.methods is not None else "elementwise"
    return {
        "function": spec.name,
        "category": category,
        "mode": mode,
        "method": method if method is not None else "",
        "softness": softness,
        "batch_size": batch_size,
        "dtype": str(dtype),
        "device": device,
        "problem_size": size,
        "fwd_ms_mean": fwd_arr.mean().item(),
        "fwd_ms_std": fwd_arr.std().item(),
        "fwd_ms_min": fwd_arr.min().item(),
        "grad_ms_mean": grad_arr.mean().item(),
        "grad_ms_std": grad_arr.std().item(),
        "grad_ms_min": grad_arr.min().item(),
        "peak_memory_bytes": peak_mem,
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def run_benchmark(
    specs: list[FunctionSpec],
    modes: list[str],
    sizes: list[int] | None,
    dtype: torch.dtype,
    device: str,
    out_path: Path | None,
    plot_callback: Callable | None = None,
    num_trials: int = 1,
    softness_values: list[float] | None = None,
    batch_sizes: list[int] | None = None,
) -> pd.DataFrame:
    if softness_values is None:
        softness_values = [1.0]
    if batch_sizes is None:
        batch_sizes = [1]

    rows: list[dict] = []

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def _record(row: dict):
        rows.append(row)
        if out_path is not None:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    # Collect all (spec, sizes) pairs
    spec_sizes_map = {}
    for spec in specs:
        is_axiswise = spec.methods is not None
        default_sizes = AXISWISE_SIZES if is_axiswise else ELEMENTWISE_SIZES
        spec_sizes_map[spec.name] = sizes if sizes is not None else default_sizes

    # All unique sizes, sorted ascending (outer loop)
    all_sizes = sorted({s for ss in spec_sizes_map.values() for s in ss})

    def _run_one(spec, mode, method, size, softness, batch_size):
        parts = [f"{spec.name} | mode={mode}"]
        if method is not None:
            parts.append(f"method={method}")
        parts.append(f"size={size}")
        if len(softness_values) > 1:
            parts.append(f"softness={softness}")
        if len(batch_sizes) > 1:
            parts.append(f"batch={batch_size}")
        label = " | ".join(parts)
        print(f"  {label}", flush=True)
        try:
            row = benchmark_single(
                spec,
                mode,
                method,
                size,
                dtype,
                device,
                num_trials,
                softness=softness,
                batch_size=batch_size,
            )
            _record(row)
        except Exception as e:
            print(f"    SKIP: {e}", flush=True)

    for size in all_sizes:
        for spec in specs:
            if size not in spec_sizes_map[spec.name]:
                continue
            is_axiswise = spec.methods is not None

            for softness in softness_values:
                for batch_size in batch_sizes:
                    for mode in modes:
                        if mode == "hard":
                            _run_one(spec, mode, None, size, softness, batch_size)
                        elif is_axiswise:
                            for method in spec.methods:
                                if (
                                    OT_MAX_SIZE is not None
                                    and method == "ot"
                                    and size > OT_MAX_SIZE
                                ):
                                    continue
                                _run_one(spec, mode, method, size, softness, batch_size)
                        else:
                            _run_one(spec, mode, None, size, softness, batch_size)

        # Plot after each size is complete
        if plot_callback is not None and rows:
            plot_callback(pd.DataFrame(rows))

    df = pd.DataFrame(rows)

    if out_path is not None:
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SoftTorch operators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", action="store_true", help="Run on GPU (default: CPU)")
    parser.add_argument("--functions", type=str, default=None, help="Comma-separated function names")
    parser.add_argument("--modes", type=str, default=None, help="Comma-separated modes (default: hard,smooth,c0,c1,c2)")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods (e.g. fast_soft_sort,softsort)")
    parser.add_argument("--sizes", type=str, default=None, help="Comma-separated problem sizes")
    parser.add_argument("--dtype", type=str, default="float32", help="Dtype (default: float32)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["axiswise", "elementwise", "all"],
        help="Category to benchmark (default: all)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of runtime trials with fresh random inputs (default: 1)",
    )
    parser.add_argument(
        "--softness",
        type=str,
        default="1.0",
        help="Comma-separated softness values (default: 1.0)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1",
        help="Comma-separated batch sizes (default: 1)",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots after each problem size")
    parser.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    # --- Device setup ---
    if args.gpu:
        if not torch.cuda.is_available():
            parser.error("GPU requested but CUDA is not available.")
        device = "cuda"
    else:
        device = "cpu"

    # --- Dtype ---
    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    dtype = dtype_map.get(args.dtype)
    if dtype is None:
        parser.error(f"Unsupported dtype: {args.dtype}")

    # --- Select specs ---
    if args.category == "axiswise":
        specs = list(AXISWISE_SPECS)
    elif args.category == "elementwise":
        specs = list(ELEMENTWISE_SPECS)
    else:
        specs = list(ALL_SPECS)

    if args.functions is not None:
        names = {n.strip() for n in args.functions.split(",")}
        specs = [s for s in specs if s.name in names]
        if not specs:
            parser.error(f"No matching functions found for: {args.functions}")

    # --- Modes ---
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else list(MODES)

    # --- Methods filter ---
    if args.methods is not None:
        methods_filter = {m.strip() for m in args.methods.split(",")}
        for spec in specs:
            if spec.methods is not None:
                spec.methods = [m for m in spec.methods if m in methods_filter]

    # --- Sizes ---
    sizes = [int(s.strip()) for s in args.sizes.split(",")] if args.sizes else None

    # --- Softness ---
    softness_values = [float(s.strip()) for s in args.softness.split(",")]

    # --- Batch sizes ---
    batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]

    # --- Output path ---
    if args.output:
        out_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f"benchmarks/results/{timestamp}")
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / f"softtorch_benchmark_{device}_{args.dtype}.csv"

    # --- Plot callback ---
    plot_callback = None
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot_results import plot_scaling

        plot_dir = out_path.parent
        fmt = args.plot_format

        def plot_callback(df):
            import matplotlib.pyplot as plt

            plt.style.use("seaborn-v0_8-whitegrid")
            print("  [plotting...]", flush=True)
            plot_scaling(df, plot_dir, fmt)

    # --- Run ---
    print(f"SoftTorch Benchmark — {device.upper()}, dtype={dtype}")
    print(f"Functions: {[s.name for s in specs]}")
    print(f"Modes: {modes}")
    if len(softness_values) > 1:
        print(f"Softness: {softness_values}")
    if len(batch_sizes) > 1:
        print(f"Batch sizes: {batch_sizes}")
    print(f"Output: {out_path}\n")

    run_benchmark(
        specs,
        modes,
        sizes,
        dtype,
        device,
        out_path,
        plot_callback,
        num_trials=args.trials,
        softness_values=softness_values,
        batch_sizes=batch_sizes,
    )


if __name__ == "__main__":
    main()
