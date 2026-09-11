"""Microbenchmarks for the two narrowly scoped PPODiffusion.loss candidates.

This script does not import the project model. It reproduces the exact tensor
operations at the formal Hopper batch scale so that the benchmark isolates the
candidate operations and records CUDA-synchronised timings and equivalence.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeats: int, warmups: int) -> tuple[list[float], torch.Tensor]:
    output = None
    for _ in range(warmups):
        output = fn()
        sync()
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        output = fn()
        sync()
        samples.append(time.perf_counter() - start)
    assert output is not None
    return samples, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this microbenchmark")
    device = torch.device("cuda:0")
    gamma = 0.99
    denoising_inds = torch.arange(args.batch, device=device, dtype=torch.long) % args.steps
    advantages = torch.linspace(-1.0, 1.0, args.batch, device=device, dtype=torch.float32)

    def official() -> torch.Tensor:
        # Exact v0.6 operation, including the host-side list construction.
        return torch.tensor(
            [gamma ** (args.steps - i - 1) for i in denoising_inds]
        ).to(device)

    def vectorized() -> torch.Tensor:
        exponent = args.steps - denoising_inds - 1
        return torch.pow(
            torch.tensor(gamma, device=advantages.device, dtype=advantages.dtype),
            exponent.to(dtype=advantages.dtype),
        )

    official_times, official_out = timed(official, args.repeats, args.warmups)
    vectorized_times, vectorized_out = timed(vectorized, args.repeats, args.warmups)
    abs_error = (official_out - vectorized_out).abs()
    denom = official_out.abs().clamp_min(torch.finfo(official_out.dtype).tiny)
    rel_error = abs_error / denom

    # Independent quantile no-op check at the same batch size.
    lower = 0.0
    upper = 1.0
    q_min = torch.quantile(advantages, lower)
    q_max = torch.quantile(advantages, upper)
    quantile_clamped = advantages.clamp(min=q_min, max=q_max)
    quantile_abs_error = (quantile_clamped - advantages).abs().max().item()
    quantile_times, _ = timed(
        lambda: advantages.clamp(
            min=torch.quantile(advantages, lower),
            max=torch.quantile(advantages, upper),
        ),
        args.repeats,
        args.warmups,
    )

    result = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "device": str(device),
        "batch": args.batch,
        "ft_denoising_steps": args.steps,
        "gamma_denoising": gamma,
        "denoising_index_pattern": "arange(batch) % steps",
        "official_times_s": official_times,
        "vectorized_times_s": vectorized_times,
        "official_median_s": statistics.median(official_times),
        "vectorized_median_s": statistics.median(vectorized_times),
        "speedup_official_over_vectorized": statistics.median(official_times)
        / statistics.median(vectorized_times),
        "max_abs_error": abs_error.max().item(),
        "max_rel_error": rel_error.max().item(),
        "official_dtype": str(official_out.dtype),
        "vectorized_dtype": str(vectorized_out.dtype),
        "official_device": str(official_out.device),
        "vectorized_device": str(vectorized_out.device),
        "all_outputs_finite": bool(torch.isfinite(official_out).all() and torch.isfinite(vectorized_out).all()),
        "quantile_lower": lower,
        "quantile_upper": upper,
        "quantile_clamp_max_abs_error": quantile_abs_error,
        "quantile_clamp_times_s": quantile_times,
        "quantile_clamp_median_s": statistics.median(quantile_times),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
