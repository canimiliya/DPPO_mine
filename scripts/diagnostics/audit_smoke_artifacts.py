"""Audit short smoke checkpoints for finiteness and an actual parameter change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def audit(run_dir: Path) -> dict:
    state0 = torch.load(run_dir / "checkpoint" / "state_0.pt", map_location="cpu", weights_only=True)
    state1 = torch.load(run_dir / "checkpoint" / "state_1.pt", map_location="cpu", weights_only=True)
    model0 = state0["model"]
    model1 = state1["model"]
    finite = True
    changed = 0
    max_abs_delta = 0.0
    for key, value0 in model0.items():
        value1 = model1[key]
        if torch.is_floating_point(value0):
            finite = finite and bool(torch.isfinite(value0).all()) and bool(torch.isfinite(value1).all())
            delta = (value1 - value0).abs().max().item()
            max_abs_delta = max(max_abs_delta, delta)
            changed += int(delta > 0.0)
    return {
        "run_dir": str(run_dir.resolve()),
        "state_0_itr": state0.get("itr"),
        "state_1_itr": state1.get("itr"),
        "floating_tensors_finite": finite,
        "changed_floating_tensors": changed,
        "max_abs_parameter_delta": max_abs_delta,
        "optimizer_update_evidence": bool(changed > 0 and max_abs_delta > 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [audit(run_dir) for run_dir in args.run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
