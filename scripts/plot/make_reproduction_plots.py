"""Plot only values emitted by official DPPO run.log files.

This script never digitizes the supplied paper screenshots and never invents
missing curves. It writes a source CSV beside each figure and refuses to call
an incomplete campaign complete in the report.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs" / "gym-finetune"
RESULT_ROOT = ROOT / "results"
PLOT_ROOT = ROOT / "plots"
ENVIRONMENTS = {"hopper-v2": "Hopper", "walker2d-v2": "Walker2D", "halfcheetah-v2": "HalfCheetah"}
FIG4_METHODS = {"rwr": "DRWR", "awr": "DAWR", "dipo": "DIPO", "idql": "IDQL", "dql": "DQL", "qsm": "QSM", "ppo_diffusion": "DPPO"}
FIG18_METHODS = {"ppo_diffusion": "DPPO", "ppo_gaussian": "Gaussian-MLP"}
LINE_RE = re.compile(r"(?P<itr>\d+): step\s+(?P<steps>\d+)\s+\|.*?reward\s+(?P<reward>[-+]?\d+(?:\.\d+)?)\s+\|.*?t:\s+(?P<seconds>[-+]?\d+(?:\.\d+)?)")


def latest_run(env: str, token: str, seed: int) -> Path | None:
    prefix = f"{env.replace('-v2', '-medium-v2')}_"
    candidates = []
    for group in LOG_ROOT.glob(f"{prefix}*"):
        if token not in group.name:
            continue
        for run in group.iterdir() if group.is_dir() else []:
            if run.is_dir() and run.name.endswith(f"_{seed}") and (run / "run.log").is_file():
                candidates.append(run)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def parse_run(run: Path, env: str, method: str, seed: int) -> list[dict]:
    rows = []
    for line in (run / "run.log").read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.search(line)
        if match:
            rows.append({"env": env, "method": method, "seed": seed, "iteration": int(match["itr"]), "env_steps": int(match["steps"]), "reward": float(match["reward"]), "iteration_seconds": float(match["seconds"]), "run_dir": str(run.resolve()), "source_log": str((run / "run.log").resolve())})
    return rows


def collect(kind: str, seed: int) -> tuple[list[dict], list[str]]:
    mapping = FIG4_METHODS if kind == "fig4" else FIG18_METHODS
    all_rows, missing = [], []
    for env in ENVIRONMENTS:
        for token, label in mapping.items():
            run = latest_run(env, token, seed)
            if run is None:
                missing.append(f"{env}/{label}/seed={seed}")
            else:
                all_rows.extend(parse_run(run, env, label, seed))
    return all_rows, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("fig4", "fig18"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, missing = collect(args.kind, args.seed)
    if not rows:
        raise SystemExit("No official run.log data found; no figure was generated.")
    out_result = RESULT_ROOT / args.kind
    out_plot = PLOT_ROOT / args.kind
    out_result.mkdir(parents=True, exist_ok=True)
    out_plot.mkdir(parents=True, exist_ok=True)
    csv_path = out_result / f"{args.kind}_seed{args.seed}_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {"DPPO": "#c83b3b", "Gaussian-MLP": "#3267a8", "DRWR": "#6f6f6f", "DAWR": "#d28b28", "DIPO": "#48a868", "IDQL": "#8756a5", "DQL": "#3d8f9f", "QSM": "#b05a3c"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=False, constrained_layout=True)
    for ax, (env, title) in zip(axes, ENVIRONMENTS.items()):
        env_rows = [row for row in rows if row["env"] == env]
        for method in sorted({row["method"] for row in env_rows}, key=lambda x: (x != "DPPO", x)):
            method_rows = sorted((row for row in env_rows if row["method"] == method), key=lambda row: row["env_steps"])
            ax.plot([row["env_steps"] / 1e6 for row in method_rows], [row["reward"] for row in method_rows], label=method, color=colors.get(method, "#444444"), lw=2.0 if method == "DPPO" else 1.1)
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlabel("Environment steps (M)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, color="#dddddd", lw=0.5, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Training reward", fontsize=7)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.08))
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    base = out_plot / f"{args.kind}_seed{args.seed}"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"source_csv={csv_path}")
    print(f"plot_png={base.with_suffix('.png')}")
    if missing:
        print("missing=" + ", ".join(missing))
        print("WARNING: figure is partial; missing curves were not fabricated.")
    else:
        print("campaign=complete-for-requested-seed")


if __name__ == "__main__":
    main()
