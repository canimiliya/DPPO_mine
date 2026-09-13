"""Build the single-seed ROBOMIMIC Can DPPO/IDQL/DIPO comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    methods = ["dppo", "idql", "dipo"]
    rows: list[dict[str, str]] = []
    for method in methods:
        path = args.repo / f"results/fig4_robomimic_can_{method}_seed42.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for item in csv.DictReader(handle):
                if item["eval_success_rate"] == "":
                    continue
                rows.append({
                    "method": method.upper(),
                    "seed": "42",
                    "iteration": item["iteration"],
                    "environment_steps": item["environment_steps"],
                    "high_level_decision_steps": item["high_level_decision_steps"],
                    "eval_success_rate": item["eval_success_rate"],
                    "eval_episode_reward": item["eval_episode_reward"],
                })
    output = args.repo / "results/fig4_robomimic_can_three_methods_seed42.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "seed", "iteration", "environment_steps", "high_level_decision_steps", "eval_success_rate", "eval_episode_reward"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    plot_dir = args.repo / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"DPPO": "#1f77b4", "IDQL": "#ff7f0e", "DIPO": "#2ca02c"}
    for metric, ylabel, suffix in [
        ("eval_success_rate", "Evaluation success rate", "success"),
        ("eval_episode_reward", "Evaluation episode reward", "reward"),
    ]:
        plt.figure(figsize=(7.4, 4.6), dpi=160)
        for method in methods:
            name = method.upper()
            subset = [row for row in rows if row["method"] == name]
            plt.plot(
                [int(row["environment_steps"]) for row in subset],
                [float(row[metric]) for row in subset],
                marker="o", markersize=3.5, linewidth=1.7,
                color=colors[name], label=name,
            )
        if metric == "eval_success_rate":
            plt.ylim(-0.02, 1.02)
        plt.xlabel("Environment steps")
        plt.ylabel(ylabel)
        plt.title("ROBOMIMIC Can | DPPO vs IDQL vs DIPO | seed=42\nsingle-seed comparison")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"fig4_robomimic_can_three_methods_seed42_{suffix}.png")
        plt.close()
    print(f"wrote {len(rows)} evaluation rows to {output}")


if __name__ == "__main__":
    main()
