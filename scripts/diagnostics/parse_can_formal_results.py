"""Parse the official Robomimic Can DPPO formal run into public artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    return parser.parse_args()


def load_results(run_dir: Path) -> list[dict]:
    with (run_dir / "result.pkl").open("rb") as handle:
        results = pickle.load(handle)
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise TypeError("result.pkl is not a list of dictionaries")
    return results


def parse_log_evals(run_dir: Path) -> list[tuple[float, float]]:
    text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\s+", " ", text)
    pattern = re.compile(
        r"eval: success rate\s+([0-9.]+)\s+\|\s+avg episode reward\s+([0-9.]+)"
    )
    return [(float(success), float(reward)) for success, reward in pattern.findall(text)]


def write_csv(results: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "iteration",
        "environment_steps",
        "high_level_decision_steps",
        "eval_episode_reward",
        "eval_success_rate",
        "train_episode_reward",
        "train_success_rate",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in results:
            row = {
                "iteration": item["itr"],
                "environment_steps": item["step"],
                "high_level_decision_steps": item["step"] / 4,
                "eval_episode_reward": item.get("eval_episode_reward", ""),
                "eval_success_rate": item.get("eval_success_rate", ""),
                "train_episode_reward": item.get("train_episode_reward", ""),
                "train_success_rate": item.get("train_success_rate", ""),
            }
            writer.writerow(row)


def write_plots(results: list[dict], repo: Path) -> None:
    import matplotlib.pyplot as plt

    eval_rows = [item for item in results if "eval_success_rate" in item]
    plot_dir = repo / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.2, 4.4), dpi=160)
    plt.plot(
        [item["step"] for item in eval_rows],
        [item["eval_success_rate"] for item in eval_rows],
        marker="o",
        linewidth=1.8,
        markersize=4,
    )
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation success rate")
    plt.title("ROBOMIMIC Can | DPPO | seed=42 | state input\nsingle-seed reproduction")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "fig4_robomimic_can_dppo_seed42_success.png")
    plt.close()

    plt.figure(figsize=(7.2, 4.4), dpi=160)
    plt.plot(
        [item["step"] for item in eval_rows],
        [item["eval_episode_reward"] for item in eval_rows],
        marker="o",
        linewidth=1.8,
        markersize=4,
    )
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation episode reward")
    plt.title("ROBOMIMIC Can | DPPO | seed=42 | state input\nsingle-seed reproduction")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "fig4_robomimic_can_dppo_seed42_reward.png")
    plt.close()


def write_audit(results: list[dict], log_evals: list[tuple[float, float]], output: Path) -> None:
    eval_rows = [item for item in results if "eval_success_rate" in item]
    if len(eval_rows) != len(log_evals):
        raise AssertionError(f"result.pkl eval count {len(eval_rows)} != run.log {len(log_evals)}")
    indices = sorted({0, len(eval_rows) // 2, len(eval_rows) - 1})
    checks = []
    for index in indices:
        item = eval_rows[index]
        log_success, log_reward = log_evals[index]
        success_error = abs(item["eval_success_rate"] - log_success)
        reward_error = abs(item["eval_episode_reward"] - log_reward)
        checks.append(
            {
                "iteration": item["itr"],
                "environment_steps": item["step"],
                "result.pkl_value": {
                    "eval_success_rate": item["eval_success_rate"],
                    "eval_episode_reward": item["eval_episode_reward"],
                },
                "run.log_value": {
                    "eval_success_rate": log_success,
                    "eval_episode_reward": log_reward,
                },
                "error": {
                    "eval_success_rate": success_error,
                    "eval_episode_reward": reward_error,
                },
                "rounding_match": bool(success_error <= 5e-5 and reward_error <= 5e-5),
                "audit_result": "PASS" if bool(success_error <= 5e-5 and reward_error <= 5e-5) else "FAIL",
            }
        )
    payload = {
        "source_result": "result.pkl",
        "source_log": "run.log",
        "evaluation_points_in_result": len(eval_rows),
        "sampled_points": checks,
        "audit_result": "PASS" if all(item["audit_result"] == "PASS" for item in checks) else "FAIL",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = load_results(args.run_dir)
    log_evals = parse_log_evals(args.run_dir)
    if len(results) != 151:
        raise AssertionError(f"expected 151 result rows, got {len(results)}")
    if results[-1].get("itr") != 150 or results[-1].get("step") != 8_100_000:
        raise AssertionError(f"unexpected final result row: {results[-1]}")
    write_csv(results, args.repo / "results/fig4_robomimic_can_dppo_seed42.csv")
    write_plots(results, args.repo)
    write_audit(results, log_evals, args.repo / "reports/P4A_RAW_DATA_AUDIT.json")
    print(json.dumps({"rows": len(results), "evals": len(log_evals), "audit": "PASS"}))


if __name__ == "__main__":
    main()
