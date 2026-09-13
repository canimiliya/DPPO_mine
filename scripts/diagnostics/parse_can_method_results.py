"""Parse a formal ROBOMIMIC Can fine-tuning run into public artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from datetime import datetime
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--method", choices=["IDQL", "DIPO"], required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    return parser.parse_args()


def load_results(run_dir: Path) -> list[dict]:
    with (run_dir / "result.pkl").open("rb") as handle:
        results = pickle.load(handle)
    if not isinstance(results, list) or not all(isinstance(row, dict) for row in results):
        raise TypeError("result.pkl must be a list of dictionaries")
    return results


def log_data(run_dir: Path) -> tuple[list[tuple[float, float]], dict[int, int], float]:
    text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    evals = [
        (float(success), float(reward))
        for success, reward in re.findall(
            r"eval: success rate\s+([0-9.]+)\s+\|\s+avg episode reward\s+([0-9.]+)",
            re.sub(r"\s+", " ", text),
        )
    ]
    steps = {
        int(itr): int(step)
        for itr, step in re.findall(r"\] - (\d+): step\s+([0-9,]+)", text)
    }
    timestamps = [
        datetime.fromisoformat(value)
        for value in re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\]", text, re.MULTILINE)
    ]
    wall_seconds = (max(timestamps) - min(timestamps)).total_seconds() if timestamps else float("nan")
    return evals, steps, wall_seconds


def write_csv(results: list[dict], output: Path) -> None:
    fields = [
        "iteration", "environment_steps", "high_level_decision_steps",
        "eval_success_rate", "eval_episode_reward", "train_success_rate",
        "train_episode_reward",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in results:
            writer.writerow({
                "iteration": item["itr"],
                "environment_steps": item["step"],
                "high_level_decision_steps": item["step"] / 4,
                "eval_success_rate": item.get("eval_success_rate", ""),
                "eval_episode_reward": item.get("eval_episode_reward", ""),
                "train_success_rate": item.get("train_success_rate", ""),
                "train_episode_reward": item.get("train_episode_reward", ""),
            })


def write_plots(results: list[dict], repo: Path, method: str) -> None:
    import matplotlib.pyplot as plt

    eval_rows = [row for row in results if "eval_success_rate" in row]
    stem = f"fig4_robomimic_can_{method.lower()}_seed42"
    plot_dir = repo / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = [row["step"] for row in eval_rows]

    plt.figure(figsize=(7.2, 4.4), dpi=160)
    plt.plot(x, [row["eval_success_rate"] for row in eval_rows], marker="o", linewidth=1.8, markersize=4)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation success rate")
    plt.title(f"ROBOMIMIC Can | {method} | seed=42 | state input\nsingle-seed reproduction")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{stem}_success.png")
    plt.close()

    plt.figure(figsize=(7.2, 4.4), dpi=160)
    plt.plot(x, [row["eval_episode_reward"] for row in eval_rows], marker="o", linewidth=1.8, markersize=4)
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation episode reward")
    plt.title(f"ROBOMIMIC Can | {method} | seed=42 | state input\nsingle-seed reproduction")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{stem}_reward.png")
    plt.close()


def write_audit(results: list[dict], log_evals: list[tuple[float, float]], log_steps: dict[int, int], output: Path) -> None:
    eval_rows = [row for row in results if "eval_success_rate" in row]
    if len(eval_rows) != len(log_evals):
        raise AssertionError(f"result.pkl eval count {len(eval_rows)} != run.log {len(log_evals)}")
    indices = sorted({0, len(eval_rows) // 2, len(eval_rows) - 1})
    checks = []
    for index in indices:
        row = eval_rows[index]
        log_success, log_reward = log_evals[index]
        expected_log_step = 0 if row["itr"] == 0 else log_steps.get(row["itr"] - 1)
        if expected_log_step is None:
            raise AssertionError(f"missing run.log step before eval itr {row['itr']}")
        errors = {
            "eval_success_rate": abs(row["eval_success_rate"] - log_success),
            "eval_episode_reward": abs(row["eval_episode_reward"] - log_reward),
            "environment_steps": abs(row["step"] - expected_log_step),
        }
        passed = errors["eval_success_rate"] <= 5e-5 and errors["eval_episode_reward"] <= 5e-5 and errors["environment_steps"] == 0
        checks.append({
            "iteration": row["itr"],
            "result_pkl": {"environment_steps": row["step"], "eval_success_rate": row["eval_success_rate"], "eval_episode_reward": row["eval_episode_reward"]},
            "run_log": {"environment_steps": expected_log_step, "eval_success_rate": log_success, "eval_episode_reward": log_reward},
            "absolute_error": errors,
            "audit_result": "PASS" if passed else "FAIL",
        })
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
    config = args()
    results = load_results(config.run_dir)
    if len(results) != config.expected_rows:
        raise AssertionError(f"expected {config.expected_rows} result rows, got {len(results)}")
    if results[-1].get("itr") != config.expected_rows - 1:
        raise AssertionError(f"unexpected final iteration: {results[-1]}")
    log_evals, log_steps, wall_seconds = log_data(config.run_dir)
    stem = f"fig4_robomimic_can_{config.method.lower()}_seed42"
    write_csv(results, config.repo / f"results/{stem}.csv")
    write_plots(results, config.repo, config.method)
    write_audit(results, log_evals, log_steps, config.repo / f"reports/P4{'B' if config.method == 'IDQL' else 'C'}_RAW_DATA_AUDIT.json")
    eval_rows = [row for row in results if "eval_success_rate" in row]
    first_ten_seconds = sum(float(row["time"]) for row in results[:10])
    payload = {
        "method": config.method,
        "rows": len(results),
        "eval_points": len(eval_rows),
        "final_iteration": results[-1]["itr"],
        "final_environment_steps": results[-1]["step"],
        "wall_seconds_from_run_log": wall_seconds,
        "first_ten_result_seconds": first_ten_seconds,
        "first_ten_average_seconds": first_ten_seconds / 10.0,
        "first_ten_projection_seconds": first_ten_seconds / 10.0 * len(results),
        "initial_success": eval_rows[0]["eval_success_rate"],
        "final_success": eval_rows[-1]["eval_success_rate"],
        "best_success": max(row["eval_success_rate"] for row in eval_rows),
        "best_iteration": next(row["itr"] for row in eval_rows if row["eval_success_rate"] == max(item["eval_success_rate"] for item in eval_rows)),
        "initial_reward": eval_rows[0]["eval_episode_reward"],
        "final_reward": eval_rows[-1]["eval_episode_reward"],
        "best_reward": max(row["eval_episode_reward"] for row in eval_rows),
        "best_reward_iteration": next(row["itr"] for row in eval_rows if row["eval_episode_reward"] == max(item["eval_episode_reward"] for item in eval_rows)),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
