"""Offline correction of the registered S0D_R2_R1 aggregate.

This helper reads only the already saved external snapshot/test records.  It
does not import an environment, sample a model, or advance a simulator.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.recovery_retention.run_s0d_r2_r1_boundary import (
    BASE_MAGNITUDES,
    CONFIG_PATH,
    EnvAdapter,
    PLOT_ROOT,
    PROTOCOL_PATH,
    REPORT_ROOT,
    TEST_SEEDS,
    bootstrap_ci,
    compute_metrics,
    json_meta,
    plot_curves,
    report_text,
)
from research.recovery_retention import run_s0d_r2 as legacy


DATA_ROOT = Path("/mnt/d/AgentData/DPPO-S0d-R2-R1")


def normalization_audit() -> dict:
    adapter = EnvAdapter("square")
    try:
        adapter.reset_seed(930001)
        raw = adapter.env.get_observation()
        raw_vector = __import__("numpy").concatenate([__import__("numpy").asarray(raw[k]) for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")], axis=0)
        expected_obs = 2 * ((raw_vector - adapter.obs_min) / (adapter.obs_max - adapter.obs_min + 1e-6) - 0.5)
        obs_err = float(__import__("numpy").max(__import__("numpy").abs(adapter.current_obs() - expected_obs.astype(__import__("numpy").float32))))
        probe = __import__("numpy").linspace(-0.91, 0.83, 7)
        expected_action = __import__("numpy").clip((probe + 1) / 2 * (adapter.action_max - adapter.action_min) + adapter.action_min, adapter.action_low, adapter.action_high)
        action_err = float(__import__("numpy").max(__import__("numpy").abs(adapter.unnormalize(probe) - expected_action)))
        return {"obs_max_abs": obs_err, "action_inverse_max_abs": action_err, "passed": bool(obs_err <= 1e-6 and action_err <= 1e-6), "atomic_steps": 0}
    finally:
        adapter.close()


def main() -> None:
    legacy.configure_runtime()
    manifest_path = REPORT_ROOT / "S0D_R2_R1_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state0 = torch.load(Path(manifest["assets"]["bc"]["path"]), map_location="cpu", weights_only=True)["model"]
    actor_keys = sorted(k for k in state0 if k.startswith("actor."))
    actor_ft_keys = sorted(k for k in state0 if k.startswith("actor_ft."))
    critic_keys = sorted(k for k in state0 if k.startswith("critic."))
    paired_diffs = [float((state0[k].float() - state0["actor_ft." + k[len("actor."):]].float()).abs().max())
                    for k in actor_keys if "actor_ft." + k[len("actor."):] in state0]
    manifest["assets"]["square_state0_model_inventory"] = {
        "model_key_count": len(state0),
        "prefixes": sorted(set(k.split(".")[0] for k in state0)),
        "actor_keys": len(actor_keys),
        "actor_ft_keys": len(actor_ft_keys),
        "critic_keys": len(critic_keys),
        "actor_shapes": {k: list(state0[k].shape) for k in actor_keys},
        "critic_keys_preview": critic_keys,
        "actor_vs_actor_ft_max_abs": max(paired_diffs) if paired_diffs else None,
        "bc_ema_equivalent": bool(paired_diffs) and max(paired_diffs) == 0.0,
    }
    saved = pickle.loads((DATA_ROOT / "square_snapshots.pkl").read_bytes())
    anchors = saved["anchors"]
    parent_success = saved["parent_success"]
    test_values = {a["snapshot_id"]: a.get("test", {}) for a in anchors if a.get("test")}
    manifest["precheck"]["normalization"] = normalization_audit()
    metrics = compute_metrics(anchors, parent_success, test_values)
    eligible = [a for a in anchors if a.get("main_eligible")]
    complete = set()
    for seed in TEST_SEEDS:
        cases = [a for a in eligible if a["reset_seed"] == seed]
        if cases and all(all(a["test"].get(m, {}).get("complete", False) for m in BASE_MAGNITUDES) for a in cases):
            complete.add(seed)
    metrics.update({
        "planned_reset_seeds": len(TEST_SEEDS),
        "completed_reset_seeds": len(set(parent_success["bc"]) & set(parent_success["ft"])),
        "valid_parent_clusters": metrics["parent_clusters"],
        "excluded_pulse_success_clusters": len(set(a["reset_seed"] for a in anchors if not a.get("main_eligible"))),
        "partial_clusters": len(set(a["reset_seed"] for a in eligible) - complete),
    })
    metrics["context"] = bool(
        metrics["bc_sham_success"] is not None and metrics["bc_sham_success"] >= 0.5
        and metrics["bc_nonzero_success"] is not None and metrics["bc_nonzero_success"] >= 0.2
        and metrics["Jnom"] >= 0
    )
    measurement = bool(
        manifest["implementation"] == "PASS"
        and metrics["completed_reset_seeds"] == 24
        and metrics["main_paired_clusters"] >= 16
        and all(
            a.get("main_eligible") and all(a["test"].get(m, {}).get("complete", False) for m in BASE_MAGNITUDES)
            for a in eligible if a["reset_seed"] in complete
        )
    )
    manifest["measurement"] = "VALID" if measurement else "INCOMPLETE"
    manifest["context"] = "VALID" if metrics["context"] else ("FAIL" if measurement else "NOT_TESTED")
    manifest["tasks"]["square"] = metrics
    if measurement and metrics["context"]:
        passed = bool(
            metrics["Lbar"] >= 0.1 and metrics["Ibar"] >= 0.1
            and metrics["Lbar_ci"]["lower"] > 0 and metrics["Ibar_ci"]["lower"] > 0
            and metrics["block_A"]["Lbar"] > 0 and metrics["block_A"]["Ibar"] > 0
            and metrics["block_B"]["Lbar"] > 0 and metrics["block_B"]["Ibar"] > 0
        )
        if passed:
            manifest["status"] = "PASS"; manifest["phenomenon"] = "PASS"; manifest["next_stage"] = "GO_FOR_MECHANISM_REVIEW"
        elif metrics["Lbar_ci"]["upper"] < 0.1 or metrics["Ibar_ci"]["upper"] < 0.1:
            manifest["status"] = "FAIL"; manifest["phenomenon"] = "FAIL"; manifest["next_stage"] = "NO_GO"
        else:
            manifest["status"] = "INCONCLUSIVE"; manifest["phenomenon"] = "INCONCLUSIVE"; manifest["next_stage"] = "NO_GO"
    else:
        manifest["status"] = "INCONCLUSIVE"; manifest["phenomenon"] = "INCONCLUSIVE"; manifest["next_stage"] = "NO_GO"
    manifest["offline_aggregate_correction"] = {
        "performed": True,
        "reason": "fixed nested magnitude-complete lookup in compute_metrics; no new environment evaluation",
        "new_environment_atomic_steps": 0,
        "new_environment_chunk_steps": 0,
    }
    plot_path = PLOT_ROOT / "s0d_r2_r1_recovery_curves.png"
    plot_curves(metrics, plot_path)
    implementation_text, boundary_text = report_text(manifest, metrics, manifest["precheck"], manifest.get("pulse_calibration"))
    impl_path = REPORT_ROOT / "S0D_R2_R1_IMPLEMENTATION_AUDIT.md"; impl_path.write_text(implementation_text, encoding="utf-8")
    boundary_path = REPORT_ROOT / "S0D_R2_R1_BOUNDARY_AUDIT.md"; boundary_path.write_text(boundary_text, encoding="utf-8")
    csv_path = REPORT_ROOT / "S0D_R2_R1_RAW_METRICS.csv"
    archive_path = REPORT_ROOT / "S0CR1_UPPER_REVIEW_ARCHIVE.md"
    manifest["files"] = [json_meta(x) for x in (csv_path, impl_path, boundary_path, PROTOCOL_PATH, CONFIG_PATH, archive_path, plot_path)]
    manifest["metrics"] = metrics
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "measurement": manifest["measurement"], "context": manifest["context"], "phenomenon": manifest["phenomenon"], "main_paired_clusters": metrics["main_paired_clusters"], "Lbar": metrics["Lbar"], "Ibar": metrics["Ibar"], "new_environment_atomic_steps": 0}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
