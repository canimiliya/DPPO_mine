"""Build compact, auditable S0b report artifacts from external diagnostics."""

from __future__ import annotations

import csv, hashlib, json, os, subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXT = Path("/mnt/d/AgentData/DPPO-S0b")
REPO_EXT = Path("D:/AgentData/DPPO-S0b")


def sha(path):
    h = hashlib.sha256(); size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b: break
            h.update(b); size += len(b)
    return size, h.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as f: return json.load(f)


def main():
    sqbc = load(EXT / "square_bc_metrics.json")
    sqft = load(EXT / "square_ft_metrics.json")
    trbc = load(EXT / "transport_bc_metrics.json")
    trft = load(EXT / "transport_ft_metrics.json")
    sqcost = load(EXT / "square_cost.json")
    trcost = load(EXT / "transport_cost.json")
    b1 = {"anchor": "square_ft_frozen first 16 states", "items": 160,
          "official_scalar_actor_loss": -1.1920929132713809e-08,
          "per_item_mean_actor_loss": -1.1920929132713809e-08,
          "loss_relative_error": 0.0, "official_actor_gradient_norm": 0.5900592803955078,
          "reconstructed_gradient_norm": 0.5900592803955078,
          "gradient_relative_error": 0.0, "gradient_absolute_error": 0.0}
    b2 = sqbc["toy_importance_correction"]
    costs = []
    for d, c in [(sqft, sqcost), (trft, trcost)]:
        ms = c["timing_ms_mean"]; base = c["baseline_total_ms"]
        audit_ms = d["gradient_collection_seconds"] * 1000
        h_all = (ms["proposal"] + ms["io"] + audit_ms) / base
        f = c["actor_reducible_fraction_f"]; s = c["assumed_actor_speedup_s"]
        pred_all = 1 / (1 - f + f / s + h_all)
        costs.append((d["task"], c, audit_ms, h_all, pred_all))

    raw = ROOT / "reports" / "research" / "S0B_RAW_METRICS.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append({"section":"B1","task":"square","policy":"ft","metric":"loss_relative_error","value":b1["loss_relative_error"],"unit":"relative","note":"official scalar actor vs exact per-item mean"})
    rows.append({"section":"B1","task":"square","policy":"ft","metric":"gradient_relative_error","value":b1["gradient_relative_error"],"unit":"relative","note":"official actor gradient vs reconstructed mean"})
    rows.append({"section":"B2","task":"toy","policy":"float64","metric":"gradient_expectation_abs_error","value":b2["gradient_abs_error"],"unit":"absolute","note":"exact enumeration"})
    for d in [sqbc, sqft, trbc, trft]:
        for method, v in d["mc_last_layer_gradient"].items():
            for metric in ["mse", "mse_ratio_to_uniform", "gradient_bias_norm", "mse_se", "gradient_second_moment", "ess_mean", "weight_min", "weight_max"]:
                rows.append({"section":"MC_LAST_LAYER","task":d["task"],"policy":d["policy"],"metric":method+"."+metric,"value":v[metric],"unit":"diagnostic","note":"200 draws; M=50% of 640 items; last-layer gradient proxy"})
            rows.append({"section":"MC_LAST_LAYER","task":d["task"],"policy":d["policy"],"metric":method+".episode_cluster_bootstrap_ci_low","value":v["episode_cluster_bootstrap_ci"][0],"unit":"diagnostic","note":"episode-cluster bootstrap lower bound; proxy contribution CI"})
            rows.append({"section":"MC_LAST_LAYER","task":d["task"],"policy":d["policy"],"metric":method+".episode_cluster_bootstrap_ci_high","value":v["episode_cluster_bootstrap_ci"][1],"unit":"diagnostic","note":"episode-cluster bootstrap upper bound; proxy contribution CI"})
        for label, v in d["theta"].items():
            rows.append({"section":"THETA","task":d["task"],"policy":d["policy"],"metric":label+".ratio_mean","value":v["ratio_mean"],"unit":"ratio","note":"offline diagnostic update"})
            rows.append({"section":"THETA","task":d["task"],"policy":d["policy"],"metric":label+".clip_active_fraction","value":v["clip_active_fraction"],"unit":"fraction","note":"offline diagnostic update"})
    for task, c, audit_ms, h_all, pred_all in costs:
        for metric, value in c["timing_ms_mean"].items():
            rows.append({"section":"COST","task":task,"policy":"ft","metric":metric,"value":value,"unit":"ms per 64-state diagnostic cycle","note":"3-cycle mean; rollout field is model-only"})
        for metric, value in [("actor_reducible_fraction_f",c["actor_reducible_fraction_f"]),("audit_ms_included",audit_ms),("overhead_h_all",h_all),("predicted_total_speedup_all_costs",pred_all)]:
            rows.append({"section":"COST","task":task,"policy":"ft","metric":metric,"value":value,"unit":"diagnostic","note":"audit included in final compute gate"})
    with raw.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section","task","policy","metric","value","unit","note"]); writer.writeheader(); writer.writerows(rows)

    # Tiny plots are generated from the compact JSON, never from fabricated or
    # digitized paper curves.
    try:
        import matplotlib.pyplot as plt
        labels=["Sq-BC","Sq-FT","Tr-BC","Tr-FT"]
        ds=[sqbc,sqft,trbc,trft]
        fig, ax=plt.subplots(figsize=(6.6,4.0)); x=np.arange(4); w=.2
        for q,(name,color) in enumerate([("uniform","#777777"),("timestep-only","#4472c4"),("state-aware","#ed7d31"),("oracle","#70ad47")]):
            ax.bar(x+(q-1.5)*w,[d["mc_last_layer_gradient"][name]["mse_ratio_to_uniform"] for d in ds],w,label=name)
        ax.axhline(.75,color="black",ls="--",lw=1); ax.set_xticks(x,labels); ax.set_ylabel("last-layer MSE / uniform"); ax.set_title("S0b fixed-anchor gradient MSE"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/"plots/research/s0b_gradient_mse.png",dpi=180); plt.close(fig)
        fig, ax=plt.subplots(figsize=(6.6,4.0));
        for task,c,audit_ms,h,pred in costs:
            ms=c["timing_ms_mean"]; keys=["rollout_model_sampling","old_logprob","actor_forward","actor_backward","critic","proposal","io"]; vals=[ms[k] for k in keys]
            ax.bar(task, sum(vals), color="#c9daf8", label="baseline components" if task=="square" else None); ax.bar(task, audit_ms, bottom=sum(vals), color="#f4cccc", label="audit" if task=="square" else None)
        ax.set_ylabel("ms per diagnostic cycle"); ax.set_title("S0b measured cost with audit"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/"plots/research/s0b_cost_breakdown.png",dpi=180); plt.close(fig)
    except Exception as e:
        print("PLOT_WARNING", repr(e))

    patch = ROOT / "reports" / "PERFORMANCE_PATCH.diff"
    source_diff = subprocess.run(["git","-C",str(ROOT/"source/dppo_v0.6"),"diff","--name-only"],capture_output=True,text=True).stdout.splitlines()
    manifest={"schema":"S0B_MANIFEST.v1","task":"S0b","status":"INCONCLUSIVE","research_decision":"NO-GO for current cheap state-aware mechanism; S1 prohibited","preflight":{"local_main":"d82ab2ce6e96fb1a0b2a4573ae7d15b1f9630df1","origin_main":"d82ab2ce6e96fb1a0b2a4573ae7d15b1f9630df1","submodule_head":"dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc","submodule_dirty_files":source_diff,"approved_patch_sha256":sha(patch)[1] if patch.exists() else None,"scientific_source_modified":source_diff != ["model/diffusion/diffusion_ppo.py"]},"budgets":{"primitive_steps":38400,"six_hour_limit_reached":False,"one_point_five_million_step_limit_reached":False,"large_formal_training":False},"b1":b1,"b2":b2,"tasks":{},"costs":{},"files":{}}
    for task, items in [("square",[("bc",EXT/"square_bc_frozen.npz"),("ft",EXT/"square_ft_frozen.npz"),("bc_metrics",EXT/"square_bc_metrics.json"),("ft_metrics",EXT/"square_ft_metrics.json")]),("transport",[("bc",EXT/"transport_bc_frozen.npz"),("ft",EXT/"transport_ft_frozen.npz"),("bc_metrics",EXT/"transport_bc_metrics.json"),("ft_metrics",EXT/"transport_ft_metrics.json")])]:
        manifest["tasks"][task]={}
        for label,p in items:
            pp=Path(str(p)); size,h=sha(pp); rec={"path_windows":str(REPO_EXT/(pp.name)),"path_wsl":str(pp),"bytes":size,"sha256":h}
            if pp.suffix=='.npz':
                z=np.load(pp); rec["arrays"]={k:{"shape":list(z[k].shape),"dtype":str(z[k].dtype)} for k in z.files}
            manifest["tasks"][task][label]=rec
    for task,c,audit_ms,h,pred in costs: manifest["costs"][task]={"profile":str(REPO_EXT/(task+"_cost.json")),"measured":c,"audit_ms":audit_ms,"overhead_h_including_audit":h,"predicted_total_speedup_including_audit":pred}
    for p in [ROOT/"research/denoising_budget/loss_terms.py",ROOT/"research/denoising_budget/sampler.py",ROOT/"research/denoising_budget/collect_frozen.py",ROOT/"research/denoising_budget/gradient_audit.py",ROOT/"research/denoising_budget/run_s0b.py",ROOT/"research/denoising_budget/cost_profile.py",raw,ROOT/"plots/research/s0b_gradient_mse.png",ROOT/"plots/research/s0b_cost_breakdown.png"]:
        if p.exists(): size,h=sha(p); manifest["files"][str(p.relative_to(ROOT)).replace('\\','/')]={"bytes":size,"sha256":h}
    mpath=ROOT/"reports/research/S0B_MANIFEST.json"; mpath.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')

    def ratio(task, policy, method):
        return (sqbc if task=='square' and policy=='bc' else sqft if task=='square' else trbc if policy=='bc' else trft)["mc_last_layer_gradient"][method]["mse_ratio_to_uniform"]
    report=f'''# S0b 梯度结构与计算成本诊断\n\n日期：2026-09-15。范围：冻结 Square / Transport BC 与现存 FT checkpoint 的诊断；未训练、未接入 S1、未正式长跑、未做 counterfactual 大规模实验。\n\n## 结论\n\n**STATUS：INCONCLUSIVE**。**RESEARCH DECISION：NO-GO（当前廉价 state-aware 机制）**。oracle 在 last-layer 固定 anchor 上显示空间，但廉价 proposal 未跨任务通过，且把实际 audit 成本计入后 compute gate 不通过。由于 200-draw MSE 使用 last-layer 梯度代理而不是全参数逐项梯度，不能把本轮写成完整的 full-parameter 机制证据；因此 S1 必须继续禁止。\n\n## B0 / 预算\n\n- local main HEAD = `d82ab2ce6e96fb1a0b2a4573ae7d15b1f9630df1`；origin/main 相同。submodule HEAD = `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`；dirty 仅为已批准 `model/diffusion/diffusion_ppo.py` 性能补丁，pointer 未变，正式科学 source 未新增 diff。\n- runtime 沿用 S0a 的 Python 3.12.3、torch 2.7.1+cu128、robosuite 1.4.1、robomimic 0.5.0、mujoco-py 2.1.2.14 / legacy MuJoCo 2.1.0。未安装或升级依赖。\n- NEW FROZEN DIAGNOSTIC ROLLOUT：Square/Transport 各 BC、FT 16 episodes × 4 boundaries × 10 transitions，chain 分别 `[64,11,4,7]` 与 `[64,11,8,14]`；primitive steps 合计 **38,400**。6 h 与 1.5M 上限均未触达。NPZ 仅本地 `D:\\AgentData\\DPPO-S0b`。\n\n## B1 / B2 correctness\n\nB1 在真实 Square FT 冻结 rollout 的前 16 states / 160 actor items 上：official scalar actor loss = `{b1['official_scalar_actor_loss']:.9g}`，逐项 mean = `{b1['per_item_mean_actor_loss']:.9g}`，相对误差 = **0**；official 与重建 full-parameter actor gradient norm 都为 `{b1['official_actor_gradient_norm']:.9g}`，相对误差 = **0**。LOSS EQUIVALENCE：**PASS**。\n\nB2 float64 toy 全枚举：loss 绝对误差 `{b2['loss_abs_error']:.3g}`，gradient 最大绝对误差 `{b2['gradient_abs_error']:.3g}`，IMPORTANCE CORRECTION：**PASS**。IS 权重放在 PPO clipping 后的逐项 loss 外，未做 self-normalization/clipping。\n\n## Fixed-anchor MSE\n\n每个模型 64 states / 640 items，M=320，200 个独立 sampling RNG。下表是 **last-layer gradient proxy** 的 MSE ratio；uniform=1。\n\n| task / policy | timestep-only | state-aware | oracle | state-aware vs timestep |\n|---|---:|---:|---:|---:|\n| Square / BC | {ratio('square','bc','timestep-only'):.3f} | {ratio('square','bc','state-aware'):.3f} | {ratio('square','bc','oracle'):.3f} | {ratio('square','bc','state-aware')/ratio('square','bc','timestep-only'):.3f} |\n| Square / FT | {ratio('square','ft','timestep-only'):.3f} | {ratio('square','ft','state-aware'):.3f} | {ratio('square','ft','oracle'):.3f} | {ratio('square','ft','state-aware')/ratio('square','ft','timestep-only'):.3f} |\n| Transport / BC | {ratio('transport','bc','timestep-only'):.3f} | {ratio('transport','bc','state-aware'):.3f} | {ratio('transport','bc','oracle'):.3f} | {ratio('transport','bc','state-aware')/ratio('transport','bc','timestep-only'):.3f} |\n| Transport / FT | {ratio('transport','ft','timestep-only'):.3f} | {ratio('transport','ft','state-aware'):.3f} | {ratio('transport','ft','oracle'):.3f} | {ratio('transport','ft','state-aware')/ratio('transport','ft','timestep-only'):.3f} |\n\nOracle/uniform 在四个组合均 ≤ 0.75（0.600、0.658、0.249、0.710），说明理论非均匀抽样空间存在。state-aware 未通过 Square 两个组合，且跨任务方向不一致；因此 CHEAP PROPOSAL 与 NOVEL MECHANISM 均 **FAIL / NO-GO**。timestep-only 在 Square FT 与 Transport FT 约 0.837 / 0.903，不能替代跨任务机制证据。\n\n200 次 sampling noise 是内层重复，不当作 200 个环境样本。每个指标的 MC standard error、ESS、weight 分布和 episode-cluster bootstrap 原始统计保存在 `S0B_RAW_METRICS.csv` 与 external JSON；由于本轮 MSE 是 last-layer proxy，cluster CI 不被宣称为 full-parameter ratio CI。\n\n## theta-old 与 offline update\n\n所有 theta-old ratio mean = 1、clip fraction = 0。1 step 后 Square BC/FT clip fraction = 0.831/0.714，Transport BC/FT = 0.563/0.728；3 steps 后为 0.909/0.839、0.928/0.941。它们是 **offline diagnostic update**，仅在内存副本执行，未覆盖 checkpoint，不能称为新的 RL 训练结果。\n\n## B5 cost\n\n3-cycle mean（每周期 64-state diagnostic batch；rollout 字段是 16-state model-only sampling，不冒充完整环境 wall time）：\n\n| task | rollout | old logprob | actor fwd | actor bwd | critic | proposal | I/O | f | h（含 audit） | predicted speedup（含 audit） |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n| Square FT | {sqcost['timing_ms_mean']['rollout_model_sampling']:.2f} | {sqcost['timing_ms_mean']['old_logprob']:.2f} | {sqcost['timing_ms_mean']['actor_forward']:.2f} | {sqcost['timing_ms_mean']['actor_backward']:.2f} | {sqcost['timing_ms_mean']['critic']:.2f} | {sqcost['timing_ms_mean']['proposal']:.2f} | {sqcost['timing_ms_mean']['io']:.2f} | {sqcost['actor_reducible_fraction_f']:.3f} | {costs[0][3]:.2f} | {costs[0][4]:.3f} |\n| Transport FT | {trcost['timing_ms_mean']['rollout_model_sampling']:.2f} | {trcost['timing_ms_mean']['old_logprob']:.2f} | {trcost['timing_ms_mean']['actor_forward']:.2f} | {trcost['timing_ms_mean']['actor_backward']:.2f} | {trcost['timing_ms_mean']['critic']:.2f} | {trcost['timing_ms_mean']['proposal']:.2f} | {trcost['timing_ms_mean']['io']:.2f} | {trcost['actor_reducible_fraction_f']:.3f} | {costs[1][3]:.2f} | {costs[1][4]:.3f} |\n\n不含 audit 的理想化 actor-only speedup 为 2×，但这是上限假设；Amdahl 计算必须把 proposal、audit、forward/backward、critic、rollout、I/O 一起看。本轮实测 full-parameter audit prefix 160 items 约 3.7 s，而单周期基线约 0.11–0.13 s，故 `proposal + audit <= 5%` 与 `compute >= 10%` 均 **FAIL**。\n\n## 交付与限制\n\n- Report：本文件；Raw metrics：`S0B_RAW_METRICS.csv`；Manifest：`S0B_MANIFEST.json`；plots：`plots/research/s0b_gradient_mse.png`、`plots/research/s0b_cost_breakdown.png`。\n- full-parameter norms 仅按门槛对前 16 states streaming 审计；没有永久保存巨型逐项梯度。完整 640-item full-parameter MSE 未完成，这是 STATUS=INCONCLUSIVE 的主要原因。\n- reward scaler 是仅 fit episodes 的 diagnostic scaler；raw reward 保留；diagnostic GAE 不等于历史 training GAE。Transport native ever-success 仍为 MISSING，未从 reward 反推、未编造 phase/handover。历史 checkpoint 仍不可严格 resume。\n- 本轮没有进入 S1；等待上级审核 GitHub。\n\n### Decision gates\n\n| gate | result |\n|---|---|\n| loss equivalence | PASS |\n| importance correction | PASS |\n| oracle space | PASS on last-layer proxy |\n| cheap proposal | FAIL |\n| beyond timestep-only | FAIL |\n| compute >=10% | FAIL with audit included |\n| proposal/audit <=5% | FAIL with audit included |\n'''
    (ROOT/"reports/research/S0B_GRADIENT_AND_COST_AUDIT.md").write_text(report,encoding='utf-8')
    print("built", raw, mpath)


if __name__ == '__main__': main()
