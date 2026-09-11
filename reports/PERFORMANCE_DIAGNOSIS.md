# DPPO 性能诊断、等价优化 A/B 与 Hopper smoke 报告

日期：2026-09-11  
项目：D:\Desktop\my_project\paper_reproduction\DPPO  
官方源码：source/dppo_v0.6，commit dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc，tag v0.6

## 1. 结论摘要

本轮没有启动完整 1000-iteration 训练、5-seed 批量、Figure 18、ROBOMIMIC 或预训练任务。

原始官方实现的训练瓶颈定位为 PPODiffusion.loss() 中对 CUDA denoising_inds 逐元素 Python 遍历。将这一处改为数学等价的 CUDA 向量化计算后，在保持 Hopper、seed=42、n_envs=40、n_steps=500、K'=10、batch=50000、update_epochs=5、gamma=0.99、GAE lambda=0.95、PPO clip=0.01、同一 checkpoint 不变的条件下：

- 完整训练 iteration 中位数：92.513 s → 14.551 s，约 6.36× 加速。
- PPO update 中位数：78.365 s → 1.313 s，约 59.68× 加速。
- patched iteration 1–3 总时间：15.317、14.551、14.332 s。
- discount 独立 microbenchmark：官方 3.472 s → 向量化 0.000202 s，约 17,208×；最大绝对/相对误差均为 0。

该 patch 已通过根仓库的 `reports/PERFORMANCE_PATCH.diff` 保存，并由
`scripts/setup/apply_performance_patch.ps1` / `.sh` 提供从项目根目录执行的幂等应用方式。
官方源码仍固定为 v0.6 commit；本地额外应用一个数学等价的 CUDA 向量化性能 patch；
patch 不改变 DPPO 数学定义和论文超参数。

## 2. 原始时间构成

计时 instrumentation 使用 CUDA synchronize，分为 rollout/environment sampling、old value/logprob/GAE 准备和 PPO update；总时间还包括 scheduler、checkpoint 和日志收尾。

| run | itr | rollout (s) | prepare (s) | PPO update (s) | total (s) |
|---|---:|---:|---:|---:|---:|
| official baseline | 1 | 13.755 | 0.392 | 78.365 | 92.513 |
| official baseline | 2 | 13.770 | 0.190 | 78.939 | 92.900 |
| official baseline | 3 | 13.581 | 0.196 | 73.520 | 87.327 |
| vectorized patch | 1 | 13.331 | 0.323 | 1.662 | 15.317 |
| vectorized patch | 2 | 13.057 | 0.179 | 1.313 | 14.551 |
| vectorized patch | 3 | 12.854 | 0.173 | 1.280 | 14.332 |

历史 5-iteration benchmark 的 91–93 s/iteration 与本轮 baseline 一致；baseline 的主要消耗在 PPO update，rollout 约 13–14 s，准备阶段小于 0.4 s。

Baseline 日志目录：  
D:\Desktop\my_project\paper_reproduction\DPPO\logs\gym-finetune\hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10\2026-09-11_11-21-53_42

Patched A/B 日志目录：  
D:\Desktop\my_project\paper_reproduction\DPPO\logs\gym-finetune\hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10\2026-09-11_11-27-25_42

Patched telemetry short 日志目录：  
D:\Desktop\my_project\paper_reproduction\DPPO\logs\gym-finetune\hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10\2026-09-11_11-31-05_42

另有一个由 PowerShell 帮助语法检查意外生成的初始化现场：
D:\Desktop\my_project\paper_reproduction\DPPO\logs\gym-finetune\hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10\2026-09-11_11-47-27_42。该目录只有 5 行模型初始化日志，没有 rollout、iteration 或 checkpoint，已保留用于审计，不计入性能 A/B。

## 3. GPU/CPU 运行观察

确认配置和设备为：device=cuda:0、NVIDIA GeForce RTX 5060 Ti、PyTorch 2.7.1+cu128、CUDA runtime 12.8。

Telemetry 文件：  
D:\Desktop\my_project\paper_reproduction\DPPO\reports\telemetry_patched_short.csv

共 51 个样本：GPU utilization 平均 12.8%、最高 93%；显存 1877–2993 MiB / 16311 MiB；SM clock 2617–2857 MHz；power 22.33–142.04 W；temperature 42–55 °C；WSL host CPU utilization 平均 5.86%、最高 10.10%。没有显示 GPU clock/power/显存或 CPU oversubscription 异常，支持首要瓶颈是官方实现把 CUDA 标量逐个带回 Python 计算并造成大量同步等待。

Telemetry 采集脚本：scripts/diagnostics/monitor_telemetry.py  
启动脚本：scripts/diagnostics/run_telemetry_short.sh

## 4. 等价优化与数值验证

官方 v0.6 写法位于 source/dppo_v0.6/model/diffusion/diffusion_ppo.py，核心是：
~~~python
discount = torch.tensor(
    [
        self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
        for i in denoising_inds
    ]
).to(self.device)
~~~

本轮 patch 使用同一个 gamma、同一个 CUDA denoising index、同一个 device，并使用 advantage 的 dtype：
~~~python
exponent = self.ft_denoising_steps - denoising_inds - 1
discount = torch.pow(
    torch.tensor(
        self.gamma_denoising,
        device=advantages.device,
        dtype=advantages.dtype,
    ),
    exponent.to(dtype=advantages.dtype),
)
~~~

测试参数为 batch=50000、ft_denoising_steps=10、gamma=0.99、CUDA、FP32；两种输出均为 torch.float32、cuda:0，max_abs_error=0，max_rel_error=0，且全部 finite。

独立检查确认 resolved quantile 默认 lower=0、upper=1；quantile(0/1)+clamp 是严格 no-op，最大绝对误差 0，中位数成本约 0.000522 s，因此没有把它混入源代码 patch。

Microbenchmark JSON：  
D:\Desktop\my_project\paper_reproduction\DPPO\reports\denoising_discount_microbenchmark.json

源码 patch diff：  
D:\Desktop\my_project\paper_reproduction\DPPO\reports\PERFORMANCE_PATCH.diff

该 diff 包含 discount 向量化和仅用于 A/B 计时的 instrumentation；没有 AMP、TF32、torch.compile、网络结构、batch、环境数量、K/K′、PPO clip、reward 或动力学修改。

## 5. Hopper IDQL/DIPO 极短 smoke

两项均使用官方 Hopper pretrained checkpoint：  
D:\Desktop\my_project\paper_reproduction\DPPO\checkpoints\official\hopper_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt

使用入口 scripts/train/run_fig4_seed42.ps1；smoke 覆盖为 seed=42、n_train_itr=2、n_envs=2、n_steps=4、batch_size=8、replay_ratio=1、n_critic_warmup_itr=0。覆盖只用于在几分钟内触发一次真实 update，不代表论文配置或科研结果；短 rollout 没有完整 episode，reward=0 是预期现象。

### IDQL

状态：PASS。环境创建、checkpoint 加载、Diffusion actor、critic/Q、rollout、finite actor loss、checkpoint 保存成功。state_0→state_1 中 64 个浮点 tensor 发生变化，最大参数绝对变化 0.0010000020，确认至少一次 optimizer update。

日志目录：  
D:\Desktop\my_project\paper_reproduction\DPPO\logs\gym-finetune\hopper-medium-v2_idql_diffusion_mlp_ta4_td20\2026-09-11_11-32-33_42

### DIPO

状态：PASS。环境创建、checkpoint 加载、Diffusion actor、critic、rollout、finite actor/critic loss、checkpoint 保存成功。state_0→state_1 中 68 个浮点 tensor 发生变化，最大参数绝对变化 0.0010000020，确认至少一次 optimizer update。

日志目录：  
D:\Desktop\my_project\paper_reproduction\DPPO\logs\gym-finetune\hopper-medium-v2_dipo_diffusion_mlp_ta4_td20\2026-09-11_11-33-06_42

smoke checkpoint 审计 JSON：  
D:\Desktop\my_project\paper_reproduction\DPPO\reports\smoke_artifact_audit.json

## 6. 本地资源核查

本地核查 JSON：  
D:\Desktop\my_project\paper_reproduction\DPPO\reports\final_local_check.json

结果：cuda_available=true；hopper-medium-v2、walker2d-medium-v2、halfcheetah-medium-v2 均 reset/step 成功，观测和 reward 均 finite；三个数据集、normalization 文件和三个官方 checkpoint 均存在且非空。

完整字节数、SHA256 和递归目录大小：  
- D:\Desktop\my_project\paper_reproduction\DPPO\reports\RESOURCE_MANIFEST.md
- D:\Desktop\my_project\paper_reproduction\DPPO\reports\RESOURCE_MANIFEST.csv
- 本轮新增资源的增量清单：D:\Desktop\my_project\paper_reproduction\DPPO\reports\RESOURCE_MANIFEST_DELTA.md 和 RESOURCE_MANIFEST_DELTA.csv

当前主要资源约为：.runtime 6.71 GiB、.cache 3.79 GiB、data 233.86 MiB、logs 79.01 MiB、reports 261.29 KiB、checkpoints 12.95 MiB；.runtime/.cache 的递归精确值和新增资源明细分别见主 manifest 与 delta manifest。

Python 环境：  
D:\Desktop\my_project\paper_reproduction\DPPO\.runtime\venv

MuJoCo 2.1.0：  
D:\Desktop\my_project\paper_reproduction\DPPO\.runtime\mujoco210

## 7. 本轮产生的诊断脚本和报告

- scripts/diagnostics/benchmark_loss_micro.py
- scripts/diagnostics/monitor_telemetry.py
- scripts/diagnostics/run_telemetry_short.sh
- scripts/diagnostics/audit_smoke_artifacts.py
- scripts/diagnostics/final_local_check.py
- reports/denoising_discount_microbenchmark.json
- reports/telemetry_patched_short.csv
- reports/smoke_artifact_audit.json
- reports/final_local_check.json
- reports/PERFORMANCE_PATCH.diff
- reports/PERFORMANCE_DIAGNOSIS.md

run_fig4_seed42.ps1 增加了可选 smoke 覆盖参数；不传这些参数时默认科学配置不变。并修正了原有 D:→WSL /mnt/d 路径转换及 hopper-v2 checkpoint 文件名前缀问题。

## 8. Git 状态与边界

本报告记录的上一轮实验没有 commit、push、merge、force reset 或删除已有 data/checkpoint/log。

- 根仓库的性能证据和 patch 脚本由本轮 P1.5 独立 commit 保存；
- source/dppo_v0.6 仍指向官方 v0.6 SHA，但性能 patch 和 instrumentation 造成未提交 dirty submodule。
- 新 A/B、smoke 和诊断产物按 release policy 选择小型日志/配置上传；大资源仍仅保留本地。
- GitHub 上已有导师说明文件未被本轮改写。

正式 1000-iteration Figure 4、五 seed、最终 Figure 4/18 复现图仍未开始；本报告不把 smoke reward 或 A/B 运行结果称为论文最终科研结果。

## 9. 上级综控 AI 所需紧凑状态

~~~text
PERFORMANCE: PASS
原始时间: 训练 iteration 中位数 92.513 s；PPO update 中位数 78.365 s
优化后时间: 训练 iteration 中位数 14.551 s；PPO update 中位数 1.313 s
加速倍率: 总 iteration 6.36x；PPO update 59.68x
主要瓶颈: PPODiffusion.loss() 中官方 Python 逐元素遍历 CUDA denoising_inds 导致同步
修改文件: source/dppo_v0.6/model/diffusion/diffusion_ppo.py；source/dppo_v0.6/agent/finetune/train_ppo_diffusion_agent.py（计时）
科学配置是否改变: NO（仅 n_train_itr 临时缩短；算法超参数未改）

LOCAL CHECK: PASS

IDQL SMOKE: PASS
DIPO SMOKE: PASS

git status: 未提交；根仓库修改、未跟踪 smoke/诊断文件、官方子模块 dirty
日志/报告路径: 见本文件第 2、5、6、7 节

建议下一步: 审核 P1.5 GitHub 归档后，再查看 Hopper 三算法正式配置短 benchmark；不要把 smoke 或短 benchmark 曲线作为最终论文结果。
~~~
