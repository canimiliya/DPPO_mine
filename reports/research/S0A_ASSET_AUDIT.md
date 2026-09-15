# S0a 资产与语义审计

日期：2026-09-15
范围：DPPO ROBOMIMIC state-observation Transport / Square 资产、代码语义和运行时审计。
结论：**PASS（S0a）**。
S0b 状态：**未获授权，不进入 S0b**。

## 1. 基线与执行边界

审计开始前已执行 `git fetch origin`。基线完全一致：

| 项目 | SHA |
|---|---|
| local `main HEAD` | `4342255866b056cef97277196450204f5ebfacc0` |
| `origin/main` | `4342255866b056cef97277196450204f5ebfacc0` |
| 当前 `HEAD` | `4342255866b056cef97277196450204f5ebfacc0` |

没有 checkout、reset、merge 或 rebase。父仓库工作树在审计前已有两类用户状态：

- `source/dppo_v0.6` 保持官方 submodule pointer `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`，但工作树 dirty；差异仅为既有 `model/diffusion/diffusion_ppo.py` denoising-discount 向量化补丁，未改 submodule pointer。
- 两个未跟踪 Gym 日志目录仍保留，未被本次提交触碰：`logs/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/`、`logs/gym-finetune/walker2d-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/`。

本轮只读取代码、配置、报告、运行时包和 checkpoint；inventory 使用 `torch.load(..., weights_only=True)`、NumPy 归档读取和源码反射，没有创建环境、reset/step、训练、正式长跑或 counterfactual 实验。

逐文件字节数、SHA256、checkpoint key/shape、normalization 数组和运行时记录见 [S0A_ASSET_MANIFEST.json](S0A_ASSET_MANIFEST.json)。可复核脚本为 [inventory.py](../../research/denoising_budget/inventory.py)。

## 2. 代码与运行时

### 2.1 指定代码审计

已实际读取用户指定的六个文件：

| 文件 | 审计结论 |
|---|---|
| `source/dppo_v0.6/model/diffusion/diffusion_vpg.py` | `ft_denoising_steps` 控制 fine-tuning 链；采样 std 和 log-prob std 分别受 `min_sampling_denoising_std` / `min_logprob_denoising_std` 下限约束；当前默认值为 `0.1`。 |
| `source/dppo_v0.6/model/diffusion/diffusion_ppo.py` | `gamma_denoising` 用于 denoising-step advantage discount；当前 dirty 差异只是等价向量化，未改变配置口径。 |
| `source/dppo_v0.6/agent/finetune/train_ppo_diffusion_agent.py` | 训练 action 取 `output[:, :act_steps]`；正式指标按 episode 内 `max(reward_traj) / act_steps` 计算。 |
| `source/dppo_v0.6/agent/finetune/train_agent.py` | `save_model()` 只写 `itr` 和 `model.state_dict()`；没有 optimizer、scheduler、RNG 或 reward scaler。 |
| `source/dppo_v0.6/env/gym_utils/wrapper/robomimic_lowdim.py` | state 由配置的 low-dim keys 拼接；normalization 使用 `obs_min/max`，动作使用 `action_min/max` 反归一化。 |
| `source/dppo_v0.6/env/gym_utils/wrapper/multi_step.py` | `n_action_steps` 个 primitive actions 聚合为一个高层 transition；reward 默认 sum，正式配置启用 episode horizon 和 `reset_within_step=true`。 |

六个文件的精确 hash 在 manifest 的 `source_files` 中；本次没有修改它们。

### 2.2 实际运行时

| 组件 | observed |
|---|---|
| Python | 3.12.3；`/mnt/d/Desktop/my_project/paper_reproduction/DPPO/.runtime/venv/bin/python` |
| PyTorch | 2.7.1+cu128；项目 venv 的 `site-packages/torch/__init__.py` |
| CUDA | torch CUDA 12.8，`cuda_available=True`，NVIDIA GeForce RTX 5060 Ti |
| Gym | 0.22.0；项目 venv 的 `site-packages/gym/__init__.py` |
| robomimic | 0.5.0；项目 venv 的 `site-packages/robomimic/__init__.py` |
| robosuite | 1.4.1；项目 venv 的 `site-packages/robosuite/__init__.py` |
| mujoco-py | 2.1.2.14；实际加载 `mujoco_py/__init__.py` 和 `generated/cymj_2.1.2.14_312_linuxcpuextensionbuilder_312.so` |
| MuJoCo backend | legacy MuJoCo 2.1.0，`MUJOCO_PY_MUJOCO_PATH=/mnt/d/Desktop/my_project/paper_reproduction/DPPO/.runtime/mujoco210`；DPPO/robomimic 路径实际导入 `mujoco_py` |
| 另装的 `mujoco` Python 包 | 3.1.6；只作为 robosuite 包兼容依赖存在，未作为本 DPPO/robomimic backend 使用 |

审计时使用了 `MUJOCO_GL=egl` 以读取安装路径和模块；正式 Transport/Square 报告记录的 headless backend 仍是 legacy `mujoco-py` / MuJoCo 2.1.0。没有套用另一项目的 MuJoCo 3 结论。

### 2.3 TwoArmTransport 实际源码

实际安装源码：

`/mnt/d/Desktop/my_project/paper_reproduction/DPPO/.runtime/venv/lib/python3.12/site-packages/robosuite/environments/manipulation/two_arm_transport.py`

SHA256：`942beebb6e22de796f642b3b64a24df0e292448b9d01cbc30870ae64c73d7586`。

实际 `_load_model()` 创建并交给 `TransportGroup` 的对象为：

- payload：`HammerObject(name="payload")`；
- trash：`BoxObject(name="trash", size=[0.02, 0.02, 0.02])`；
- lid：`Lid(name="transport_start_bin_lid")`；同时存在 start/target/trash 三个 bin。

`TransportGroup` 实际源码为 `.../robosuite/models/objects/group/transport.py`，SHA256 为 `13efa7bddde2ac597f196d3c3cd0976999082b7668f96d457b47574f0119c28e`。

原生 success 函数的实际语义是：

```text
payload_in_target_bin AND trash_in_trash_bin
```

两项都由对应物体与目标 bin base 的接触判断得到。没有根据“运输任务通常有交接”生成阶段标签。

## 3. 模型与 normalization inventory

### 3.1 Transport

正式报告对应本机目录：

`D:\AgentData\DPPO-P5-B\formal_run_20260914_110100\robomimic-finetune\transport_ft_diffusion_mlp_ta8_td20_tdf10\2026-09-14_10-59-02_42\checkpoint\`

报告记录的三个模型均实际存在；它们未上传 GitHub：

| itr | bytes | SHA256 | state_dict 前缀 | critic |
|---:|---:|---|---|---|
| 0 | 20,019,779 | `48f6c0bf65fe12f000087b25bf5e336db03a669687dd5cf6363abf80b3d112ae` | `network.` 12、`actor.` 12、`actor_ft.` 12 | `critic.Q1.layers.*`，8 keys |
| 100 | 20,020,431 | `8fd07e05481a36eeb26e13e0cff210dcd2a9c72331911bb9a1bc736ad1e29ad1` | 同上 | 同上 |
| 200 | 20,020,431 | `b4b828312915442ede54a672f74a6276fe407ea3cf3b3830ba20bcb734d8d592` | 同上 | 同上 |

Transport actor 形状为 time embedding `[64,32]`、`[32,64]`，MLP `[1024,203]`、两个 `[1024,1024]` residual layers、输出 `[112,1024]`；`112 = 8 × 14`。critic 首层为 `[256,59]`，输出为 `[1,256]`。

BC checkpoint：

`D:\Desktop\my_project\paper_reproduction\DPPO\logs\robomimic-pretrain\transport\transport_pre_diffusion_mlp_ta8_td20\2024-07-08_11-18-59\checkpoint\state_8000.pt`

bytes `19,425,707`，SHA256 `141e0178b2b31d4c0fce505f0e62ab3e59efe7e924ff3d492a9f59ea2afb7485`；top-level keys 是 `epoch/model/ema`。`state_0` 的 12 个 base `actor.*` 与 12 个 `actor_ft.*` 均逐项等于 BC 的 `ema` 对应 `network.*`（max absolute diff `0.0`）；与 BC 的非 EMA `model` 不相等（max absolute diff `0.53662109375`）。因此实际加载源是 BC `ema`，不是 checkpoint 整体 hash 的近似判断。

### 3.2 Square

正式报告对应本机目录：

`D:\AgentData\DPPO-P5-A\formal_run_20260914_074500\robomimic-finetune\square_ft_diffusion_mlp_ta4_td20_tdf10\2026-09-14_07-43-43_42\checkpoint\`

按要求选择一个现存 FT 模型 `state_200.pt`：bytes `19,012,935`，SHA256 `18031b2ed2ee422f6e85bd8c113e199d46e0b29f3f2b2d6b4828e67449e0cf25`。另读取 `state_0.pt` 做 BC 参数语义对照：bytes `19,012,267`，SHA256 `c9e22005b58ba2524782e3068c8341554e419acf66f095e238e56df41d9a7aa3`。

Square actor 形状为 time embedding `[64,32]`、`[32,64]`，condition MLP `[512,23]`、`[64,512]`，MLP `[1024,124]`、两个 `[1024,1024]` residual layers、输出 `[28,1024]`；`28 = 4 × 7`。critic 首层为 `[256,23]`，输出为 `[1,256]`；`state_200` 仍有 `network/actor/actor_ft/critic` 前缀。

BC checkpoint：

`D:\Desktop\my_project\paper_reproduction\DPPO\logs\robomimic-pretrain\square\square_pre_diffusion_mlp_ta4_td20\2024-07-10_01-46-16\checkpoint\state_8000.pt`

bytes `18,454,123`，SHA256 `368ce587294896b224eb549ba5f3a41cde3ebd249a1f6625eadd6926695c14b1`；top-level keys 是 `epoch/model/ema`。`state_0` 的 16 个 base `actor.*` 和 16 个 `actor_ft.*` 均逐项等于 BC `ema` 对应 `network.*`（max absolute diff `0.0`）；与 BC 非 EMA `model` 的 max absolute diff 为 `0.6477103233337402`。

### 3.3 normalization

| task | file bytes / SHA256 | keys | shapes | finite |
|---|---|---|---|---|
| Transport | 2,186 / `f6b893bdee28ab798c1d4293d7de565db61680c57d8909d5c046771a027e5296` | `obs_min/obs_max/action_min/action_max` | obs `[59]`，action `[14]` | all true |
| Square | 1,498 / `68ec0abfd989d5f0121f0e9a1dd074b49f859c167ff941c70d2eff8006074ff3` | `obs_min/obs_max/action_min/action_max` | obs `[23]`，action `[7]` | all true |

文件实际位于 `data/robomimic/transport/normalization.npz` 与 `data/robomimic/square/normalization.npz`，未被替换或重算。

## 4. 配置与指标口径

### 4.1 Expected vs observed

Transport resolved config 的 observed 值为：`obs_dim=59`、`action_dim=14`、`horizon_steps=8`、`act_steps=8`、`denoising_steps=20 (K)`、`ft_denoising_steps=10 (K')`、`n_envs=50`、`n_steps=400`、`max_episode_steps=800`、`batch_size=10000`、`update_epochs=5`、`gamma_denoising=0.99`、`min_sampling_denoising_std=0.1`、`min_logprob_denoising_std=0.1`。

这与本轮给定的后续诊断输入预期一致。observed 还包括 `train.gamma=0.999`、`reward_scale_running=true`、`n_train_itr=201`、`val_freq=10`；这些是正式运行配置中的额外字段，未被隐去。

PDF Table 7 记录的 Transport sampling std 下限为 `0.08`、`Nθ=10`。本机实际运行遵循官方 DPPO v0.6 YAML：sampling/log-prob std 下限均为 `0.1`，`ft_denoising_steps=10`。因此结论是：**遵循官方 v0.6 配置，但不是逐项等于 PDF Table 7**；没有为了匹配 PDF 静默修改本机配置。

Square resolved config 的 observed 值为 `obs_dim=23`、`action_dim=7`、`horizon_steps=4`、`act_steps=4`、`K=20`、`K'=10`、`50 envs`、`400 rollout steps`、`400 atomic-step episode`、`batch_size=10000`、`update_epochs=10`、`gamma_denoising=0.99`、std 下限均为 `0.1`。

### 4.2 Success 不混用

正式 DPPO report/result 的 success 是代码中明确实现的 reward-derived 口径：

```text
max(chunk_reward) / act_steps >= 1
```

本机 `TwoArmTransport._check_success()` 的 native success 是 payload/trash 两个接触条件的合取。现有正式 `run.log` 与 `result.pkl` 没有逐 episode native success 轨迹，所以：

- DPPO reported success：**PRESENT**，仅表示 reward-derived threshold success；
- native ever-success：**MISSING**，本轮不从 reward 或阶段标签反推；
- 两者没有在 manifest 或本报告中合并。

## 5. Resume 能力与不可恢复状态

官方 `save_model()` 只保存 `itr` 与 `model.state_dict()`。对 Transport `state_0/100/200` 和 Square 选定 FT 模型的 top-level keys 均未发现：optimizer state、LR scheduler state、Python/NumPy/Torch RNG、running reward scaler。由于 resolved config 启用了 `reward_scale_running=true`，严格从历史 checkpoint 接续原训练的 claim 不成立。

因此：

- 现有 state checkpoint 可用于后续只读参数诊断、对照和明确指定模型的 evaluation；
- 不可声称能够严格 resume 原训练过程；
- 不缺失本轮要求的 Transport `state_0/100/200`；没有重跑训练补 checkpoint；
- 不存在 early checkpoint 的历史状态被本轮修复或替换；缺少的 optimizer/RNG/scaler 状态属于官方保存格式不可恢复的信息。

## 6. S0a 判定与停止条件

判定：**PASS**。理由是 Transport 三个指定 FT checkpoint、Square 一个现存 FT checkpoint、两套 BC checkpoint、两套 normalization、实际 MuJoCo 2.1/mujoco-py backend、指定源文件以及 Transport 原生 success 语义均有本机证据；配置口径和指标口径已拆开，缺失项已显式标记。

允许范围内可进入后续诊断的输入：参数级 BC/FT 对照、checkpoint 结构审计、配置敏感性审计，以及在上级另行授权后的 S0b 工作。
未授权事项：本轮不执行 S0b、不训练、不正式长跑、不做 counterfactual 大规模实验；必须等待新的明确指令 **“S0a验收通过，允许进入S0b”**。
