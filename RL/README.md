# Fishbot RL: MuJoCo Navigation

<a id="zh-cn"></a>

**中文** | [English](#english)

本目录是小型无人车项目的强化学习导航分支，包含早期 PPO baseline 和当前 SAC + MuJoCo 连续目标验证。

## 演示

当前演示在 MuJoCo 中运行履带式 UGV，使用激光观测、真实碰撞几何和连续目标切换。

<p align="center">
  <img src="docs/images/sac_mujoco_3d_seed2.gif" alt="MuJoCo 3D SAC demo" width="46%">
  <img src="docs/images/sac_multigoal_seed2.gif" alt="SAC continuous-goal demo" width="46%">
</p>

<p align="center">
  <img src="docs/images/sac_mujoco_3d_seed2.png" alt="MuJoCo 3D snapshot" width="46%">
  <img src="docs/images/sac_mujoco_pipeline.svg" alt="SAC MuJoCo workflow" width="46%">
</p>

## 当前结果

选用 checkpoint：

```text
fishbot_mujoco_sac/runs_dyn/sac_20260825_010443/best_model.zip
```

该 checkpoint 约 324 MB，因此没有直接提交到 Git。需要共享权重时建议使用 Git LFS 或 GitHub Release asset。

| 策略 | 安全层 | Seeds | 结果 |
| --- | --- | --- | --- |
| SAC checkpoint | 无 | 2, 7, 8, 10 | 部分成功；seed 2 和 7 仍可能碰撞 |
| SAC checkpoint | heading guard | 2, 7, 8, 10 | 测试中 8/8 goals，0 碰撞 |
| SAC checkpoint | heading guard | seed 2 GUI | 10/10 goals，0 碰撞，0 timeout |

<p align="center">
  <img src="docs/results/sac_curve.png" alt="SAC training curve" width="76%">
</p>

## 运行

```bash
cd RL/fishbot_mujoco_sac
python3.12 -m venv mujoco_env
source mujoco_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/fishbot_pycache ./mujoco_env/bin/python scripts/view_multigoal_dyn.py \
  --model runs_dyn/sac_20260825_010443/best_model.zip \
  --stage 3 --scene boxes \
  --goals 10 \
  --seed 2 \
  --sleep 0.03 \
  --goal-radius 0.35 \
  --max-steps-per-goal 900 \
  --heading-guard
```

heading guard 不是规划器，只是在 guide 方向和车体朝向差异较大时限制前进速度，避免连续切换目标时出现高速大半径转弯。

## 文件结构

```text
RL/
├── fishbot_mujoco_sac/
│   ├── assets/tracked_nav.xml      # MuJoCo 履带车场景
│   ├── scripts/env_dyn.py          # Gymnasium 环境
│   ├── scripts/train_sac_dyn.py    # SAC 训练
│   ├── scripts/eval_dyn.py         # 评估
│   └── scripts/view_multigoal_dyn.py
├── docs/images/                    # GIF 和图
├── docs/results/                   # 曲线与评估摘要
├── train_mj_ppo.py                 # 早期 PPO baseline
└── docs/MODEL_CARD.md              # checkpoint 与验证说明
```

后续重点是继续提高 SAC 在 MuJoCo 中的稳定性，并补 ROS 2 桥接用于真机验证。

---

<a id="english"></a>

## English

[中文](#zh-cn) | **English**

This folder contains the reinforcement-learning navigation branch of the small UGV project. It includes the earlier PPO baseline and the current SAC + MuJoCo continuous-goal validation.

## Demo

The current demo runs a tracked UGV in MuJoCo with lidar-style observations, real collision geometry, and consecutive goal switching.

<p align="center">
  <img src="docs/images/sac_mujoco_3d_seed2.gif" alt="MuJoCo 3D SAC demo" width="46%">
  <img src="docs/images/sac_multigoal_seed2.gif" alt="SAC continuous-goal demo" width="46%">
</p>

<p align="center">
  <img src="docs/images/sac_mujoco_3d_seed2.png" alt="MuJoCo 3D snapshot" width="46%">
  <img src="docs/images/sac_mujoco_pipeline.svg" alt="SAC MuJoCo workflow" width="46%">
</p>

## Current Result

Selected checkpoint:

```text
fishbot_mujoco_sac/runs_dyn/sac_20260825_010443/best_model.zip
```

The checkpoint is about 324 MB, so it is not committed to Git. Use Git LFS or a GitHub Release asset if the weight file needs to be shared.

| Policy | Safety layer | Seeds | Result |
| --- | --- | --- | --- |
| SAC checkpoint | none | 2, 7, 8, 10 | partially successful; seed 2 and 7 can still collide |
| SAC checkpoint | heading guard | 2, 7, 8, 10 | 8/8 goals, 0 collision in tested seeds |
| SAC checkpoint | heading guard | seed 2 GUI | 10/10 goals, 0 collision, 0 timeout |

<p align="center">
  <img src="docs/results/sac_curve.png" alt="SAC training curve" width="76%">
</p>

## Run

```bash
cd RL/fishbot_mujoco_sac
python3.12 -m venv mujoco_env
source mujoco_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/fishbot_pycache ./mujoco_env/bin/python scripts/view_multigoal_dyn.py \
  --model runs_dyn/sac_20260825_010443/best_model.zip \
  --stage 3 --scene boxes \
  --goals 10 \
  --seed 2 \
  --sleep 0.03 \
  --goal-radius 0.35 \
  --max-steps-per-goal 900 \
  --heading-guard
```

The heading guard is not a planner. It only limits forward speed when the guide direction is far from the robot heading, which prevents large-radius turns during continuous goal switches.

## Files

```text
RL/
├── fishbot_mujoco_sac/
│   ├── assets/tracked_nav.xml      # MuJoCo tracked UGV scene
│   ├── scripts/env_dyn.py          # Gymnasium environment
│   ├── scripts/train_sac_dyn.py    # SAC training
│   ├── scripts/eval_dyn.py         # Evaluation
│   └── scripts/view_multigoal_dyn.py
├── docs/images/                    # GIFs and figures
├── docs/results/                   # Curves and evaluation summaries
├── train_mj_ppo.py                 # Earlier PPO baseline
└── docs/MODEL_CARD.md              # Checkpoint and validation notes
```

The next step is to improve SAC robustness in MuJoCo and add a ROS 2 bridge for real-robot validation.
