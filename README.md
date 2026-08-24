# 基于拓扑优化的无人小型车辆规划算法

Planning Algorithm for Unmanned Small Vehicles Based on Topology Optimization

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus)
![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python)
![MuJoCo](https://img.shields.io/badge/Sim-MuJoCo-black)
![SAC](https://img.shields.io/badge/RL-SAC-6A5ACD)

本仓库包含两条小型无人车导航路线：

- `multi_ugv_planner/`：V-PRM 全局规划 + Contouring MPC 路径跟踪与避障
- [`RL/`](RL/README.md)：MuJoCo 动力学环境下的强化学习导航验证

## MuJoCo 3D Demo

当前 RL 展示使用履带式 UGV、MuJoCo 碰撞动力学、激光观测和连续目标切换。演示策略为 SAC checkpoint 加轻量转向保护，主要用于验证策略能否在同一场景中连续完成多个目标点。

<p align="center">
  <img src="RL/docs/images/sac_mujoco_3d_seed2.gif" alt="MuJoCo 3D SAC demo" width="49%">
  <img src="RL/docs/images/sac_multigoal_seed2.gif" alt="SAC continuous-goal demo" width="49%">
</p>

<p align="center">
  <img src="RL/docs/images/sac_mujoco_pipeline.svg" alt="SAC MuJoCo workflow" width="86%">
</p>

## Project Layout

```text
.
├── multi_ugv_planner/          # ROS 2 Python nodes: V-PRM, fake odom, fake scan
├── include/multi_ugv_planner/  # MPC solver headers
├── src/                        # Contouring MPC and planner node
├── config/ugv_params.yaml      # Main planner parameters
├── launch/ugv_mpc.launch.py    # ROS 2 launch file
├── docs/images/                # Main planner figures
└── RL/                         # MuJoCo RL navigation branch
```

## Classic Planner

The classic stack is a ROS 2 Humble navigation pipeline for a differential-drive UGV. It combines an online V-PRM planner with an acados-based contouring MPC controller.

<p align="center">
  <img src="docs/images/mpc_architecture_figure2.svg" alt="V-PRM + Contouring MPC architecture" width="92%">
</p>

```bash
cd <ros2_ws>/src
cp -r <this_repo> multi_ugv_planner
cd <ros2_ws>
source /opt/ros/humble/setup.bash
colcon build --packages-select multi_ugv_planner
source install/setup.bash

ros2 run multi_ugv_planner fake_odom.py &
ros2 run multi_ugv_planner fake_scan.py &
ros2 launch multi_ugv_planner ugv_mpc.launch.py
```

Obstacle handling can be switched in `config/ugv_params.yaml`:

```yaml
mpc:
  use_linear_constraints: true
```

`true` uses linearized hard half-space constraints. `false` uses soft ellipsoid penalties.

## RL Navigation

The [`RL/`](RL/README.md) folder keeps the earlier PPO baseline and adds the current MuJoCo SAC experiment. The public files include the environment, training scripts, evaluation scripts, curves, and demo media. Large trained checkpoints are not committed to the repository.

```bash
cd RL/fishbot_mujoco_sac
python3.12 -m venv mujoco_env
source mujoco_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

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

Current note: the raw SAC checkpoint is not yet fully robust in every tested seed. The showcased continuous-goal demo uses a small heading-speed guard to reduce wide, high-speed turns during goal switching.

## License

Apache-2.0. See [LICENSE](LICENSE).
