# 基于拓扑优化的无人小型车辆规划算法 · Contouring MPC

Planning Algorithm for Unmanned Small Vehicles Based on Topology Optimization
（Fishbot 两轮差速小车 · MPC 路径跟踪与避障）

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus)
![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python)
![acados](https://img.shields.io/badge/acados-v1.4-009688)
![V-PRM](https://img.shields.io/badge/Planner-V--PRM-orange)

---

> **English** | 中文见下（English README, Chinese below）

A **model predictive control (MPC) path-tracking and obstacle-avoidance stack** for the
**Fishbot** two-wheel differential-drive robot (ESP32 + YDLidar X2), running on ROS 2 Humble.

It couples a **V-PRM global planner** (with local-goal approach / arrival detection) with a
real-time **contouring MPC** solved by **acados** (SQP-RTI + HPIPM). Two obstacle constraint
formulations are implemented and switchable by one parameter:

| Test | Constraint formulation | `use_linear_constraints` | Remark |
|------|------------------------|--------------------------|--------|
| **A** | **Ellipsoid soft penalty** — per-stage circle `(x-ox)²+(y-oy)²-r²` penalized in the cost | `false` | DecompUtil-style, easy to warm start, may squeeze the corridor when many lidar points hit one obstacle |
| **B** | **Hard linearized half-space** — `con_h_expr`: `h = a1·x + a2·y - b ≤ 0` (12 half-spaces/stage), obstacle linearized at the **warm-start anchor** (previous solution shifted + radial projection) | `true` | mpc_planner-style, no QP infeasibility from overlapping ellipsoids; keeps hard safety guarantees |

Both tests are validated to **circle around an obstacle, run the approach straight line and stop at the goal**.

---

## 中文简介

为 **Fishbot 两轮差速小车**（ESP32 主控 + YDLidar X2 激光雷达，ROS 2 Humble / micro-ROS）实现的
**MPC 路径跟踪与避障**完整链路：

- **V-PRM 全局规划**：基于雷达点云在线构建 PRM 绕行路径，带 0.35 m 障碍安全余量；
  终点直线区（距目标 < 1.5 m）直接发直线、不跑 PRM，并对短路径延长覆盖 MPC 预测时域。
- **Contouring MPC**：acados SQP-RTI 实时求解（N=10，dt=0.4 s），`warm start = 上一帧解 shift + 障碍径向投影`。
- **两种障碍约束，一个参数切换**（见上表 Test A / Test B）：
  - A：椭球软罚（N_ELL=12 椭球，代价项）；
  - B：线性半空间硬约束（`con_h_expr`，NH=12/阶段，障碍在 warm-start 锚点处线性化）。
- **到达检测**：距目标 < 0.35 m 停车，防止冲过目标后反向规划导致倒车。

## Features / 特性

- **Two obstacle constraint modes** (soft ellipsoid vs. hard linearized half-space), one-parameter switch
- **acados SQP-RTI** with HPIPM, `PARTIAL_CONDENSING_HPIPM`, multi-RTI iterations
- **Warm start from previous solution + obstacle radial projection** → avoids QP infeasibility / MINSTEP
- **Overlapping-ellipsoid merging** (laser sweep of one physical obstacle → merged circle)
- **V-PRM global planner** with flip suppression, approach segment, arrival detection
- **Pure ROS 2 Humble** C++ node, no DecompUtil / no multi-body legacy code

## System Architecture / 系统架构

```
 /scan ──┐
 /odom ──┼──▶ fishbot_vprm_node.py ──▶ /reference_path ──┐
         │      (V-PRM global plan,                    │
         │       approach & arrival)                   ▼
 /odom ──┼──────────────────────────────────▶ usv_planner_node_exe (Contouring MPC)
         │                                          │  acados SQP-RTI
 /scan ──┘  ──▶ scan_obstacles_ (ellipsoid/        │  NH=12 half-spaces or
                    half-space obstacles)           ▼  N_ELL=12 soft ellipsoids
                                               /cmd_vel
```

## Quick Start / 快速开始

### Dependencies / 依赖

- ROS 2 Humble
- [acados](https://github.com/acados/acados) (v1.4, with BLASFEO/HPIPM), installed to `/home/<user>/.local/share/acados`
- Eigen3

```bash
# 1. Build (colcon)
cd <your_ws>/src
cp -r fishbot_mpc_planner multi_usv_planner
cd <your_ws>
source /opt/ros/humble/setup.bash
colcon build --packages-select multi_usv_planner

# 2. Simulation with fake odom/scan (no hardware)
source install/setup.bash
ros2 run multi_usv_planner fake_odom.py &   # integrate /cmd_vel -> /odom at 50 Hz
ros2 run multi_usv_planner fake_scan.py &   # fixed obstacle at (1.0, 0.0) r=0.15
ros2 launch multi_usv_planner fishbot_mpc.launch.py
```

> `fishbot_mpc.launch.py` starts the V-PRM node and the MPC node with
> `config/usv_params_fishbot.yaml`. It references `scripts/fishbot_vprm_node.py`
> (in this repo under `multi_usv_planner/`).

### Test A vs Test B / A/B 测试切换

Edit `config/usv_params_fishbot.yaml`:

```yaml
mpc:
  use_linear_constraints: true    # false -> Test A (soft ellipsoid), true -> Test B (hard linearized)
```

### Real robot / 真机（Fishbot）

- 上位机通过 UDP 8888 连接 micro-ROS Agent（Humble 固件版本必须匹配）；
- 雷达通过 TCP 8889 转发到 `/scan`；
- 话题：`/odom`(nav_msgs/Odometry)、`/scan`(sensor_msgs/LaserScan)、`/cmd_vel`(geometry_msgs/Twist)。

## Key Implementation Notes / 关键实现

- **Warm start**（Test B 能解的根基）：`solve()` 先算 warm-start 轨迹——k 步用上一帧第 k+1 步解（速度下限 0.3 m/s），
  再按各阶段椭球把轨迹点径向推出障碍（+1e-3）；**锚点 = warm-start 轨迹点而非参考路径点**，
  避免"第一步跳到障碍另一侧"的动力学不可行。
- **重叠椭球合并**：激光把同一障碍扫成多个相邻点（每个 r=0.4）会压死可行域导致 QP INFEASIBLE；
  中心距 < 0.5·(r1+r2) 则合并（中心取中点、半径取外切覆盖、dist 取 min）。
- **蠕行修复**：`acados_weight_velocity: 3.0`（速度跟踪主导）+ V-PRM `margin: 0.3`（绕行路径不过度外扩）。
- **RTI_ITERATIONS=6**：2 次从低速度 warm start 收敛不到最优（会蠕行），6 次让非线性充分收敛。

## Directory Layout / 目录结构

```
fishbot_mpc_planner/
├── CMakeLists.txt / package.xml       # ROS 2 ament 包
├── include/multi_usv_planner/        # 头文件 (solver, constraint builder, types…)
├── src/                              # usv_planner_node.cpp (主节点), acados_contouring_solver.cpp…
├── acados/generated/contouring_solver/  # 生成的 acados 求解器 (contouring_unicycle, NH=12, NP=126)
├── config/usv_params_fishbot.yaml    # 参数 (Test A/B 切换)
├── launch/fishbot_mpc.launch.py      # V-PRM + MPC 一键启动
├── scripts/generate_contouring_solver.py  # 重新生成 acados 求解器
└── multi_usv_planner/                # Python: fishbot_vprm_node.py, fake_odom.py, fake_scan.py
```

## License

Apache-2.0. See [LICENSE](LICENSE).
