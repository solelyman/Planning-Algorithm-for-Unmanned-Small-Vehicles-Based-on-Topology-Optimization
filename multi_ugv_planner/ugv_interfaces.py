#!/usr/bin/env python3
"""
UGV V-PRM + MPC 部署 —— 接口定义与传感器转换层
========================================================
本文档/模块把 UGV 的 V-PRM + Acados MPC 方法论接到 UGV 真车上，
确定"传感器信息怎么转化成规划器需要的东西"这一层接口。

数据流总览:

  UGV 硬件
  ├─ YDLidar X2 ─TCP 8889─> ugv_laser_ros2.py ─> /scan        (sensor_msgs/LaserScan,  RELIABLE,   0°=车头逆时针)
  ├─ ESP32      ─micro-ROS 8888─> agent             ─> /odom       (nav_msgs/Odometry,       BEST_EFFORT)
  └─ ESP32      <─ agent                           <─ /cmd_vel     (geometry_msgs/Twist,     BEST_EFFORT)

  上位机规划层（本接口的三段）:
  ① 感知转换    scan+odom ──> 障碍点图(世界坐标) / 障碍椭球 / UGVState
  ② V-PRM 全局  VPRM.plan(start,goal) ──> 世界坐标参考路径 ──> /reference_path (nav_msgs/Path)
  ③ MPC 局部    AcadosContouringSolver.solve(UGVState, ReferencePath, Ellipsoids)
               ──> Trajectory[{x,y,psi,v,omega}] ──> /cmd_vel (Twist, 限幅后)

  MPC 求解器是 UGV 的 C++ AcadosContouringSolver (contouring_unicycle 模型:
  5 状态 [x,y,psi,v,s], 2 控制 [a,w]), 与 UGV 差速 unicycle 模型同构,
  控制量 a/w 直接对应 cmd_vel.linear.x / cmd_vel.angular.z.
  V-PRM 用 vprm.py 里的 VPRM 类 (Python, 增量点图+净空加权)。

用法:
  python ugv_interfaces.py            # 自测: 合成数据验证全部转换函数
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# =====================================================================
# 1. UGV 硬件 / 模型参数
# =====================================================================

@dataclass(frozen=True)
class UGVModel:
    """UGV 差速小车运动学/几何参数 (对齐 URDF 与固件)"""
    # 几何
    body_radius: float = 0.10        # base_link 圆柱半径 [m]
    body_length: float = 0.12        # base_link 圆柱长度 [m]
    wheel_base: float = 0.20         # 轮间距 [m]
    wheel_radius: float = 0.065      # 轮径 [m]
    laser_offset_z: float = 0.075    # laser_link 相对 base_link 高度 [m]
    # 运动学限幅 (微处理器速度环硬限幅, 也是 MPC 输出必须 clamp 的边界)
    v_max: float = 0.26              # 最大线速度 [m/s]
    w_max: float = 1.0               # 最大角速度 [rad/s]
    a_max: float = 0.5               # 最大加速度 [m/s^2] (MPC 控制限幅用)
    # 到达判定
    goal_radius: float = 0.25        # 距离目标 < 此值视为到达 [m]


UGV = UGVModel()

# =====================================================================
# 2. ROS 话题 / QoS 约定
# =====================================================================

@dataclass(frozen=True)
class TopicSpec:
    name: str
    msg_type: str
    qos: str                     # RELIABLE / BEST_EFFORT
    role: str


TOPICS = [
    TopicSpec("/scan",            "sensor_msgs/LaserScan",    "RELIABLE",    "激光雷达原始数据 (V-PRM 障碍点图/MPC 椭球输入)"),
    TopicSpec("/scan_fused",      "sensor_msgs/LaserScan",    "RELIABLE",    "YOLO 视觉注入后的融合扫描 (优先于 /scan)"),
    TopicSpec("/odom",            "nav_msgs/Odometry",        "BEST_EFFORT", "ESP32 里程计 (x,y,yaw,v,w)"),
    TopicSpec("/reference_path",  "nav_msgs/Path",            "RELIABLE",    "V-PRM 输出的世界坐标参考路径 (MPC 输入)"),
    TopicSpec("/final_goal",      "geometry_msgs/Point",      "RELIABLE",    "目标点 (世界坐标 odom 系), 可选/可固定"),
    TopicSpec("/cmd_vel",         "geometry_msgs/Twist",      "BEST_EFFORT", "速度指令 (linear.x=v, angular.z=w)"),
]

QOS_ODOM = ("BEST_EFFORT", "ESP32 固件 micro-ROS 默认 BEST_EFFORT, 用 RELIABLE 订阅会收不到")


def odom_qos_profile():
    """构造 /odom 订阅 QoS: BEST_EFFORT + KeepLast(10) (对齐固件发布端)"""
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    return QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )


# =====================================================================
# 3. 传感器 → 规划器输入 转换
# =====================================================================

@dataclass
class UGVState:
    """等价 UGV C++ MultiUGV::UGVState"""
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    received: bool = False


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """四元数 → yaw (ROS 惯例, 世界系 Z 轴)"""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def odom_to_state(msg) -> UGVState:
    """nav_msgs/Odometry → UGVState (MPC/控制器输入)
    UGV 固件 /odom: pose 为世界系, twist 为车体系 (v 沿车头)."""
    st = UGVState(
        x=msg.pose.pose.position.x,
        y=msg.pose.pose.position.y,
        psi=quat_to_yaw(msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w),
        v=msg.twist.twist.linear.x,
        omega=msg.twist.twist.angular.z,
        received=True,
    )
    return st


def scan_to_vprm_update_args(msg, state: UGVState):
    """sensor_msgs/LaserScan + 当前位姿 → VPRM.update(ranges, angle_min, angle_inc, pos, yaw)
    雷达坐标系: 0°=车头, 索引递增角度递增 (YDLidar X2, 与部署脚本约定一致).
    注意: /scan 的 frame 是 laser_link, 与 base_link 共面同向, V-PRM 直接
    以车体 yaw 作为世界系偏移即可 (激光中心在车体正中心上方)."""
    return (list(msg.ranges), float(msg.angle_min),
            float(msg.angle_increment),
            np.array([state.x, state.y]), float(state.psi))


def scan_to_ellipsoids(msg, state: UGVState, radius: float = 0.35,
                       max_dist: float = 2.5, max_ell: int = 12):
    """sensor_msgs/LaserScan → MPC 椭球障碍列表 [(ox, oy, r)] (世界坐标)
    只取 <max_dist 的近距有效回波, 相邻点按角向聚合成圆 (作为 MPC 软惩罚).
    对接 UGV StageConstraints.EllipsoidObstacle{ox, oy, r}.
    半径 = 障碍半径 + 车体半径 (即 MPC 直接考虑整车安全半径)."""
    pts = []  # (x, y) 世界坐标
    n = len(msg.ranges)
    for i in range(n):
        r = float(msg.ranges[i])
        if not (0.05 < r < max_dist) or not math.isfinite(r):
            continue
        a = state.psi + float(msg.angle_min) + i * float(msg.angle_increment)
        pts.append((state.x + math.cos(a) * r, state.y + math.sin(a) * r))
    if not pts:
        return []

    # 角向分桶: 车体系方位角 → 桶, 每桶取最近点, 再以车体半径膨胀
    P = np.array(pts)
    rel = P - np.array([state.x, state.y])
    ang = np.arctan2(rel[:, 1], rel[:, 0]) - state.psi
    dist = np.linalg.norm(rel, axis=1)
    bin_w = 2 * math.asin(radius / max(0.3, dist.max())) * 180 / math.pi  # 近似角宽
    bins = {}
    for a, d, p in zip(ang, dist, P):
        k = int(round((math.degrees(a) % 360.0) / max(bin_w, 10.0)))
        if k not in bins or d < bins[k][0]:
            bins[k] = (d, p)
    ells = []
    for (_, p) in bins.values():
        ells.append((float(p[0]), float(p[1]), radius + UGV.body_radius))
        if len(ells) >= max_ell:
            break
    return ells


def lidar_to_36(ranges, angle_min, angle_increment, fov_deg=360.0,
                n_rays=36, ray_range=3.0):
    """/scan → 36 束重采样 (每 10° 扇区取最小值)
    供需要 36 维观测的模块复用."""
    out = np.full(n_rays, ray_range, dtype=np.float32)
    for i in range(n_rays):
        ang_deg = i * (fov_deg / n_rays)
        lo_idx = int((math.radians(ang_deg - fov_deg / n_rays / 2.0) - angle_min) / angle_increment)
        hi_idx = int((math.radians(ang_deg + fov_deg / n_rays / 2.0) - angle_min) / angle_increment)
        lo_idx = max(lo_idx, 0)
        hi_idx = min(hi_idx, len(ranges) - 1)
        if hi_idx < lo_idx:
            continue
        seg = ranges[lo_idx:hi_idx + 1]
        valid = seg[(seg > 0) & np.isfinite(seg)]
        if len(valid) > 0:
            out[i] = min(float(valid.min()), ray_range)
    return out


# =====================================================================
# 4. V-PRM / MPC 接口 (对齐 UGV 结构)
# =====================================================================

def vprm_parameters() -> dict:
    """V-PRM 参数 (真车调优值)"""
    return dict(
        margin=0.5,       # 边净空阈值: 净空 < margin 的边断连 (窄缝过滤)
        n_samples=240,    # 每周期采样点数
        neighbor_r=2.0,   # 可见性连线最大半径
        clearance=0.6,    # 窄通道判定 (低于此值加权)
        w_narrow=5.0,     # 窄通道惩罚权重
        lookahead=1.8,    # 引导点前视距离
        keep_s=5.0,       # 障碍点驻留时间
        map_span=5.0,     # 单帧雷达量程上限
        cell=0.06,        # 距离变换网格
    )


def mpc_parameters() -> dict:
    """MPC (AcadosContouringSolver) 参数 — 需与 C++ Params / 生成代码一致
    生成代码硬限幅: v∈[-0.01,1.9] w∈[-0.8,0.8] a∈[-2,2]
    (UGV 侧再由 traj_to_twist 二次 clamp 到 0.26 / 1.0)"""
    return dict(
        N=10, dt=0.4,                 # 预测步数 / 步长 → 4s 预测窗
        desired_speed=0.20,           # 参考速度 (UGV 巡航, 远小于 UGV 的 1.25)
        weight_acc=0.3, weight_angvel=1.0, weight_velocity=0.3,
        weight_contour=0.04, weight_lag=0.15,
        weight_terminal_angle=100.0, weight_terminal_contour=10.0,
        weight_obstacle=500.0,        # 椭球软惩罚权重
        robot_radius=0.10,            # 车体半径
        obstacle_clearance=0.5,       # 障碍净空
        RTI_ITERATIONS=10,
    )


# =====================================================================
# 5. 规划器输出 → /cmd_vel 转换
# =====================================================================

def traj_to_twist(traj, limits: Optional[UGVModel] = None):
    """UGV Trajectory (list of {x,y,psi,v,omega,s}) → geometry_msgs/Twist
    取第一预测步的速度/角速度, clamp 到 UGV 限幅.
    若 traj 为空返回全零指令."""
    from geometry_msgs.msg import Twist
    m = limits or UGV
    cmd = Twist()
    if traj is None or len(traj) == 0:
        return cmd
    p0 = traj[0]
    cmd.linear.x = float(np.clip(p0.get("v", 0.0), -m.v_max, m.v_max))
    cmd.angular.z = float(np.clip(p0.get("omega", 0.0), -m.w_max, m.w_max))
    return cmd


def path_to_path_msg(path: np.ndarray, frame_id: str = "odom", stamp=None):
    """V-PRM 世界坐标路径 (N,2) → nav_msgs/Path (发布到 /reference_path)"""
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    from builtin_interfaces.msg import Time
    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp or Time()
    for (x, y) in path:
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        msg.poses.append(ps)
    return msg


def path_msg_to_ref_path(msg):
    """nav_msgs/Path → [(x, y), ...] (喂给 C++ AcadosContouringSolver 前构造)"""
    return [(p.pose.position.x, p.pose.position.y) for p in msg.poses]


# =====================================================================
# 自测: 合成数据验证转换 (不依赖真车)
# =====================================================================

def _self_test():
    from types import SimpleNamespace

    # 合成 /odom: 世界系 (1.0, 2.0), yaw=90°, v=0.1, w=0.05
    qw = math.cos(math.pi / 4.0); qz = math.sin(math.pi / 4.0)
    odom = SimpleNamespace(
        pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw))),
        twist=SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(x=0.1, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.05))))
    st = odom_to_state(odom)
    assert abs(st.psi - math.pi / 2) < 1e-6, f"yaw={st.psi}"
    assert st.v == 0.1 and st.omega == 0.05
    print(f"[OK] odom_to_state: x={st.x:.2f} y={st.y:.2f} psi={math.degrees(st.psi):.0f}° v={st.v} w={st.omega}")

    # 合成 /scan: 正前方 1.0m 一堵墙, 其余 5.0m 空旷
    n = 360
    ranges = np.full(n, 5.0, dtype=float)
    for i in range(350, 360): ranges[i % n] = 1.0
    for i in range(0, 10):    ranges[i] = 1.0
    scan = SimpleNamespace(
        ranges=ranges, angle_min=0.0, angle_increment=2 * math.pi / n,
        range_max=12.0, range_min=0.02)

    r36 = lidar_to_36(ranges, 0.0, 2 * math.pi / n)
    assert r36[0] == 1.0 and r36[18] == 3.0
    print(f"[OK] lidar_to_36: 正前={r36[0]:.1f}m 后方={r36[18]:.1f}m")

    ells = scan_to_ellipsoids(scan, st, max_dist=2.0)
    assert len(ells) > 0
    print(f"[OK] scan_to_ellipsoids: {len(ells)} 个椭球, 首个 (ox={ells[0][0]:.2f}, oy={ells[0][1]:.2f}, r={ells[0][2]:.2f})")

    traj = [dict(x=1.1, y=2.0, psi=math.pi / 2, v=0.5, omega=3.0)]
    tw = traj_to_twist(traj)
    assert tw.linear.x == 0.26 and tw.angular.z == 1.0, f"clamp 失败: {tw}"
    print(f"[OK] traj_to_twist: clamp 0.5→{tw.linear.x}, 3.0→{tw.angular.z}")

    path = np.array([[0, 0], [1, 0], [2, 0.5]])
    pm = path_to_path_msg(path)
    assert len(pm.poses) == 3 and pm.header.frame_id == "odom"
    assert path_msg_to_ref_path(pm)[1] == (1.0, 0.0)
    print(f"[OK] path <-> Path 消息互转: {len(pm.poses)} 个点, frame={pm.header.frame_id}")

    print("\n=== 全部接口转换自测通过 ===")


if __name__ == "__main__":
    _self_test()
