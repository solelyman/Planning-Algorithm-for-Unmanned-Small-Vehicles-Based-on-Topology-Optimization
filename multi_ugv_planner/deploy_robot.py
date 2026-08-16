#!/usr/bin/env python3
"""
FishBot 真车 NavRL 部署: 雷达 -> V-PRM 全局路径 -> RL(PO) -> cmd_vel
  - 订阅 /scan (鱼香ROS ydlidar 节点发布)
  - V-PRM 增量点图 (世界坐标) + 净空加权建图 -> 前方引导点
    (与训练 env 的 V-PRM 同语义: obs[36] 剩余弧长/15, obs[37] 引导点角)
  - 36 束雷达 -> RL 推理 (短程/长程模型均可, --norm path|euclid)
  - 发布 /cmd_vel (差速速度指令)
  - 目标: 默认正前方 2m (可改), 或 RViz 2D Goal
用法:
  python deploy_robot.py --model models/best_prm_warm_v4.zip --goal-x 6 --norm path
依赖: ros2 环境 + ydlidar 节点已跑
"""
import os
import sys
import math
import time
import heapq
import argparse
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from extractors import NavRLMLPExtractor, NavRLCNNExtractor
    _HAS_EXTRACTORS = True
except Exception:
    _HAS_EXTRACTORS = False

# ---------- 参数 ----------
N_RAYS = 36          # 训练用的雷达束数
RAY_RANGE = 3.0      # 训练雷达量程
MAX_LIN = 0.26       # 训练线速度上限
MAX_ANG = 1.0        # 训练角速度上限
GOAL_RADIUS = 0.25   # 到达判定
CTRL_HZ = 20.0       # 控制频率 (dt=0.05s, 训练 dt=0.02s, 用 20Hz 推理)
FRONT_WARN = 1.0     # 预警敏感半径: 前方障碍 <1m 就减速提前绕 (Nav2 inflation)
DEFAULT_GOAL_DIST = 2.0  # 默认目标距离 (车前方, 短程)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def lidar_to_36(ranges, angle_min, angle_increment, fov_deg=360.0):
    """把 /scan 的雷达数据重采样成 36 束 (每 10°, 覆盖 360°, i=0 车头正前)
    对齐训练 env._get_ranges: ang = yaw + i*10°, i=0 正前, 逆时针递增.
    /scan 约定: angle_min=0, 索引递增角度递增 (0°=车头, 逆时针).
    每束取该 10° 扇区内全部有效点的最小值, 避免单束丢点漏掉近距障碍."""
    out = np.full(N_RAYS, RAY_RANGE, dtype=np.float32)
    for i in range(N_RAYS):
        # 束 i 的中心角: 0°, 10°, ..., 350° (i=0 正前)
        ang_deg = i * 10.0
        # 该束覆盖 [ang-5°, ang+5°] 的 /scan 索引区间
        lo_idx = int((math.radians(ang_deg - 5.0) - angle_min) / angle_increment)
        hi_idx = int((math.radians(ang_deg + 5.0) - angle_min) / angle_increment)
        lo_idx = max(lo_idx, 0)
        hi_idx = min(hi_idx, len(ranges) - 1)
        if hi_idx < lo_idx:
            continue
        seg = ranges[lo_idx:hi_idx + 1]
        valid = seg[(seg > 0) & np.isfinite(seg)]
        if len(valid) > 0:
            out[i] = min(float(valid.min()), RAY_RANGE)
    return out


def max_gap_angle(ranges36, gap_thresh=1.0):
    """找 36 束里最长的连续空旷扇区, 返回其中心束索引 (0=正前, 顺时针).
    比'扇区最小值'更接近 Nav2 costmap 的膨胀层语义:
    一束 0.45m 的读数不再掩盖同侧 12m 的出口."""
    N = len(ranges36)
    best_center, best_len, cur_len, cur_start = None, 0, 0, 0
    for i in range(N * 2):
        i2 = i % N
        if ranges36[i2] > gap_thresh:
            if cur_len == 0:
                cur_start = i2
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_center = (cur_start + cur_len // 2) % N
            cur_len = 0
    if cur_len > best_len:
        best_len = cur_len
        best_center = (cur_start + cur_len // 2) % N
    if best_center is None:
        return 0
    return best_center


class VPRM:
    """真车 visibility-PRM (对齐训练 env 的 V-PRM 语义):
    /scan -> 世界坐标增量点图 -> 网格距离变换 -> 净空加权可见性建图 ->
    Dijkstra 选路. 输出前方引导点 + 剩余路径弧长给 RL 局部策略.
    与 mpc_planner-main 的 visibility-PRM 同构: 边按 margin 膨胀过滤
    (窄缝断连), 低于 clearance 的窄通道按缺额加权 (宁可绕路不挤缝).
    """
    def __init__(self, margin=0.5, n_samples=240, neighbor_r=2.0,
                 clearance=0.6, w_narrow=5.0, lookahead=1.8,
                 keep_s=5.0, max_points=1500, sample_stride=3,
                 map_span=5.0, cell=0.06):
        self.margin = margin
        self.n_samples = n_samples
        self.neighbor_r = neighbor_r
        self.clearance = clearance
        self.w_narrow = w_narrow
        self.lookahead = lookahead
        self.keep_s = keep_s
        self.max_points = max_points
        self.sample_stride = sample_stride
        self.map_span = map_span
        self.cell = cell
        self._pts = []        # (t, x, y) 世界坐标障碍点
        self.path = None      # np.ndarray (N,2) 世界坐标路径
        self.goal = None
        self._g_dist = None
        self._g_origin = None
        self._g_w = self._g_h = 0

    # ---------- 增量点图 ----------
    def update(self, ranges, angle_min, angle_inc, pos, yaw):
        """把 /scan 射线转世界坐标并入点图, 剔除过期点"""
        now = time.time()
        self._pts = [(t, x, y) for (t, x, y) in self._pts if now - t < self.keep_s]
        n = len(ranges)
        if n > 0 and angle_inc > 0:
            px, py = float(pos[0]), float(pos[1])
            for i in range(0, n, self.sample_stride):
                r = float(ranges[i])
                if not (0.1 < r < self.map_span) or not math.isfinite(r):
                    continue
                a = yaw + angle_min + i * angle_inc
                self._pts.append((now, px + math.cos(a) * r, py + math.sin(a) * r))
        if len(self._pts) > self.max_points:
            self._pts = self._pts[-self.max_points:]

    def _points(self):
        if not self._pts:
            return np.zeros((0, 2))
        return np.array([[x, y] for _, x, y in self._pts])

    # ---------- 网格距离变换 ----------
    def _build_grid(self, lo, hi):
        cell = self.cell
        w = int(math.ceil((hi[0] - lo[0]) / cell)) + 1
        h = int(math.ceil((hi[1] - lo[1]) / cell)) + 1
        occ = np.zeros((h, w), dtype=bool)
        P = self._points()
        if len(P):
            gi = ((P[:, 0] - lo[0]) / cell).astype(int)
            gj = ((P[:, 1] - lo[1]) / cell).astype(int)
            ok = (gi >= 0) & (gi < w) & (gj >= 0) & (gj < h)
            occ[gj[ok], gi[ok]] = True
        INF = 1e9
        d = np.where(occ, 0.0, INF)
        for i in range(h):       # 前向
            for j in range(w):
                if d[i, j] == 0:
                    continue
                m = d[i, j]
                if i > 0:                     m = min(m, d[i - 1, j] + 1.0)
                if j > 0:                     m = min(m, d[i, j - 1] + 1.0)
                if i > 0 and j > 0:           m = min(m, d[i - 1, j - 1] + 1.4142)
                if i > 0 and j < w - 1:       m = min(m, d[i - 1, j + 1] + 1.4142)
                d[i, j] = m
        for i in range(h - 1, -1, -1):       # 后向
            for j in range(w - 1, -1, -1):
                if d[i, j] == 0:
                    continue
                m = d[i, j]
                if i < h - 1:                 m = min(m, d[i + 1, j] + 1.0)
                if j < w - 1:                 m = min(m, d[i, j + 1] + 1.0)
                if i < h - 1 and j < w - 1:   m = min(m, d[i + 1, j + 1] + 1.4142)
                if i < h - 1 and j > 0:       m = min(m, d[i + 1, j - 1] + 1.4142)
                d[i, j] = m
        self._g_dist, self._g_origin = d, lo
        self._g_w, self._g_h = w, h

    def _clearance_at(self, p):
        """点处到最近障碍的距离 (m); 出网格按 0 (保守)"""
        gi = int((p[0] - self._g_origin[0]) / self.cell)
        gj = int((p[1] - self._g_origin[1]) / self.cell)
        if not (0 <= gi < self._g_w and 0 <= gj < self._g_h):
            return 0.0
        return float(self._g_dist[gj, gi]) * self.cell

    def _seg_clearance(self, p1, p2):
        c = 1e9
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            c = min(c, self._clearance_at(p1 + t * (p2 - p1)))
        return c

    # ---------- 规划 ----------
    def plan(self, start, goal):
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        self.goal = goal
        self.path = np.array([start, goal])   # 兜底直达
        lo = np.minimum(start, goal) - 2.5
        hi = np.maximum(start, goal) + 2.5
        self._build_grid(lo, hi)
        if len(self._pts) == 0:               # 空地图: 直线引导
            return self.path

        rng = np.random.default_rng()
        pts = [start, goal]
        tries = 0
        while len(pts) < self.n_samples and tries < self.n_samples * 50:
            tries += 1
            p = rng.uniform(lo, hi)
            if self._clearance_at(p) > self.margin:
                pts.append(p)
        pts = np.array(pts)
        n = len(pts)

        # 可见性连线: 净空 < margin 断开 (窄缝断连); 边权净空加权
        edges = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pts[i] - pts[j]) > self.neighbor_r:
                    continue
                c = self._seg_clearance(pts[i], pts[j])
                if c < self.margin:
                    continue
                d = float(np.linalg.norm(pts[i] - pts[j]))
                w = d
                if c < self.clearance:
                    w *= 1.0 + self.w_narrow * (self.clearance - c) / self.clearance
                edges[i].append((j, w))
                edges[j].append((i, w))

        # Dijkstra 最短(净空加权)
        INF = 1e9
        dist = [INF] * n
        prev = [-1] * n
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d, i = heapq.heappop(pq)
            if d > dist[i]:
                continue
            for j, w in edges[i]:
                nd = d + w
                if nd < dist[j]:
                    dist[j] = nd
                    prev[j] = i
                    heapq.heappush(pq, (nd, j))
        if dist[1] >= INF:
            return self.path
        path = []
        cur = 1
        while cur != -1:
            path.append(pts[cur])
            cur = prev[cur]
        path.reverse()
        self.path = np.array(path)
        return self.path

    # ---------- 引导 ----------
    def lookahead_point(self, pos):
        """沿路径前方 LOOKAHEAD 处引导点 (对齐 env._lookahead_point)"""
        path = self.path
        pos = np.asarray(pos, dtype=float)
        best_d, best_i, best_t = 1e9, 0, 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-9:
                continue
            t = float(np.clip(((pos - a) @ ab) / L2, 0.0, 1.0))
            p = a + t * ab
            d = float(np.linalg.norm(p - pos))
            if d < best_d:
                best_d, best_i, best_t = d, i, t
        look = self.lookahead
        i, t = best_i, best_t
        a = path[i] + t * (path[i + 1] - path[i])
        while True:
            b = path[i + 1]
            seg = float(np.linalg.norm(b - a))
            if seg < 1e-9:
                if i + 2 < len(path):
                    i += 1
                    a = path[i]
                    continue
                return path[-1]
            if seg >= look:
                return a + (look / seg) * (b - a)
            look -= seg
            a = b
            i += 1
            if i + 1 >= len(path):
                return path[-1]

    def remaining(self, pos):
        """沿路径到终点的剩余弧长 (obs[36] 归一化用)"""
        path = self.path
        pos = np.asarray(pos, dtype=float)
        total = 0.0
        best_s, best_d, s = 0.0, 1e9, 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-9:
                continue
            t = float(np.clip(((pos - a) @ ab) / L2, 0.0, 1.0))
            p = a + t * ab
            d = float(np.linalg.norm(p - pos))
            if d < best_d:
                best_d, best_s = d, s + t * math.sqrt(L2)
            s += math.sqrt(L2)
            total += math.sqrt(L2)
        return max(total - best_s, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(MODELS_DIR, "best_margin_v6.zip"))
    ap.add_argument("--goal-x", type=float, default=None,
                    help="固定目标 X (车体坐标系, 前方), 默认 2.0")
    ap.add_argument("--goal-y", type=float, default=0.0,
                    help="固定目标 Y (车体坐标系, 左正右负), 默认 0.0")
    ap.add_argument("--norm", choices=["path", "euclid"], default="path",
                    help="obs[36] 距离归一化: path=路径剩余弧长/15 (prm/新模型, 默认), "
                         "euclid=欧氏距离/3 (v6 旧模型)")
    ap.add_argument("--arch", choices=["cnn", "mlp", "auto"], default="auto",
                    help="特征提取器结构: auto=按模型自动探测 (推荐), cnn=NavRLCNNExtractor, "
                         "mlp=NavRLMLPExtractor")
    ap.add_argument("--no-ros", action="store_true",
                    help="不用 ROS, 从 /scan socket 直连 (8889) 调试")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    custom_objects = {}
    if _HAS_EXTRACTORS:
        # 特征提取器定义与训练时一致 (模块路径), py3.10 也能反序列化
        # arch=auto: 先按模型内保存的 policy_kwargs 探测, 加载失败再换
        archs = [args.arch] if args.arch != "auto" else ["cnn", "mlp"]
        model = None
        last_err = None
        for arch in archs:
            cls = NavRLCNNExtractor if arch == "cnn" else NavRLMLPExtractor
            custom_objects["policy_kwargs"] = dict(
                features_extractor_class=cls,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=[256, 256],
            )
            try:
                model = PPO.load(args.model, custom_objects=custom_objects)
                print(f"[deploy] 加载模型 {args.model} (features_extractor={arch})")
                break
            except (RuntimeError, ValueError) as e:
                last_err = e
        if model is None:
            print(f"[deploy] 模型加载失败: {last_err}")
            print("[deploy] 可用模型 arch: cnn=NavRLCNNExtractor (v4/v5/v6/prm), "
                  "mlp=NavRLMLPExtractor (v8/unified)")
            sys.exit(1)
    else:
        model = PPO.load(args.model)

    goal_body = np.array([args.goal_x if args.goal_x is not None else DEFAULT_GOAL_DIST,
                          args.goal_y])
    print(f"[deploy] 目标 (车体坐标系): {goal_body}, obs距离归一化: {args.norm}")

    if args.no_ros:
        run_no_ros(model, goal_body)
    else:
        run_ros(model, goal_body, args.norm)


def run_ros(model, goal_body, dist_norm="path"):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry

    rclpy.init()
    node = Node("navrl_deploy")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)
    scan_raw = {"ranges": None, "angle_min": 0.0, "angle_inc": 0.0}
    scan_fused = {"ranges": None, "angle_min": 0.0, "angle_inc": 0.0}
    odom = {"pos": None, "yaw": 0.0, "goal_set": False}
    goal_world = {"val": np.array([0.0, 0.0])}
    prm = VPRM()            # 真车 V-PRM (增量点图 + 净空加权)
    replan_every = 0.5      # 秒
    last_plan = 0.0
    last_map_update = 0.0

    def scan_cb(msg):
        # 原始 /scan (回退源)
        scan_raw["ranges"] = np.array(msg.ranges, dtype=float)
        scan_raw["angle_min"] = msg.angle_min
        scan_raw["angle_inc"] = msg.angle_increment

    def scan_fused_cb(msg):
        # /scan_fused: 融合了 YOLO 虚拟激光 (视觉障碍注入), 优先使用
        scan_fused["ranges"] = np.array(msg.ranges, dtype=float)
        scan_fused["angle_min"] = msg.angle_min
        scan_fused["angle_inc"] = msg.angle_increment

    def odom_cb(msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        odom["pos"] = np.array([p.x, p.y])
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        odom["yaw"] = yaw
        # 首次收到 odom: 把车体坐标目标 (goal_body) 转到世界坐标固定住
        if not odom["goal_set"]:
            ca, sa = math.cos(yaw), math.sin(yaw)
            goal_world["val"][0] = p.x + ca * goal_body[0] - sa * goal_body[1]
            goal_world["val"][1] = p.y + sa * goal_body[0] + ca * goal_body[1]
            odom["goal_set"] = True
            node.get_logger().info(f"[deploy] 目标已固定世界坐标: {goal_world['val']}")

    node.create_subscription(LaserScan, "/scan", scan_cb, 10)
    node.create_subscription(LaserScan, "/scan_fused", scan_fused_cb, 10)
    # /odom 来自 micro-ROS agent (BEST_EFFORT), 默认 RELIABLE 收不到
    odom_qos = rclpy.qos.QoSProfile(
        depth=10,
        reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
        durability=rclpy.qos.DurabilityPolicy.VOLATILE)
    node.create_subscription(Odometry, "/odom", odom_cb, odom_qos)

    # 车体坐标目标 -> 世界坐标 (需要 odom, 先默认 0 朝向)
    lin = ang = 0.0
    rate = node.create_rate(CTRL_HZ)
    _dbg_last = -1
    _stuck_t0 = None
    _recover_until = None
    _r36_hist = deque(maxlen=5)

    print("[deploy] 开始导航: V-PRM 全局路径 + RL 局部避障 (视觉融合 /scan_fused). Ctrl+C 停止")
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
        # 优先用 /scan_fused (YOLO 视觉已注入), 融合节点没跑则回退原始 /scan
        src = scan_fused if scan_fused["ranges"] is not None else scan_raw
        if src["ranges"] is None:
            continue

        now = time.time()
        # 低频更新点图 (雷达 20Hz, 每 0.1s 抽一次即可)
        if odom["pos"] is not None and now - last_map_update > 0.1:
            prm.update(src["ranges"], src["angle_min"],
                       src["angle_inc"], odom["pos"], odom["yaw"])
            last_map_update = now

        ranges36 = lidar_to_36(src["ranges"],
                               src["angle_min"], src["angle_inc"])
        # 中值滤波: 抑制单帧丢点/噪声 (sim2real 标准做法)
        _r36_hist.append(ranges36)
        if len(_r36_hist) >= 3:
            ranges36 = np.median(np.asarray(_r36_hist), axis=0).astype(np.float32)

        # 目标角: V-PRM 路径上的前方引导点 (车体坐标系), 需要 odom
        if odom["pos"] is not None and odom["goal_set"]:
            pos = np.asarray(odom["pos"], dtype=float)
            # 周期重规划: 目标世界坐标固定, 障碍点图增量更新
            if now - last_plan > replan_every:
                prm.plan(pos, goal_world["val"])
                last_plan = now
                if len(prm.path) > 2:
                    node.get_logger().info(
                        f"[deploy] V-PRM 重规划: {len(prm.path)} 点, "
                        f"剩余 {prm.remaining(pos):.2f}m")
            guide = prm.lookahead_point(pos)
            to_guide = guide - pos
            goal_dir_ang = math.atan2(to_guide[1], to_guide[0]) - odom["yaw"]
            if dist_norm == "path":
                dist_obs = prm.remaining(pos)      # 沿路径剩余弧长 (对齐 env /15)
            else:
                dist_obs = float(np.linalg.norm(goal_world["val"] - pos))  # v6 欧氏 /3
            dist = float(np.linalg.norm(goal_world["val"] - pos))
        else:
            goal_dir_ang = 0.0
            dist_obs = float(np.linalg.norm(goal_body))
            dist = dist_obs
        while goal_dir_ang > math.pi:
            goal_dir_ang -= 2 * math.pi
        while goal_dir_ang < -math.pi:
            goal_dir_ang += 2 * math.pi

        obs = np.concatenate([
            ranges36 / RAY_RANGE,
            [min(dist_obs / 15.0 if dist_norm == "path" else dist_obs / 3.0, 1.0),
             goal_dir_ang / math.pi,
             lin / MAX_LIN, ang / MAX_ANG],
        ]).astype(np.float32)

        act, _ = model.predict(obs, deterministic=True)
        lin = float(np.clip(act[0], -MAX_LIN, MAX_LIN))
        ang = float(np.clip(act[1], -MAX_ANG, MAX_ANG))

        # 调试: 周期性打印关键观测与动作
        if int(now * 2) % 10 == 0 and int(now * 2) != _dbg_last:
            _dbg_last = int(now * 2)
            node.get_logger().info(
                f"[dbg] f前6={np.round(ranges36[:6],2)} "
                f"fmin={np.min(ranges36[:6]):.2f} "
                f"dist={dist_obs:.2f} goal_ang={math.degrees(goal_dir_ang):.0f}° "
                f"act=({lin:.2f},{ang:.2f}) front={front if 'front' in dir() else -1:.2f}")

        # 到达判定: 目标 < 0.25m 停车
        if dist < GOAL_RADIUS:
            lin = ang = 0.0
            print("[deploy] 到达目标!")

        # ===== 两级避障 (敏感半径, 对齐 Nav2 inflation radius 思路) =====
        # 正前方 ±30° = 束 33,34,35,0,1,2,3 (i=0 是正前)
        front = float(np.min(np.concatenate([ranges36[33:], ranges36[:4]])))
        gap_center = max_gap_angle(ranges36)     # 最长空旷扇区中心束
        # 转向方向: gap 在左侧(束4..18) => 左转, 右侧(束18..32) => 右转
        gap_ang = gap_center * 10.0
        gap_side = "L" if gap_ang < 180 else "R"
        gap_turn = 0.6 if gap_side == "L" else -0.6
        # 前方障碍留出的侧向空隙 (max_gap 长度*10° 应足够宽, 否则算倒车)
        if front < 0.35:
            # 紧急: 朝最大空隙侧强制转向; 空隙太窄就立刻倒车脱困
            # (不原地硬磨: 倒一点 + 打方向, 一有空隙就转出去)
            ang = gap_turn
            if abs(gap_ang - 180) < 70:          # 最大空隙在正后方 => 前面被围死
                lin = -0.12
            else:
                lin = 0.0
            if int(now * 2) % 10 == 0 and int(now * 2) != _dbg_last:
                _dbg_last = int(now * 2)
                node.get_logger().info(
                    f"[dbg] EMERG front={front:.2f} gap={gap_ang:.0f}°({gap_side}) "
                    f"-> turn={'L' if ang > 0 else 'R'}, back={lin < 0}")
        elif front < FRONT_WARN:
            # 预警敏感半径内: 减速, RL 转向不够果断就强制朝最大空隙转
            # (模型在 0.35~1.0m 区间常输出小转向, 不强制就会直撞)
            lin *= 0.4
            if abs(ang) < 0.3:
                ang = gap_turn
            if int(now * 2) % 10 == 0 and int(now * 2) != _dbg_last:
                _dbg_last = int(now * 2)
                node.get_logger().info(
                    f"[dbg] WARN front={front:.2f} gap={gap_ang:.0f}°({gap_side}) "
                    f"-> turn={'L' if ang > 0 else 'R'}")

        # ===== 停滞检测 recover (Nav2 recover 思路): 3s 没前进且前方堵 -> 倒车打方向 =====
        if odom["pos"] is not None and odom["goal_set"]:
            if _recover_until is not None and now < _recover_until:
                lin, ang = -0.15, 0.7          # 强制倒车 + 打方向
                if int(now * 2) % 10 == 0 and int(now * 2) != _dbg_last:
                    _dbg_last = int(now * 2)
                    node.get_logger().info(
                        f"[dbg] RECOVER backing up (left={now - (_recover_until - 1.5):.1f}s in)")
            else:
                _recover_until = None
                if _stuck_t0 is None:
                    _stuck_t0, _stuck_d0 = now, dist
                elif now - _stuck_t0 > 3.0:
                    if dist <= _stuck_d0 - 0.05:
                        _stuck_t0, _stuck_d0 = now, dist   # 确实在前进, 重置
                    elif front < 0.8:
                        _recover_until = now + 1.5         # 卡死: 倒车 1.5s
                        _stuck_t0 = None
                        node.get_logger().info(
                            f"[dbg] STUCK dist={dist:.2f} front={front:.2f} -> RECOVER")
                    else:
                        _stuck_t0, _stuck_d0 = now, dist
        else:
            _stuck_t0 = None

        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        pub.publish(twist)

    node.destroy_node()
    rclpy.shutdown()


def run_no_ros(model, goal_body):
    """无 ROS 调试: 从 ydlidar TCP 直读 (8889), 打印输出不发布"""
    import socket
    import struct
    print("[deploy] 无 ROS 模式: 直连雷达 8889, 仅打印 (不发布 cmd_vel)")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("192.168.3.130", 8889))
    s.settimeout(2.0)
    buf = b""
    lin = ang = 0.0
    try:
        while True:
            data = s.recv(4096)
            if not data:
                continue
            buf += data
            # 找 YDLidar 帧头 (AA 55 或类似), 简化处理
            # 这里仅演示连接, 实际协议解析参考 ydlidar_node.py
            print(f"[deploy] 收到 {len(data)} 字节 (连接正常)")
            break
    finally:
        s.close()


if __name__ == "__main__":
    main()
