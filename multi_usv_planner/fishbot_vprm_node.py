#!/usr/bin/env python3
"""
Fishbot 上车 V-PRM 节点: /scan + /odom -> VPRM.plan() -> /reference_path
====================================================================
V-PRM 全局规划(世界坐标), 输出参考路径给 paper2 的 C++ unicycle MPC
(usv_planner_node_exe). 复用 deploy_robot.py 验证过的 VPRM 类 +
fishbot_interfaces.py 的传感器转换.

话题:
  订阅 /scan      (sensor_msgs/LaserScan,  RELIABLE)    雷达
  订阅 /odom      (nav_msgs/Odometry,     BEST_EFFORT)  里程计
  发布 /reference_path (nav_msgs/Path,     RELIABLE+transient_local)
  订阅 /goal_pose (geometry_msgs/PoseStamped, 可选, RViz 2D Goal)

用法:
  python fishbot_vprm_node.py --goal-x 2.0 --goal-y 0.0
"""
import argparse
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fishbot_interfaces import (FISHBOT, USVState, odom_qos_profile,
                                odom_to_state, scan_to_vprm_update_args,
                                path_to_path_msg)
from deploy_robot import VPRM

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point


class FishbotVPRMNode(Node):
    def __init__(self, goal_xy):
        super().__init__("fishbot_vprm_node")
        self.vprm = VPRM(**{
            'margin': 0.3, 'n_samples': 240, 'neighbor_r': 2.0,
            'clearance': 0.35, 'w_narrow': 5.0, 'lookahead': 1.8,
            'keep_s': 5.0, 'map_span': 5.0, 'cell': 0.06,
        })
        self.state = USVState()
        self.goal = np.array(goal_xy, dtype=float)
        self._last_path = None      # 上一条安全路径 (翻转抑制)
        self._last_path_time = 0.0

        # 订阅: /scan RELIABLE, /odom BEST_EFFORT(固件)
        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_cb, 10)
        self.sub_odom = self.create_subscription(
            Odometry, "/odom", self.odom_cb, odom_qos_profile())
        self.sub_goal = self.create_subscription(
            PoseStamped, "/goal_pose", self.goal_cb, 10)

        # 发布: 参考路径 (transient_local, 对齐 paper2 prm_node 发布方式)
        qos = rclpy.qos.QoSProfile(
            depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_path = self.create_publisher(Path, "/reference_path", qos)
        # 发布: 目标转发给 usv_planner 的 /final_goal (Point), 触发 episode 重置
        self.pub_goal = self.create_publisher(Point, "/final_goal", 10)

        self.timer = self.create_timer(0.3, self.loop)  # 3.3Hz: 路径稳定, 避免 MPC 追移动起点原地打转
        # 终点直线区半径: 距 goal 小于此值时不再跑 V-PRM 绕行规划,
        # 直接发布 [车位置→goal] 直线路径让 MPC 冲线停车, 避免绕圈/倒车
        self.approach_r = 1.5
        self.get_logger().info(
            f"Fishbot V-PRM 启动 | goal=({self.goal[0]:.2f},{self.goal[1]:.2f}) | "
            f"margin=0.6 lookahead=1.8 | 发布 /reference_path")

    def scan_cb(self, msg):
        if not self.state.received:
            return
        ranges, amin, ainc, pos, yaw = scan_to_vprm_update_args(msg, self.state)
        self.vprm.update(ranges, amin, ainc, pos, yaw)

    def odom_cb(self, msg):
        self.state = odom_to_state(msg)

    def goal_cb(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y], dtype=float)
        pt = Point()
        pt.x = float(self.goal[0]); pt.y = float(self.goal[1])
        self.pub_goal.publish(pt)   # 转发给 usv_planner 重置 episode
        self.get_logger().info(f"新目标: ({self.goal[0]:.2f}, {self.goal[1]:.2f}), 已转发 /final_goal")

    def _densify(self, path, spacing=0.05):
        """把 VPRM 稀疏 waypoints 插值成 <=spacing 间隔的密集点。
        paper2 的 acados contouring solver 期望密集参考路径(仿真里 V-PRM 输出
        就是密集点), 2 点路径会退化成样条无约束且易触发求解失败。"""
        pts = [np.asarray(p, dtype=float) for p in path]
        if len(pts) < 2:
            return pts
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            seg = np.linalg.norm(b - a)
            if seg < 1e-6:
                continue
            n = max(1, int(math.ceil(seg / spacing)))
            for i in range(1, n + 1):
                out.append(a + (b - a) * (i / n))
        return out

    def _project_to_safety(self, pts, safe_clearance=0.25, max_iter=30, step=0.05):
        """线性化硬约束: 沿距离场梯度把路径点推到离障碍 >= safe_clearance 处.
        对齐 mpc_planner-main 的 projectToSafety (Douglas-Rachford 投影的
        距离场简化版). 这样 MPC 看到的参考路径天然无碰撞, 椭球惩罚只兜底,
        不会出现"MPC 认为沿路径必撞 → v=0"的卡死. """
        vprm = self.vprm
        if vprm._g_dist is None:
            return pts
        g, origin, cell = vprm._g_dist, vprm._g_origin, vprm.cell
        h, w = g.shape
        out = []
        for p in pts:
            pp = np.asarray(p, dtype=float).copy()
            for _ in range(max_iter):
                gi = int((pp[0] - origin[0]) / cell)
                gj = int((pp[1] - origin[1]) / cell)
                if not (0 < gi < w - 1 and 0 < gj < h - 1):
                    break
                c = float(g[gj, gi]) * cell
                if c >= safe_clearance:
                    break
                # 距离场梯度 = 离开障碍最快的方向 (数值差分, 单位 m)
                gx = (g[gj, gi + 1] - g[gj, gi - 1]) * cell
                gy = (g[gj + 1, gi] - g[gj - 1, gi]) * cell
                gn = math.hypot(gx, gy)
                if gn < 1e-6:
                    break
                pp[0] += gx / gn * step
                pp[1] += gy / gn * step
            out.append(pp)
        return out

    def loop(self):
        if not self.state.received:
            return
        start = np.array([self.state.x, self.state.y])
        dist_goal = float(np.linalg.norm(start - self.goal))
        # 到达检测: 车在 goal 附近(0.35m)时停止规划, 发布原地单点路径让 MPC 停车.
        # 否则车冲过 goal 后 vprm 会从车位置往"后方"的 goal 反向规划 → MPC 倒车.
        if dist_goal < 0.35:
            self.get_logger().info(
                f"[ARRIVED] 距 goal {dist_goal:.2f}m, 停车")
            self.pub_path.publish(path_to_path_msg([start, start], frame_id="odom"))
            return
        # 终点直线区: 距 goal < approach_r 时不再跑 V-PRM 绕行规划.
        # V-PRM 每 0.3s 从车位置重采样, 车冲过 goal 后路径反复换向,
        # MPC 追着移动路径在终点绕圈 → 这里直接发 [车→goal] 直线冲线停车.
        if dist_goal < self.approach_r:
            path = [start, np.asarray(self.goal, dtype=float)]
            self.get_logger().info(f"[APPROACH] 距 goal {dist_goal:.2f}m, 直线冲线")
        else:
            path = list(self.vprm.plan(start, self.goal))  # ndarray 无 append, 先转 list
            if path is None or len(path) < 2:
                return
            # DBG: 打印 plan 原始输出形状 (区分回退直线 [start,goal] 与 V-PRM 绕行路径)
            head = ", ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in path[:3])
            tail = ", ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in path[-3:])
            self.get_logger().info(f"[PATH] start=({start[0]:.2f},{start[1]:.2f}) n={len(path)} head=[{head}] tail=[{tail}]")
        # 路径延长: V-PRM 区沿路径末段延长 0.8m 触发 arrived;
        # approach 直线区沿 [车→goal] 方向延长 3.5m 覆盖 MPC 预测时域
        # (N=10*dt=0.4 → 4s 视野), 否则 MPC 看到 0.7m 的短路径以为
        # "马上走完该停了", vref 被压到 0.15 蠕行. 延长后车保持速度沿
        # 直线冲线, 到 0.35m 内由 ARRIVED 检测发单点路径停车.
        g = np.asarray(self.goal, dtype=float)
        if dist_goal >= self.approach_r:
            last = np.asarray(path[-1], dtype=float)
            d = last - g
            dl = float(np.linalg.norm(d))
            if dl > 1e-6:
                ext_dir = d / dl
            else:
                ext_dir = np.array([1.0, 0.0])
            path.append(g + ext_dir * 0.8)
        else:
            to_goal = g - start
            tg = float(np.linalg.norm(to_goal))
            if tg > 1e-6:
                ext_dir = to_goal / tg
            else:
                ext_dir = np.array([1.0, 0.0])
            path.append(g + ext_dir * 3.5)
        dense = self._densify(path)
        safe = self._project_to_safety(dense)
        if len(safe) < 2:
            return
        # DBG: 发布前安全路径形状 (project_to_safety 后的头尾, MPC 实际看到的)
        s_head = ", ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in safe[:4])
        s_tail = ", ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in safe[-2:])
        self.get_logger().info(f"[SAFE] n={len(safe)} head=[{s_head}] tail=[{s_tail}]")
        # 翻转抑制: 仅 V-PRM 绕行区有效. 直线冲线区(approach)路径方向就是
        # 指向 goal, 若与旧 V-PRM 路径方向不同会被误判 FLIP 沿用旧路径 → 绕圈.
        now = self.get_clock().now().nanoseconds / 1e9
        if (dist_goal >= self.approach_r and self._last_path is not None
                and (now - self._last_path_time) < 1.0):
            new_dir = np.asarray(safe[1]) - np.asarray(safe[0])
            old_dir = np.asarray(self._last_path[1]) - np.asarray(self._last_path[0])
            nd, od = float(np.linalg.norm(new_dir)), float(np.linalg.norm(old_dir))
            if nd > 1e-6 and od > 1e-6:
                new_dir /= nd
                old_dir /= od
                if new_dir[1] * old_dir[1] < -0.25:
                    self.get_logger().info("[FLIP] 路径翻转, 沿用旧路径")
                    safe = self._last_path
        self._last_path = [np.asarray(p, dtype=float).copy() for p in safe]
        self._last_path_time = now
        self.pub_path.publish(path_to_path_msg(safe, frame_id="odom"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-x", type=float, default=2.0)
    ap.add_argument("--goal-y", type=float, default=0.0)
    args = ap.parse_args()

    rclpy.init()
    node = FishbotVPRMNode((args.goal_x, args.goal_y))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
