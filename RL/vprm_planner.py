#!/usr/bin/env python3
"""
FishBot 轻量 V-PRM 全局规划器 (替换 A*)
核心 (对齐 mpc_planner 的 guidance planner / Visibility-PRM):
  1. 视线采样 Visibility Sampling: 在 free space 均匀采样节点, 与邻居直线连线,
     连线不穿过障碍 (膨胀半径) 才建边 -> 构成 PRM 图
  2. 同调类比较 (H-signature / Homology): 对每条候选路径, 计算绕每个障碍的
     符号角度积分 (环绕数), 作为该路径的拓扑签名. 不同签名 = 不同拓扑
  3. 多路径并行 + 选择: 用 A* 在 PRM 图上找 k 条不同同调的路径,
     按 selection_weights (consistency + length + acceleration) 选最优
  4. 找不到路 -> 返回空, 不退化直线穿墙 (避免"脑溢血路线"根因)

用法: vprm_path = vprm_plan(occ, excluded, start, goal, prev_sig)
  返回 (path_points, signature) 或 (None, None)
"""
import math
import random

import numpy as np

# 与 deploy_gz_vision.py 对齐的参数
PATH_BOUND = 6.0          # 世界范围 ±6m (11x11 地图)
PATH_CELL = 0.25          # 栅格尺寸 (与 A* 一致, 便于复用 occ)
ROBOT_R = 0.20            # 机器人半径
INFLATE = ROBOT_R + 0.08  # 障碍膨胀半径 (采样点不能进, 连线不能穿)

N_SAMPLES = 60            # PRM 采样节点数 (mpc_planner n_samples=30, 我们更多保证连通)
N_NEIGH = 6               # 每节点连接的最近邻居数
K_PATHS = 3               # 找 k 条不同同调路径


def astar_grid_plan(occ, start, goal, path_cell=0.2, path_bound=5.6, inflate=0.28):
    """网格 A* 全局路径 (逐格 4 邻域搜索, 保证能穿过窄口/死角).
    occ: 障碍栅格集合 (栅格坐标); start/goal: 世界坐标 (x,y)
    返回世界坐标路径点列 (含起点终点) 或 None
    """
    import heapq
    from collections import deque

    def _world_to_grid(p):
        gx = int(round(p[0] / path_cell))
        gy = int(round(p[1] / path_cell))
        # 越界或落障碍 -> 就近找自由格
        if (gx, gy) in occ or abs(p[0]) > path_bound or abs(p[1]) > path_bound:
            best = None
            for r in range(1, 10):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if abs(dx) != r and abs(dy) != r:
                            continue
                        ngx, ngy = gx + dx, gy + dy
                        if (ngx, ngy) not in occ and abs(ngx * path_cell) <= path_bound and abs(ngy * path_cell) <= path_bound:
                            best = (ngx, ngy)
                            break
                    if best:
                        break
                if best:
                    break
            if best is None:
                return None
            return best
        return (gx, gy)

    def _inflate_occ():
        # 障碍格外扩 inflate/path_cell 圈 (车体不能贴障碍)
        # inflate<=0: 不额外膨胀 (occ 已由调用方精确膨胀, 避免双重膨胀把可绕路径堵死)
        if inflate <= 0:
            return set(occ)
        inf = set()
        k = max(1, int(round(inflate / path_cell)))
        for (gx, gy) in occ:
            for dx in range(-k, k + 1):
                for dy in range(-k, k + 1):
                    inf.add((gx + dx, gy + dy))
        return inf

    occ_inf = _inflate_occ()

    # 距离场: 每个自由格到最近障碍的曼哈顿距离(格). A* 扩展时把"贴障碍"
    # 变成更高代价, 从而主动偏离障碍而不是只做硬膨胀.
    free_d = {}
    q = deque()
    gmax = int(path_bound / path_cell)
    for gx in range(-gmax, gmax + 1):
        for gy in range(-gmax, gmax + 1):
            if (gx, gy) in occ_inf:
                free_d[(gx, gy)] = 0
                q.append((gx, gy))
    while q:
        cx, cy = q.popleft()
        nd = free_d[(cx, cy)] + 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cx + dx, cy + dy)
            if abs(nb[0] * path_cell) > path_bound or abs(nb[1] * path_cell) > path_bound:
                continue
            if nb not in free_d:
                free_d[nb] = nd
                q.append(nb)

    s = _world_to_grid(start)
    g = _world_to_grid(goal)
    if s is None or g is None:
        return None
    if s in occ_inf or g in occ_inf:
        return None

    # A* 4 邻域
    open_heap = [(0.0, s)]
    came_from = {}
    g_score = {s: 0.0}
    f_score = {s: abs(s[0] - g[0]) + abs(s[1] - g[1])}
    closed = set()
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == g:
            break
        if cur in closed:
            continue
        closed.add(cur)
        for dgx, dgy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cur[0] + dgx, cur[1] + dgy)
            if nb in occ_inf or nb in closed:
                continue
            if abs(nb[0] * path_cell) > path_bound or abs(nb[1] * path_cell) > path_bound:
                continue
            # 距离场启发式: 离障碍越近惩罚越大, 让路径主动远离障碍.
            h_goal = abs(nb[0] - g[0]) + abs(nb[1] - g[1])
            h_obs = 0.0 if nb in free_d else max(0, 8 - free_d.get(nb, 8))
            step_cost = 1.0 + 0.15 * h_obs
            tentative = g_score[cur] + step_cost
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = cur
                g_score[nb] = tentative
                f_score[nb] = tentative + h_goal + 0.1 * h_obs
                heapq.heappush(open_heap, (f_score[nb], nb))
    if g not in came_from and s != g:
        return None

    # 回溯路径 (栅格坐标)
    grid_path = [g]
    while grid_path[-1] != s:
        grid_path.append(came_from[grid_path[-1]])
    grid_path.reverse()
    # 转世界坐标 (栅格中心)
    world = [((gx + 0.5) * path_cell, (gy + 0.5) * path_cell) for gx, gy in grid_path]
    # 去冗余共线点, 只保留拐点
    pts = world[:1]
    for i in range(1, len(world) - 1):
        a = np.array(world[i - 1]); b = np.array(world[i]); c = np.array(world[i + 1])
        v1 = b - a; v2 = c - b
        if v1[0] * v2[1] - v1[1] * v2[0] == 0 and v1 @ v2 > 0:
            continue  # 共线同向
        pts.append(world[i])
    # 简单平滑: 3 点滑动平均 + 细分插值, 让轨迹不是直插折线
    if len(pts) >= 3:
        sm = [pts[0]]
        for i in range(1, len(pts) - 1):
            sm.append(0.25 * np.array(pts[i - 1]) + 0.5 * np.array(pts[i]) + 0.25 * np.array(pts[i + 1]))
        sm.append(pts[-1])
        pts = [np.array(p) for p in sm]
    return pts


class PRM:
    def __init__(self, occ, excluded, seed=1):
        self.occ = occ
        self.excluded = excluded or set()
        self.rng = random.Random(seed)
        self.nodes = []       # [(x, y)]
        self.edges = []       # [(i, j, dist)]

    def _clear(self, x, y):
        """点 (x,y) 是否在 free space (远离所有障碍+禁忌)
        栅格约定与 deploy costmap 一致 (无偏移): gx = round(x / PATH_CELL)"""
        gx = int(round(x / PATH_CELL))
        gy = int(round(y / PATH_CELL))
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx * dx + dy * dy <= 9 and ((gx + dx, gy + dy) in self.occ or (gx + dx, gy + dy) in self.excluded):
                    return False
        return True

    def _line_clear(self, x1, y1, x2, y2):
        """线段 (x1,y1)-(x2,y2) 是否不穿过障碍 (密集采样检查)"""
        d = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(d / 0.05), 8)
        for k in range(1, steps):
            t = k / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self._clear(x, y):
                return False
        return True

    def sample(self, start, goal):
        """采样 + 建图 (起点终点必入图; 分层确定性采样保证覆盖)"""
        self.nodes = [start, goal]
        candidates = []
        # 1) 分层网格采样 (确定性覆盖整个地图)
        step = 1.2
        for gx in np.arange(-PATH_BOUND + 0.6, PATH_BOUND - 0.4, step):
            for gy in np.arange(-PATH_BOUND + 0.6, PATH_BOUND - 0.4, step):
                x = float(gx + self.rng.uniform(-0.15, 0.15))
                y = float(gy + self.rng.uniform(-0.15, 0.15))
                if self._clear(x, y):
                    candidates.append((x, y))
        # 2) 随机补充 (保证数量)
        while len(candidates) < N_SAMPLES:
            x = self.rng.uniform(-PATH_BOUND + 0.5, PATH_BOUND - 0.5)
            y = self.rng.uniform(-PATH_BOUND + 0.5, PATH_BOUND - 0.5)
            if self._clear(x, y):
                candidates.append((x, y))
        self.nodes += candidates[:N_SAMPLES]
        # 建边: 每节点连最近 N_NEIGH 个可见邻居
        n = len(self.nodes)
        dists = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(self.nodes[i][0] - self.nodes[j][0], self.nodes[i][1] - self.nodes[j][1])
                dists[i][j] = dists[j][i] = d
        for i in range(n):
            neigh = sorted(range(n), key=lambda j: dists[i][j])[1:N_NEIGH + 1]
            for j in neigh:
                xi, yi = self.nodes[i]
                xj, yj = self.nodes[j]
                if self._line_clear(xi, yi, xj, yj):
                    self.edges.append((i, j, dists[i][j]))

    def astar_on_graph(self):
        """PRM 图上 A*: 返回 (path_node_idx, dist) 或 None"""
        n = len(self.nodes)
        adj = {}
        for i, j, d in self.edges:
            adj.setdefault(i, []).append((j, d))
            adj.setdefault(j, []).append((i, d))
        import heapq
        pq = [(0.0, 0)]
        came = {}
        g = {0: 0.0}
        while pq:
            dcur, u = heapq.heappop(pq)
            if u == 1:
                # 回溯
                path = [1]
                while path[-1] != 0:
                    path.append(came[path[-1]])
                path.reverse()
                return path, dcur
            if dcur > g.get(u, 1e18):
                continue
            for v, dv in adj.get(u, []):
                nd = dcur + dv
                if nd < g.get(v, 1e18):
                    g[v] = nd
                    came[v] = u
                    heapq.heappush(pq, (nd, v))
        return None


def h_signature(path_xy, obstacles):
    """H-signature: 对每个障碍 (cx,cy), 计算路径绕它的符号角度积分.
    返回 tuple (每障碍的半整数环绕数) -> 路径的拓扑签名.
    半整数量化 (步长 0.5): 绕墙上方 ≈ -0.5 / 下方 ≈ +0.5 / 整圈 ±1...
    (不能用 round 到整数: Python round(±0.5)=0 会把上下两条路合并成一个拓扑!)"""
    sig = []
    for cx, cy in obstacles:
        ang_sum = 0.0
        for k in range(len(path_xy) - 1):
            x1, y1 = path_xy[k]
            x2, y2 = path_xy[k + 1]
            a1 = math.atan2(y1 - cy, x1 - cx)
            a2 = math.atan2(y2 - cy, x2 - cx)
            da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
            ang_sum += da
        sig.append(round(ang_sum / math.pi * 2) / 2)  # 半整数环绕数
    return tuple(sig)


def vprm_plan(occ, excluded, start, goal, prev_sig=None, obstacles=None,
              path_cell=0.2, path_bound=5.6, robot_r=0.20):
    """主入口: V-PRM 规划. 返回 (path_points_world, signature) 或 (None, None)
    prev_sig: 上次选中的路径签名.
      选择规则 (防蛇形核心):
        - 当前拓扑 (prev_sig) 的路径仍可达 → 无条件沿用, 绝不轻易换路
        - 当前拓扑被排除堵死 / 首次规划 → 才换拓扑, 且其他路线加 switch 惩罚
    path_cell/path_bound: 与调用方 costmap 栅格一致 (deploy 用 0.2/5.6)
    robot_r: 膨胀半径, 找不到路时自动减半宽松重试 (全局路径只是引导, 局部安全层兜底)"""
    obstacles = sorted(obstacles or [], key=lambda p: (p[0], p[1]))  # 稳定排序 -> 签名结构稳定

    def _build(occ_, excluded_, rr):
        global PATH_CELL, PATH_BOUND, ROBOT_R, INFLATE
        PATH_CELL, PATH_BOUND, ROBOT_R = path_cell, path_bound, rr
        INFLATE = rr + 0.08
        prm = PRM(occ_, excluded_, seed=1)
        prm.sample(start, goal)
        # 枚举所有 PRM 路径的同调类, 保留各类最短
        found = {}
        for _ in range(30):
            res = prm.astar_on_graph()
            if res is None:
                break
            path, dist = res
            path_xy = [prm.nodes[i] for i in path]
            sig = h_signature(path_xy, obstacles)
            if sig not in found or dist < found[sig][1]:
                found[sig] = (path_xy, dist)
            # 从图里移除这条路径的边 (迫使下条路径拓扑不同)
            for k in range(len(path) - 1):
                e = (path[k], path[k + 1])
                rev = (e[1], e[0])
                prm.edges = [x for x in prm.edges if (x[0], x[1]) != e and (x[0], x[1]) != rev]
        return found

    found = _build(occ, excluded, robot_r)
    if not found and robot_r > 0.10:
        found = _build(occ, excluded, max(robot_r * 0.5, 0.06))  # 宽松回退 (仍不直线穿墙)

    if not found:
        return None, None

    # ===== 选择 (防蛇形核心) =====
    SIG_SWITCH_PENALTY = 8.0   # 换拓扑惩罚 (m): 换路代价足够大, 只有堵死才换
    if prev_sig is not None and prev_sig in found:
        # 当前拓扑仍可达 → 无条件沿用 (被排除堵死的路不会出现在 found 里)
        path_xy, _ = found[prev_sig]
        return [np.asarray(p, dtype=float) for p in path_xy], prev_sig
    # 当前拓扑堵死 / 首次规划: 换拓扑 (最短优先, 加 switch 惩罚避免来回横跳)
    best_eff, best_sig, best_path = None, None, None
    for sig, (path_xy, dist) in found.items():
        eff = dist + (SIG_SWITCH_PENALTY if prev_sig is not None else 0.0)
        if best_eff is None or eff < best_eff:
            best_eff, best_sig, best_path = eff, sig, path_xy
    return [np.asarray(p, dtype=float) for p in best_path], best_sig
