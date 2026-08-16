#!/usr/bin/env python3
"""
Visibility-PRM 增量点图规划器 (世界坐标).
与 mpc_planner 的 visibility-PRM 同构: 边按 margin 膨胀过滤(窄缝断连),
低于 clearance 的窄通道按缺额加权(宁可绕路不挤缝).
"""
import math
import time
import heapq

import numpy as np

class VPRM:
    """真车 visibility-PRM:
    /scan -> 世界坐标增量点图 -> 网格距离变换 -> 净空加权可见性建图 ->
    Dijkstra 选路. 输出前方引导点 + 剩余路径弧长给局部规划器.
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

