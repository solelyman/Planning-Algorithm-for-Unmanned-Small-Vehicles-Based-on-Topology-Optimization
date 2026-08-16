"""
端到端导航环境 (MuJoCo) — 全面对齐 DRL-Transformer Scout 配方
  obs     : image (4,128,160) 鱼眼灰度图 4 帧堆叠 (归一化 0~1)
          + goal (4,) = [Dist/15, beta/pi, 最小激光/16, 前方60°平均激光/16]
  action  : Box(-1,1,2)  ->  target v = [(a0+1)*MAX_LIN/2, a1*MAX_ANG]  (DRL 接口)
  reward  : DRL 原版 = r_heuristic(20*ddist) + r_action(a0*2-|a1|) + r_smooth(-|da1|/4)
            + r_target(+100) + r_collision(-100)   (不再加 r_safety/r_freeze 干扰)
  动力学  : 目标速度直接到位 (DRL 语义), Scout 尺度
"""
import math
import os
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np

from mj_offscreen import MuJoCoOffscreenRenderer
from vprm_planner import vprm_plan, astar_grid_plan

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ugv.xml")


class End2EndNavEnv(gym.Env):
    # ★ ugv 真车动力学 (用户确认: max 0.28 m/s; 低速差速车)
    MAX_LIN = 0.28           # 最大前进线速度 (m/s)
    MAX_ANG = 2.0            # 最大角速度 (rad/s)
    DELTA_LIN = 0.5          # 仅供旧代码/老师动作比例换算保留, 主线不再用 delta-v
    DELTA_ANG = 0.8          # 仅供旧代码/老师动作比例换算保留
    ROOM_HALF = 5.4          # 房间半宽 (gazebo 外墙 ±5.425, 对齐部署)
    DT = 0.05                # 50Hz
    # 鱼眼图像
    IMG_H, IMG_W = 128, 160
    FOVY = 120.0             # 广角 (模拟鱼眼)
    FRAME_STACK = 4
    # 激光 (复用 X2N sim2real 噪声模型, 对齐 env.py 参数)
    RAY_RANGE = 16.0         # 真实 X2N: 0.01~16m
    N_RAYS = 90              # 90 束, 每 4°, 覆盖 360°
    # DRL 判定 (Scout 车体更大; 走廊场景 0.4 更合理)
    COLLISION_DIST = 0.28    # min_laser < 0.28 = 碰撞 (车体半径0.10 + 安全余量; 杜绝"贴墙蹭"偷鸡)
    ROBOT_RADIUS = 0.10      # UGV 车体半径 (m)
    GOAL_RADIUS = 0.5        # Dist < 0.5 = 到达
    MAX_STEPS = 1800         # ugv 0.28m/s: 1800步×0.05s×0.28=25.2m 行程 (远目标5-8m+绕路余量)

    def __init__(self, seed=0, n_obstacles=None, goal_dist=(2.0, 3.0), obs_mode="laser", goal_dim=4, goal_margin=0.15, detour_ratio=0.6):
        super().__init__()
        # obs_mode="laser": 纯激光+目标 (PPO 课程训练, 快); "vision": 鱼眼图+目标 (BC/SAC)
        self.obs_mode = obs_mode
        # n_obstacles: None=空地图, int>=0=随机 n 个 box 障碍 (课程)
        self.n_obstacles = n_obstacles
        # goal_dist: 目标距离范围 (课程从近到远)
        self.goal_dist = goal_dist
        # goal_margin: 目标距障碍最小距离 (小=允许目标贴障碍, 逼策略学绕障到达)
        self.goal_margin = goal_margin
        # detour_ratio: 目标被障碍直线遮挡(必须绕路)的比例. 高=训练分布接近部署(gz 必须绕),
        # 防止到达率虚高(全是直线简单局). 但太高会太难练, 默认 0.6
        self.detour_ratio = detour_ratio
        # goal_dim=2 时 pstate 只含 [Dist/15, beta/pi] (对齐 BC 预训练 nb_pstate=2)
        self.goal_dim = goal_dim
        self._rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fisheye")
        if cam_id >= 0:
            self.model.cam_fovy[cam_id] = self.FOVY
        self.data = mujoco.MjData(self.model)
        self._renderer = MuJoCoOffscreenRenderer(self.model, "fisheye", self.IMG_H, self.IMG_W)

        if self.obs_mode == "laser":
            # [激光 90 束归一化(0~1), dist/15, beta/pi, lin/MAX, ang/MAX] 拼接向量, MlpPolicy 直接用
            self.observation_space = gym.spaces.Box(-1.0, 1.0, (self.N_RAYS + 4,), np.float32)
        else:
            self.observation_space = gym.spaces.Dict({
                "image": gym.spaces.Box(0.0, 1.0, (self.FRAME_STACK, self.IMG_H, self.IMG_W), np.float32),
                "goal": gym.spaces.Box(-1.0, 1.0, (self.goal_dim,), np.float32),
                "laser": gym.spaces.Box(0.0, 1.0, (self.N_RAYS,), np.float32),
            })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        self._frames = deque(maxlen=self.FRAME_STACK)
        self._pos_trace = deque(maxlen=10)   # 卡死检测: 最近 10 步位置
        self._prev_act = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2)
        self.step_count = 0
        self._collision = False
        self._arrived = False
        self._yaw = 0.0   # 标量航向角积分 (修复原四元数点积 bug)
        # 边界墙 OBB (cx, cy, hx, hy, yaw) — 与 gazebo outer_wall ±5.425m 对齐
        self._walls = [
            (0.0, 5.425, 5.5, 0.15, 0.0), (0.0, -5.425, 5.5, 0.15, 0.0),
            (5.425, 0.0, 0.15, 5.5, 0.0), (-5.425, 0.0, 0.15, 5.5, 0.0),
        ]

    # ============ 场景生成 (课程: 空地图 -> 随机 n 障碍) ============

    def _place_obstacles(self):
        """n_obstacles=None -> 空地图; 'nav2' -> gazebo ugv_square_world2 的固定结构
        (L角/红房子/十字墙/家具, 正交化重建); int>=0 -> 随机 n 个 box"""
        # 障碍 body 全部先移出场地 (位置/旋转在 body 上, geom_quat 在此 mujoco 版本赋值不生效)
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("obsb_"):
                self.model.body_pos[i] = np.array([30.0, 30.0, 0.3])
                self.model.body_quat[i] = np.array([1.0, 0.0, 0.0, 0.0])
        self._obs = []
        # ★ 闭合障碍的空腔 (house 4墙 / triangle 3墙 围成的内部): 激光穿不过墙,
        #   空腔对导航等价于实体, 起/目标/路径都绝不能进去
        self._cavities = []
        if isinstance(self.n_obstacles, str):
            # "nav2*" -> 固定布局 (含子类型 nav2_lcorner/house/furniture); 数字字符串 -> 转 int
            if self.n_obstacles.startswith("nav2"):
                return self._place_nav2_layout()
            self.n_obstacles = int(self.n_obstacles) if self.n_obstacles.isdigit() else 0
        if self.n_obstacles is None or self.n_obstacles <= 0:
            return
        for k in range(min(int(self.n_obstacles), 16)):
            for _ in range(80):
                hx = float(self._rng.uniform(0.25, 0.7))
                hy = float(self._rng.uniform(0.25, 0.7))
                cx = float(self._rng.uniform(-5.0, 5.0))
                cy = float(self._rng.uniform(-5.0, 5.0))
                if abs(cx) + hx > self.ROOM_HALF - 0.6 or abs(cy) + hy > self.ROOM_HALF - 0.6:
                    continue
                if any(abs(cx - ox) < hx + oxh + 0.8 and abs(cy - oy) < hy + oyh + 0.8
                       for ox, oy, oxh, oyh, _ in self._obs):
                    continue
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obsb_{k}")
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{k}")
                if bid < 0 or gid < 0:
                    break
                self.model.body_pos[bid] = np.array([cx, cy, 0.3])
                self.model.body_quat[bid] = np.array([1.0, 0.0, 0.0, 0.0])  # 随机障碍不旋转
                self.model.geom_size[gid] = np.array([hx, hy, 0.3])
                self.model.geom_contype[gid] = 1
                self.model.geom_conaffinity[gid] = 1
                self._obs.append((cx, cy, hx, hy, 0.0))   # 随机障碍不旋转
                break

    def _shuffle_groups(self, full):
        """布局内随机摆放: 每组障碍(红房子/十字/三角/L/桌/柜/书架/消防栓)整体随机挪位,
        组内相对结构保持; 组间不重叠且在界内. 返回挪动后的 layout (与 full 同构 9 元组)"""
        gtypes = ["house", "cross", "triangle", "lcorner", "table", "cabinet", "bookshelf", "hydrant"]
        groups = {t: [b for b in full if b[5] == t] for t in gtypes}
        layout = []
        placed = []   # 已放置组的 OBB (cx, cy, hx, hy, yaw) 用于重叠检查
        for t in gtypes:
            gb = groups[t]
            if not gb:
                continue
            base_cx = float(np.mean([b[0] for b in gb]))
            base_cy = float(np.mean([b[1] for b in gb]))
            cand_final = None
            for _ in range(400):
                ncx = float(self._rng.uniform(-4.0, 4.0))
                ncy = float(self._rng.uniform(-4.0, 4.0))
                dx, dy = ncx - base_cx, ncy - base_cy
                cand = []
                ok = True
                for b in gb:
                    cx, cy, hx, hy, yaw, tt, h, c, vis = b
                    wx, wy = cx + dx, cy + dy
                    if abs(wx) > self.ROOM_HALF - 0.5 or abs(wy) > self.ROOM_HALF - 0.5:
                        ok = False
                        break
                    cand.append((wx, wy, hx, hy, yaw, tt, h, c, vis))
                if not ok:
                    continue
                # 与已放置障碍重叠检查 (角点法)
                if any(self._obb_hits(wx, wy, hx, hy, yaw, px, py, phx, phy, pyaw, gap=0.9)
                       for wx, wy, hx, hy, yaw, *_ in cand
                       for px, py, phx, phy, pyaw in placed):
                    continue
                cand_final = cand
                break
            if cand_final is None:
                cand_final = gb   # 兜底放原位 (几乎不发生)
            layout.extend(cand_final)
            placed.extend([(b[0], b[1], b[2], b[3], b[4]) for b in cand_final])
        return layout

    @staticmethod
    def _obb_hits(cx1, cy1, hx1, hy1, yaw1, cx2, cy2, hx2, hy2, yaw2, gap=0.9):
        """两 OBB 是否重叠 (box1 任一角点落入 box2 放大 gap 内则重叠)"""
        c1, s1 = math.cos(yaw1), math.sin(yaw1)
        c2, s2 = math.cos(yaw2), math.sin(yaw2)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            lx, ly = sx * hx1, sy * hy1
            wx = cx1 + lx * c1 - ly * s1
            wy = cy1 + lx * s1 + ly * c1
            qx, qy = wx - cx2, wy - cy2
            l2x, l2y = qx * c2 + qy * s2, -qx * s2 + qy * c2
            if abs(l2x) < hx2 + gap and abs(l2y) < hy2 + gap:
                return True
        return False

    def _place_nav2_layout(self):
        """★ 官方 DRL-Transformer SquareWorld2.world 精确布局 (与 gazebo square_world2.sdf 1:1 对齐):
        - n_obstacles='nav2_lcorner': 只留 L 角 (专项学绕角)
        - n_obstacles='nav2_house'  : 只留红房子 (专项学穿门)
        - n_obstacles='nav2_furniture': 只留家具
        - n_obstacles='nav2'        : 全量布局
        数据源: SquareWorld2.world 的 wall_11/13/15-18/24-29 + cabinet/bookshelf/table/fire_hydrant 绝对坐标
        (cx, cy, hx, hy, yaw, type, height, color, visual_only)
        - 桌: 桌面在激光/车高上方 (gazebo 桌面 z=1.0, 激光 z=0.30 打不到, 车可钻过) -> 桌面仅可视化
          桌腿 r=0.02 圆柱 -> 小 OBB 0.02, 参与激光/碰撞 (与 gazebo 一致: 激光能扫到腿)
        """
        def _B(cx, cy, hx, hy, yaw, t, h, c, vis=False):
            return (cx, cy, hx, hy, yaw, t, h, c, vis)

        full = [
            # ===== 红房子 4 墙 (Bricks 红砖, 空心矩形, 内部 ~1.0x0.77m) =====
            _B(2.724,   3.029,   0.63395, 0.075, 0.0,       "house", 0.5, (0.72, 0.36, 0.30)),
            _B(3.2915,  2.6215,  0.48250, 0.075, -1.54994,  "house", 0.5, (0.72, 0.36, 0.30)),
            _B(2.75,    2.214,   0.62500, 0.075, 3.14159,   "house", 0.5, (0.72, 0.36, 0.30)),
            _B(2.1825,  2.6215,  0.48250, 0.075, 1.61372,   "house", 0.5, (0.72, 0.36, 0.30)),
            # ===== 十字 A 左上 (wall_11+13, 开放式十字) =====
            _B(-2.152,  1.857,   1.25000, 0.075, 0.0,       "cross", 0.5, (0.85, 0.82, 0.73)),
            _B(-2.155,  1.764,   1.37500, 0.075, -1.5708,   "cross", 0.5, (0.85, 0.82, 0.73)),
            # ===== 三角 B 左下 (wall_24+25+26 斜墙三角) =====
            _B(-2.89576, -2.49088, 0.98852, 0.075, 1.09459,  "triangle", 0.5, (0.85, 0.82, 0.73)),
            _B(-1.97077, -2.49088, 1.03178, 0.075, -1.01325, "triangle", 0.5, (0.85, 0.82, 0.73)),
            _B(-2.38953, -3.28527, 1.00000, 0.075, 3.14159,  "triangle", 0.5, (0.85, 0.82, 0.73)),
            # ===== L 角 C 右下 (wall_28+29) =====
            _B(2.38813, -2.48047, 0.87500, 0.075, 0.0,       "lcorner", 0.5, (0.85, 0.82, 0.73)),
            _B(3.20562, -1.28798, 1.25000, 0.075, 1.5708,    "lcorner", 0.5, (0.85, 0.82, 0.73)),
            # ===== 家具 (细分类型便于 shuffle 按组移动: table 桌面+4腿整体/cabinet/bookshelf/hydrant) =====
            _B(4.98173, 5.02016,  0.225, 0.225, 0.0, "cabinet", 0.51, (0.55, 0.35, 0.20)),   # cabinet 0.45x0.45
            _B(4.51394, -3.63417, 0.450, 0.200, 0.0, "bookshelf", 0.60, (0.50, 0.40, 0.20)),   # bookshelf 0.9x0.4
            # 桌面 1.5x0.8 @ z=1.0 (激光/车高上方, 仅可视化, 不参与碰撞/激光)
            _B(-4.48069, -0.04466, 0.750, 0.400, 0.0, "table", 0.03, (0.45, 0.40, 0.30), True),
            # 4 条桌腿 r=0.02 @ (±0.68, ±0.38) 相对桌面中心 (参与激光/碰撞, 对齐 gazebo 圆柱腿)
            _B(-3.80069, 0.33534, 0.02, 0.02, 0.0, "table", 0.5, (0.25, 0.25, 0.25)),
            _B(-3.80069, -0.42466, 0.02, 0.02, 0.0, "table", 0.5, (0.25, 0.25, 0.25)),
            _B(-5.16069, 0.33534, 0.02, 0.02, 0.0, "table", 0.5, (0.25, 0.25, 0.25)),
            _B(-5.16069, -0.42466, 0.02, 0.02, 0.0, "table", 0.5, (0.25, 0.25, 0.25)),
            # 消防栓 (圆柱 r=0.15, 用 box 近似)
            _B(-3.75684, 3.84985,  0.150, 0.150, 0.0, "hydrant", 0.40, (0.80, 0.10, 0.10)),
        ]
        # 按子布局类型过滤 (每个 = 官方布局中一组障碍: 十字/三角/L/红房子/家具)
        keep = {
            "nav2": "all",
            "nav2_cross": "cross",
            "nav2_triangle": "triangle",
            "nav2_lcorner": "lcorner",
            "nav2_house": "house",
            "nav2_furniture": ("table", "cabinet", "bookshelf", "hydrant"),
        }
        # 渐进组合: nav2_mixN = 前 N 组障碍 (N=1..5), 组顺序: 红房子->十字->三角->L->家具 (组内全放)
        group_order = ["house", "cross", "triangle", "lcorner", "table", "cabinet", "bookshelf", "hydrant"]
        # nav2_seqN = 按官方列表顺序取前 N 个障碍 (每次只加 1 个, 平滑渐进)
        if self.n_obstacles == "nav2_shuffle":
            # ★ 布局内随机摆放: 保持 gz 障碍类型/数量/朝向, 但每次 reset 按组随机挪位
            # 让策略学"通用避障"而非"背地图", 泛化到 gz 成功率更高
            layout = self._shuffle_groups(full)
        elif self.n_obstacles and isinstance(self.n_obstacles, str) and self.n_obstacles.startswith("nav2_seq"):
            n_seq = int(self.n_obstacles.split("nav2_seq")[1])
            layout = full[:n_seq]
        elif self.n_obstacles and isinstance(self.n_obstacles, str) and self.n_obstacles.startswith("nav2_mix"):
            n_mix = int(self.n_obstacles.split("nav2_mix")[1])
            want = group_order[:n_mix]
            layout = [b for b in full if b[5] in want]
        else:
            want = keep.get(self.n_obstacles, "all")
            if isinstance(want, tuple):
                layout = [b for b in full if b[5] in want]
            else:
                layout = [b for b in full if want == "all" or b[5] == want]
        # ★ 红房子专项 (nav2_house): 把红房子整体平移到地图中央 — 用户要求
        # "红房子放中间, UGV 起点与终点在红房子两侧, 必须绕过去".
        # 平移只改位置不改朝向/尺寸, 专项训练每局起点+目标都在红房子两侧必绕
        if self.n_obstacles == "nav2_house" and layout:
            cx0 = float(np.mean([b[0] for b in layout]))
            cy0 = float(np.mean([b[1] for b in layout]))
            layout = [(b[0] - cx0, b[1] - cy0, b[2], b[3], b[4], b[5], b[6], b[7], b[8]) for b in layout]
        # ★ 闭合障碍空腔: 同类型的墙 (house 4面 / triangle 3面) 首尾相接围成内部空腔,
        #   激光穿不过墙 -> 空腔等价实体. 用各墙中心连线多边形近似空腔 (墙厚 0.075,
        #   中心连线已足够保守). 其余类型 (cross 十字/lcorner L) 不闭合, 无空腔
        self._cavities = []
        cavity_pts = {}
        for b in layout:
            t = b[5]
            if t in ("house", "triangle"):
                cavity_pts.setdefault(t, []).append((b[0], b[1]))
        for t, pts in cavity_pts.items():
            if len(pts) < 3:
                continue
            # 绕质心按角度排序, 保证多边形顶点顺序 (凸包近似)
            cx0 = sum(p[0] for p in pts) / len(pts)
            cy0 = sum(p[1] for p in pts) / len(pts)
            pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy0, p[0] - cx0))
            self._cavities.append(pts)
        for used, b in enumerate(layout):
            cx, cy, hx, hy, yaw, t, h, c, vis = b
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obsb_{used}")
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{used}")
            if bid < 0 or gid < 0:
                break
            # 位置/旋转改在 body 上 (geom_quat 运行时赋值在此 mujoco 版本不生效, body_quat 生效)
            self.model.body_pos[bid] = np.array([cx, cy, 1.0 if vis else h])
            self.model.body_quat[bid] = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
            self.model.geom_size[gid] = [hx, hy, h]
            self.model.geom_rgba[gid] = np.array([*c, 1.0])
            if vis:
                # 纯可视化 (桌面): 不参与真实物理碰撞/接触
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0
                continue
            self.model.geom_contype[gid] = 1
            self.model.geom_conaffinity[gid] = 1
            self._obs.append((cx, cy, hx, hy, yaw))

    def _point_in_cavity(self, px, py, margin=0.0):
        """点是否落在任一闭合障碍 (红房子/三角) 的空腔内部. 激光穿不过墙,
        空腔对导航等价实体 -> 起/目标/路径/检测都绝不能进入"""
        for poly in self._cavities:
            if self._point_in_poly(px, py, poly, margin):
                return True
        return False

    @staticmethod
    def _point_in_poly(px, py, poly, margin=0.0):
        """点在多边形内判定 (射线法). margin>0 时外扩: 把每条边向外推 margin,
        用"点到边距离"近似, 这里用简单实现: 判断点到多边形最近边距离 < margin 也算内"""
        # 射线法 (含边界)
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > py) != (y2 > py):
                xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if px < xint:
                    inside = not inside
        if inside or margin <= 0:
            return inside
        # 外扩: 点到任意边距离 < margin 视为内
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            vx, vy = x2 - x1, y2 - y1
            seg = math.hypot(vx, vy)
            if seg < 1e-9:
                continue
            t = ((px - x1) * vx + (py - y1) * vy) / (seg * seg)
            t = max(0.0, min(1.0, t))
            cx = x1 + t * vx
            cy = y1 + t * vy
            if math.hypot(px - cx, py - cy) < margin:
                return True
        return False

    def _point_free(self, x, y, margin=0.55):
        if abs(x) > self.ROOM_HALF - margin or abs(y) > self.ROOM_HALF - margin:
            return False
        # ★ 空腔内部 = 实体: 起/目标绝不能落在红房子/三角空腔内 (进了就出不来)
        if self._point_in_cavity(x, y, margin=max(0.0, margin - 0.30)):
            return False
        # OBB 碰撞检测: 点变换到 box 局部系, 落在半边长+margin 之外才算 free (支持斜墙)
        for cx, cy, hx, hy, yaw in self._obs:
            cosy, siny = math.cos(yaw), math.sin(yaw)
            qx, qy = x - cx, y - cy
            lx = qx * cosy + qy * siny
            ly = -qx * siny + qy * cosy
            if abs(lx) < hx + margin and abs(ly) < hy + margin:
                return False
        return True

    def _goal_reachable(self, gx, gy):
        """目标可达: 目标点周围 0.85m 内无障碍 (车能靠近并停车), 避免被围死.
        否则全量布局 43% 目标在封闭区, 到达率上限只有 ~57%, 永远练不到 0.7"""
        # 空腔内部/开口处不可达
        if self._point_in_cavity(gx, gy, margin=0.30):
            return False
        for cx, cy, hx, hy, yaw in self._obs:
            cosy, siny = math.cos(yaw), math.sin(yaw)
            qx, qy = gx - cx, gy - cy
            lx = qx * cosy + qy * siny
            ly = -qx * siny + qy * cosy
            if abs(lx) < hx + 0.85 and abs(ly) < hy + 0.85:
                return False
        return True

    def _line_blocked(self, sx, sy, gx, gy, margin=0.30):
        """起点->目标 直线是否被任一障碍遮挡 (需要绕路才到). 全量布局 ~57% 被挡"""
        d = math.hypot(gx - sx, gy - sy)
        n = max(2, int(d / 0.05))
        for k in range(1, n):
            px, py = sx + (gx - sx) * k / n, sy + (gy - sy) * k / n
            # 穿过空腔也算被挡 (激光穿不过墙)
            if self._point_in_cavity(px, py, margin=0.10):
                return True
            for cx, cy, hx, hy, yaw in self._obs:
                cosy, siny = math.cos(yaw), math.sin(yaw)
                qx, qy = px - cx, py - cy
                lx = qx * cosy + qy * siny
                ly = -qx * siny + qy * cosy
                if abs(lx) < hx + margin and abs(ly) < hy + margin:
                    return True
        return False

    def _sample_detour(self):
        """★ 强制绕障局 (不随机碰运气): 主动选一个障碍, 起点放在其一侧,
        目标放在其正后方另一侧 (起点-障碍中心-目标近似共线) -> 起点->目标
        直线必然穿过障碍中心, 100% 必须绕行. 优先绕大障碍(红房子墙/长墙),
        避免选桌腿 0.02m 小障碍. 红房子专项 (nav2_house) 时红房子在中间,
        起点/目标在两侧, 正是用户要的"必须绕红房子"."""
        if not self._obs:
            return None
        # 按面积降序: 优先绕红房子/十字/三角/L 这种大障碍
        obs = sorted(self._obs, key=lambda b: b[2] * b[3], reverse=True)
        for cx, cy, hx, hy, yaw in obs[:6]:
            hmax = max(hx, hy)
            for _ in range(120):
                th = float(self._rng.uniform(-math.pi, math.pi))
                # 起点/目标距障碍中心 = 障碍半长 + 0.9~1.6m (保证墙外起步, 且目标在
                # 障碍后方 ~1m, _goal_reachable 0.85m 恰好通过)
                r1 = hmax + float(self._rng.uniform(0.9, 1.6))
                r2 = hmax + float(self._rng.uniform(0.9, 1.6))
                sx, sy = cx + r1 * math.cos(th), cy + r1 * math.sin(th)
                gx, gy = cx - r2 * math.cos(th), cy - r2 * math.sin(th)
                d = math.hypot(gx - sx, gy - sy)
                lo, hi = self.goal_dist
                if d < lo * 0.7 or d > hi * 2.2:
                    continue   # 距离太近/太远都不好练
                if not (self._point_free(sx, sy, margin=0.55) and
                        self._point_free(gx, gy, margin=0.55)):
                    continue
                if not self._goal_reachable(gx, gy):
                    continue
                # ★ 起点/目标必须在所有障碍外部 (闭合障碍如红房子内是死胡同,
                #   车一出生就卡死, 永远出不来). _point_free 只查距墙>0.35,
                #   不查"是否在闭合障碍内部", 这里用 OBB 点内判定补上
                if self._point_in_any_obb(sx, sy) or self._point_in_any_obb(gx, gy):
                    continue
                # 直线必然穿过障碍中心 (o 在 start-goal 线段上), 无需再 _line_blocked
                return np.array([sx, sy], np.float32), np.array([gx, gy], np.float32)
        return None

    def _point_in_any_obb(self, px, py, margin=0.05):
        """点是否在任一障碍 OBB 内部 (闭合障碍内侧是死胡同, 起/目标不能放里面)"""
        # 空腔内部 = 实体
        if self._point_in_cavity(px, py, margin=margin):
            return True
        for cx, cy, hx, hy, yaw in self._obs:
            cosy, siny = math.cos(yaw), math.sin(yaw)
            qx, qy = px - cx, py - cy
            lx = qx * cosy + qy * siny
            ly = -qx * siny + qy * cosy
            if abs(lx) < hx + margin and abs(ly) < hy + margin:
                return True
        return False

    def _sample_spawn(self):
        """课程采样: 起点随机(界内靠中), 目标在 goal_dist 距离内, 均避开障碍.
        目标必须可达 (周围 0.85m 无障碍, 否则被围死练不到).
        detour_ratio 比例的局: 主动把目标放在障碍正后方 (起点-障碍-目标共线,
        直线被挡, 必须绕路) — 不再随机碰运气 (旧版 0.95 也只有 37% 绕障,
        大量"直线即到"的无效局导致到达率虚高, gz 部署(必须绕)就崩)."""
        for _ in range(400):
            need_detour = self._rng.random() < self.detour_ratio
            if need_detour:
                pair = self._sample_detour()
                if pair is not None:
                    return pair
                # 该障碍方向采样失败 -> 落回自由采样 (下一轮继续尝试)
            start = np.array([self._rng.uniform(-3.0, 3.0), self._rng.uniform(-3.0, 3.0)])
            if not self._point_free(start[0], start[1], margin=0.55):
                continue
            if self._point_in_any_obb(start[0], start[1]):
                continue   # 起点不能落进闭合障碍 (红房子) 内部
            r = float(self._rng.uniform(*self.goal_dist))
            th = float(self._rng.uniform(-math.pi, math.pi))
            goal = start + np.array([r * math.cos(th), r * math.sin(th)])
            if abs(goal[0]) > self.ROOM_HALF - 0.6 or abs(goal[1]) > self.ROOM_HALF - 0.6:
                continue
            if not (self._point_free(goal[0], goal[1], margin=self.goal_margin) and
                    self._goal_reachable(goal[0], goal[1])):
                continue
            if self._point_in_any_obb(goal[0], goal[1]):
                continue
            if need_detour and not self._line_blocked(start[0], start[1], goal[0], goal[1]):
                continue   # 要求绕路但这条直线没被挡 -> 重采样
            return start.astype(np.float32), goal.astype(np.float32)
        # 兜底: 原地 1m 直线目标
        return np.array([0.0, 0.0], np.float32), np.array([1.0, 0.0], np.float32)

    # ============ OBB (旋转 box) 几何工具 ============

    @staticmethod
    def _ray_obb(px, py, dx, dy, cx, cy, hx, hy, yaw):
        """射线(起点p, 方向d单位向量) vs 旋转 box (中心c, 半长hx/hy, 偏航yaw) 求交.
        返回射线 t (距离), None=不相交. 变换到 OBB 局部系做 AABB 求交. 支持斜墙"""
        cosy, siny = math.cos(yaw), math.sin(yaw)
        # 逆旋转 + 平移到局部系
        qx, qy = px - cx, py - cy
        lx = qx * cosy + qy * siny
        ly = -qx * siny + qy * cosy
        ldx = dx * cosy + dy * siny
        ldy = -dx * siny + dy * cosy
        tmin, tmax = 0.0, 1e9
        for d, p, lo, hi in ((ldx, lx, -hx, hx), (ldy, ly, -hy, hy)):
            if abs(d) < 1e-9:
                if p < lo or p > hi:
                    return None
            else:
                t1, t2 = (lo - p) / d, (hi - p) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return None
        return tmin

    # ============ 真实物理碰撞 (对齐 gazebo) ============

    def _is_collision(self):
        """mj_forward 后检查接触对: 只要 base_link 与墙/障碍(wall_*/obs_*)接触即真撞.
        忽略轮子/地面/自身接触 (车正常行驶时轮子触地不算碰撞)"""
        if self.data.ncon == 0:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            # base_link 与 墙/障碍 的接触
            if g1 == "base_link" and (g2.startswith("wall_") or g2.startswith("obs_")):
                return True
            if g2 == "base_link" and (g1.startswith("wall_") or g1.startswith("obs_")):
                return True
        return False

    # ============ 传感器 ============

    def _get_yaw(self):
        return self._yaw

    def _get_ranges(self):
        """90 束射线 vs OBB (旋转 box, 支持 gazebo 斜墙) 精确相交 + X2N 噪声 (sim2real); 扫墙+障碍"""
        pos = self.data.qpos[0:2]
        yaw = self._get_yaw()
        ranges = np.full(self.N_RAYS, self.RAY_RANGE, dtype=np.float32)
        for i in range(self.N_RAYS):
            ang = yaw + i * 2 * math.pi / self.N_RAYS
            dx, dy = math.cos(ang), math.sin(ang)
            best = self.RAY_RANGE
            for box in self._walls + self._obs:
                t = self._ray_obb(pos[0], pos[1], dx, dy, *box)
                if t is not None and t > 1e-6 and t < best:
                    best = t
            ranges[i] = best
        ranges += self._rng.normal(0.0, 0.02, self.N_RAYS)
        drop = self._rng.random(self.N_RAYS) < 0.05
        ranges = np.where(drop, self.RAY_RANGE, ranges)
        return np.clip(ranges, 0.01, self.RAY_RANGE)

    def _render_gray(self):
        """128x160 鱼眼灰度图 uint8"""
        return self._renderer.render_gray(self.data)

    # ============ V-PRM 老师 (生成演示: 朝路径引导点前进) ============

    def _get_waypoint(self):
        """返回路径引导点 (世界坐标): 取 A* 绕障路径上距车 ~1.0m 的前方点.
        path=None (空地图/规划失败) 时返回真实目标 -> 直线引导, 与旧版一致.
        奖励基于"朝引导点接近" -> 绕行不再被距离惩罚 (修复 cost 设计错误:
        旧版 r_heuristic 用目标距离, 绕行第一步必然远离目标每步 -20, 策略学到
        直线冲墙->墙边徘徊, 到达率 0)"""
        pos = self.data.qpos[0:2]
        if self._path is None or len(self._path) < 2:
            return self.goal.copy()
        # A* 返回的是 tuple 点列, 统一转 ndarray (避免 tuple-tuple 减法)
        path = [np.asarray(p, dtype=float) for p in self._path]
        # 路径上距车最近点
        best, best_d = 0, 1e9
        for i, p in enumerate(path):
            d = float(np.linalg.norm(p - pos))
            if d < best_d:
                best_d, best = d, i
        # 从最近点累加段长, 取 ~LOOKAHEAD 处 (插值), 兜底终点
        LOOK = 1.0
        target = path[-1]
        acc = 0.0
        for i in range(best, len(path) - 1):
            seg = float(np.linalg.norm(path[i + 1] - path[i]))
            if acc + seg >= LOOK:
                t = (LOOK - acc) / seg if seg > 1e-9 else 0.0
                target = path[i] * (1 - t) + path[i + 1] * t
                break
            acc += seg
        return np.array(target, dtype=float)

    def vprm_demo_path(self, start=None):
        """全局路径: A* 大膨胀 + 带障碍斥力的平滑 (生成尽量远离障碍的可跟踪轨迹).
        用户思路: 全局膨胀大一点让路径尽可能避开障碍, 平滑轨迹也带避障,
        后续局部控制按动力学跟随即可."""
        occ = set()
        PATH_CELL, PATH_BOUND = 0.2, 5.3   # 墙在 ±5.425, 路径规划界内
        INFLATE = 0.75   # ★ 全局大膨胀: 路径离障碍最小间隙 0.75m (碰撞0.28+车0.10+余量),
                         #   让全局路径本身就远离障碍, 后续跟踪更安全
        # 精确栅格化: 每个栅格点用 OBB 距离判定 (支持斜墙)
        obstacles = self._walls + self._obs
        gmin, gmax = int(-PATH_BOUND / PATH_CELL), int(PATH_BOUND / PATH_CELL)
        for gx in range(gmin, gmax + 1):
            for gy in range(gmin, gmax + 1):
                px, py = gx * PATH_CELL, gy * PATH_CELL
                # ★ 空腔内部 = 实体: 红房子/三角空腔栅格直接占用, A* 绝不穿越
                if self._point_in_cavity(px, py, margin=0.30):
                    occ.add((gx, gy))
                    continue
                for cx, cy, hx, hy, yaw in obstacles:
                    cosy, siny = math.cos(yaw), math.sin(yaw)
                    qx, qy = px - cx, py - cy
                    lx = qx * cosy + qy * siny
                    ly = -qx * siny + qy * cosy
                    dx = max(abs(lx) - hx, 0.0)
                    dy = max(abs(ly) - hy, 0.0)
                    if math.hypot(dx, dy) < INFLATE:
                        occ.add((gx, gy))
                        break
        # ★ start 必须显式传入 (reset 时 qpos 尚未赋成 start, 直接用会从错误起点规划)
        pos = np.asarray(start, dtype=float) if start is not None else self.data.qpos[0:2].copy()
        goal = self.goal.copy()
        # ★ 用网格 A* (occ 已按 OBB 精确膨胀, astar 内不再二次膨胀)
        path = astar_grid_plan(occ, tuple(pos), tuple(goal),
                               path_cell=PATH_CELL, path_bound=PATH_BOUND, inflate=0.0)
        if path is None or len(path) < 2:
            return path
        # ★ 插值加密到 ~0.2m 间隔 (纯追踪不切弯)
        dense = [np.asarray(path[0], dtype=float)]
        for i in range(len(path) - 1):
            a = np.asarray(path[i], dtype=float)
            b = np.asarray(path[i + 1], dtype=float)
            seg = float(np.linalg.norm(b - a))
            n = max(1, int(round(seg / 0.2)))
            for k in range(1, n + 1):
                dense.append(a + (b - a) * (k / n))
        # ★ 平滑 + 障碍斥力: 迭代把靠近障碍的轨迹点往开阔方向推 (保持端点/拓扑不变)
        #   这样平滑轨迹本身就远离障碍, 局部控制器跟上去更安全
        pts = dense
        for _ in range(12):
            moved = False
            for i in range(1, len(pts) - 1):
                p = pts[i]
                dmin, dcx, dcy = 1e9, 0.0, 0.0
                for cx, cy, hx, hy, yaw in obstacles:
                    d = self._obb_dist(p[0], p[1], cx, cy, hx, hy, yaw)
                    if d < dmin:
                        dmin, dcx, dcy = d, cx, cy
                if dmin < 0.9 and dmin > 0.05:
                    # ★ 推离方向: 直接沿 点->障碍中心 向量 (永远远离中心, 简单可靠).
                    #   旧版用 OBB 外法向, 点在角落/障碍内时方向错/符号反 -> 往墙里推
                    vec = np.array([p[0] - dcx, p[1] - dcy])
                    norm = float(np.linalg.norm(vec))
                    if norm > 1e-6:
                        push = min(0.9 - dmin, 0.15)
                        nw = p + vec / norm * max(push, 0.0)
                        # 平滑约束: 不过分远离邻居, 保持路径形状
                        nw = 0.5 * nw + 0.25 * np.array(pts[i - 1]) + 0.25 * np.array(pts[i + 1])
                        pts[i] = nw
                        moved = True
            if not moved:
                break
        # ★ 强制连接终点: A* 大膨胀(0.75) 下 goal 附近可能被膨胀圈覆盖, 路径终点
        #   停在离 goal 最近自由格, 不连终点 -> 直接 append goal 点, 保证连到终点
        if len(pts) >= 2:
            if float(np.linalg.norm(np.asarray(pts[-1]) - goal)) > 0.25:
                pts.append(np.asarray(goal, dtype=float))
        return [np.asarray(p, dtype=float) for p in pts]

    def _obb_dist(self, px, py, cx, cy, hx, hy, yaw):
        """点->OBB(旋转 box) 最近距离 (局部系变换, 支持斜墙)"""
        cosy, siny = math.cos(yaw), math.sin(yaw)
        qx, qy = px - cx, py - cy
        lx = qx * cosy + qy * siny
        ly = -qx * siny + qy * cosy
        dx = max(abs(lx) - hx, 0.0)
        dy = max(abs(ly) - hy, 0.0)
        return math.hypot(dx, dy)

    def _dwa_action(self, path, lookahead=0.5):
        """LOS/lookahead 局部跟踪 (用户: 全局大膨胀+平滑避障后, 局部按动力学跟上):
        1. 找平滑轨迹上距车最近点
        2. 沿轨迹弧长取 lookahead 前方点作为引导目标
        3. 朝引导点转向 (角速度由角度偏差 PID 式给出), 线速度偏差大时减速
        4. 前方过近 (激光 < 安全) 才做最小规避 (减速/打方向), 不主动偏离轨迹
        """
        pos = self.data.qpos[0:2]
        yaw = self._get_yaw()
        # ★ 距真实目标 < 1.2m: 直接朝目标全速 (不再沿 path, 避免到终点附近
        #   卡在 path[-1] 与 GOAL_RADIUS 之间的盲区. 终点连线覆盖最后一段)
        d_goal = float(np.linalg.norm(self.goal - pos))
        if d_goal < 1.2:
            target = self.goal
        elif path is not None and len(path) >= 2:
            path = [np.array(p, dtype=float) for p in path]
            # 最近点索引
            best_i = 0
            for i in range(len(path)):
                if np.linalg.norm(path[i] - pos) < np.linalg.norm(path[best_i] - pos):
                    best_i = i
            # 沿路径弧长取 lookahead 前方点
            target = path[-1]
            acc = 0.0
            for i in range(best_i, len(path) - 1):
                seg = float(np.linalg.norm(path[i + 1] - path[i]))
                if acc + seg >= lookahead:
                    t = (lookahead - acc) / seg if seg > 1e-9 else 0.0
                    target = path[i] * (1 - t) + path[i + 1] * t
                    break
                acc += seg
                target = path[i + 1]
            target = np.array(target)
        else:
            target = self.goal
        goal_ang = math.atan2(target[1] - pos[1], target[0] - pos[0]) - yaw
        goal_ang = (goal_ang + math.pi) % (2 * math.pi) - math.pi
        beta = goal_ang / math.pi                       # [-1,1]
        # ★ 提速: 偏差减速系数 0.6->0.35 (以前偏差大就大幅减速, 平均速度低
        #   800 步到不了. 现在保持前进, 转向交给 a1)
        a1 = float(np.clip(beta * 1.6, -1.0, 1.0))
        a0 = float(np.clip(1.0 - abs(beta) * 0.35, 0.0, 1.0))
        # ★ 到终点 0.8m 内: 全速直冲 (不减速), 直接连接终点
        if d_goal < 0.8:
            a0 = 1.0
        # ★ 前方过近: 障碍在安全圈内, 必须处理.
        #   若两侧都近(墙角/窄口) -> 原地转向到较开阔侧 (a0=0, 不打滑磨),
        #   避免"减速顶墙慢慢磨" (ep8/10/12 卡在红房子墙角就是这问题)
        ranges = self._get_ranges()
        N = self.N_RAYS
        front = float(np.min(np.concatenate([ranges[N // 2 - 2:N // 2 + 3], ranges[:2], ranges[-2:]])))
        if front < 0.40:
            left = float(np.min(ranges[N * 3 // 4:N * 7 // 8]))
            right = float(np.min(ranges[N // 8:N * 1 // 4]))
            if left > right and left > 0.45:
                a1 = 0.7; a0 = 0.2
            elif right > left and right > 0.45:
                a1 = -0.7; a0 = 0.2
            else:
                # 两侧都堵 = 被墙角夹住: 原地转永远出不来 (转到哪都近),
                # ★ 必须倒车脱困: 边倒边转向较开阔侧 (ep8/10/12 卡死根因)
                a0 = -0.5
                a1 = 0.8 if left >= right else -0.8
        return np.array([a0, a1], dtype=np.float32)

    def _follow_path_action(self, path, lookahead=0.5):
        """老师跟踪: 无误差按路径点直冲 (TEB 思路, 供演示).
        - 路径点足够密 (A* 已插值 0.2m), 不需要花哨的自适应前瞻/切弯
        - 目标角度温和映射, 不原地打转; 偏差大减速
        - 保持与 path=None (go-to-goal) 一致结构, 教师/演示共用
        """
        pos = self.data.qpos[0:2]
        yaw = self._get_yaw()
        if path is not None and len(path) >= 2:
            path = [np.array(p, dtype=float) for p in path]
            # 取路径上距车 lookahead 前方点 (沿路径累加距离)
            target = path[-1]
            acc = 0.0
            # 先找距车最近点索引
            best_i = 0
            for i in range(len(path)):
                if np.linalg.norm(path[i] - pos) < np.linalg.norm(path[best_i] - pos):
                    best_i = i
            for i in range(best_i, len(path) - 1):
                seg = float(np.linalg.norm(path[i + 1] - path[i]))
                if acc + seg >= lookahead:
                    t = (lookahead - acc) / seg if seg > 1e-9 else 0.0
                    target = path[i] * (1 - t) + path[i + 1] * t
                    break
                acc += seg
                target = path[i + 1]
            target = np.array(target)
        else:
            target = self.goal
        goal_ang = math.atan2(target[1] - pos[1], target[0] - pos[0]) - yaw
        goal_ang = (goal_ang + math.pi) % (2 * math.pi) - math.pi
        beta = goal_ang / math.pi                       # [-1, 1]
        a1 = float(np.clip(beta * 1.5, -1.0, 1.0))      # 温和转向, 不原地打转
        a0 = float(np.clip(1.0 - abs(beta) * 0.6, 0.0, 1.0))  # 偏差大减速
        return np.array([a0, a1], dtype=np.float32)

    def demo_rollout_vprm(self):
        """V-PRM 老师跑一条完整轨迹, 返回 [(obs, act)] 演示"""
        path = self.vprm_demo_path()
        if path is None or len(path) < 2:
            return []
        demos = []
        for _ in range(self.MAX_STEPS):
            act = self._follow_path_action(path)
            demos.append((self._get_obs(), act))
            _, _, term, trunc, _ = self.step(act)
            if term:
                break
        return demos

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._place_obstacles()
        start, self.goal = self._sample_spawn()
        self._path = self.vprm_demo_path(start=start)   # ★ 全局绕障路径 (A*), 空地图=None -> 直线

        self.data.qpos[0:2] = start
        # ★ 初始朝向: 沿 A* 路径首段方向 (出生就朝绕行方向, 避免随机朝向
        #   导致 LOS 边转边走切弯钻墙 — ep8/10/12 卡死根因). 无路径=朝目标
        if self._path is not None and len(self._path) >= 2:
            p0 = np.asarray(self._path[0], dtype=float)
            p1 = np.asarray(self._path[1], dtype=float)
            yaw0 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        else:
            yaw0 = math.atan2(self.goal[1] - start[1], self.goal[0] - start[0])
        self._yaw = yaw0
        self.data.qpos[3:7] = np.array([math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)])
        self.data.qpos[7:] = 0.0
        self.data.qvel[:] = 0.0
        # 更新目标标记位置 (金色球, GUI 可视化)
        gm = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")
        if gm >= 0:
            self.model.geom_pos[gm] = np.array([self.goal[0], self.goal[1], 0.4])
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self._collision = False
        self._arrived = False
        self._prev_dist = float(np.linalg.norm(start - self.goal))
        self._prev_wp_dist = float(np.linalg.norm(self._get_waypoint() - start))
        self._prev_act = np.zeros(2, dtype=np.float32)
        self._pos_trace.clear()
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self._frames.clear()
        if self.obs_mode != "laser":
            g = self._render_gray() / 255.0
            for _ in range(self.FRAME_STACK):
                self._frames.append(g)
        return self._get_obs(), {}

    def _get_obs(self):
        pos = self.data.qpos[0:2]
        dist = float(np.linalg.norm(pos - self.goal))
        beta = math.atan2(self.goal[1] - pos[1], self.goal[0] - pos[0]) - self._get_yaw()
        beta = (beta + math.pi) % (2 * math.pi) - math.pi
        ranges = self._get_ranges()
        laser = (ranges / self.RAY_RANGE).astype(np.float32)  # 90 束归一化
        if self.obs_mode == "laser":
            # [激光(90), dist/15, beta/pi, lin/MAX, ang/MAX] 速度信息帮助策略感知自身运动
            return np.concatenate([laser,
                                   np.array([dist / 15.0, beta / math.pi,
                                             float(np.clip(self.lin_vel / self.MAX_LIN, -1, 1)),
                                             float(np.clip(self.ang_vel / self.MAX_ANG, -1, 1))],
                                            np.float32)]).astype(np.float32)
        img = np.stack(self._frames, axis=0).astype(np.float32)  # (4,128,160)
        # pstate: goal_dim=2 时只含 [Dist/15, beta/pi] (与 BC 预训练一致)
        goal = np.array([dist / 15.0, beta / math.pi], dtype=np.float32)
        if self.goal_dim >= 4:
            min_scan = float(ranges.min())
            fwd = ranges[:int(self.N_RAYS * 1 / 6)]
            goal = np.concatenate([goal, [min_scan / self.RAY_RANGE,
                                          float(fwd.mean()) / self.RAY_RANGE]]).astype(np.float32)
        return {"image": img, "goal": goal, "laser": laser}

    def step(self, action):
        a0, a1 = float(action[0]), float(action[1])
        # ★ 动作映射: lin=a0*MAX_LIN 允许倒车 (贴墙可脱困), ang=a1*MAX_ANG
        self.lin_vel = float(np.clip(a0 * self.MAX_LIN, -self.MAX_LIN, self.MAX_LIN))
        self.ang_vel = float(np.clip(a1 * self.MAX_ANG, -self.MAX_ANG, self.MAX_ANG))
        lin, ang = self.lin_vel, self.ang_vel
        pos = self.data.qpos[0:2]
        yaw = self._get_yaw()
        dt = self.DT
        self.data.qpos[0] += lin * math.cos(yaw) * dt
        self.data.qpos[1] += lin * math.sin(yaw) * dt
        # ★ 墙碰撞: 位置 clamp 在墙内 (车被墙顶住不穿墙), 激光持续扫到墙 -> 策略被迫学脱困
        wall_lim = self.ROOM_HALF - 0.25   # 5.15 (墙内表面 5.275 - 车半径 0.1)
        before_wall = self.data.qpos[0:2].copy()
        self.data.qpos[0] = np.clip(self.data.qpos[0], -wall_lim, wall_lim)
        self.data.qpos[1] = np.clip(self.data.qpos[1], -wall_lim, wall_lim)
        # ★ 障碍碰撞 clamp (OBB, 支持斜墙): 车被顶在障碍外 R=0.30 处 (激光<0.28 就物理过不去 = 必须绕行).
        #   对齐 gazebo 真碰撞: 车体半径 0.10 + 安全余量 0.20, 激光 0.28 即被顶住 -> 策略学"绕"而非"挤"
        px, py = float(self.data.qpos[0]), float(self.data.qpos[1])
        R = 0.30
        for cx, cy, hx, hy, yaw in self._obs:
            cosy, siny = math.cos(yaw), math.sin(yaw)
            # 变换到 box 局部系
            qx, qy = px - cx, py - cy
            lx = qx * cosy + qy * siny
            ly = -qx * siny + qy * cosy
            hx2, hy2 = hx + R, hy + R
            nx, ny = max(-hx2, min(lx, hx2)), max(-hy2, min(ly, hy2))   # 外扩后最近点
            ddx, ddy = lx - nx, ly - ny
            if ddx * ddx + ddy * ddy < 1e-12:
                # 圆心在 box 内 (外扩后) -> 沿最小穿透方向推出
                ovl = (hx2 - abs(lx), hy2 - abs(ly))
                if ovl[0] < ovl[1]:
                    nx = hx2 if lx > 0 else -hx2
                else:
                    ny = hy2 if ly > 0 else -hy2
            else:
                d = math.hypot(ddx, ddy)
                if d < R:
                    nx, ny = nx, ny   # 推到最近点 (局部系)
                else:
                    continue
            # 局部系修正位置 -> 世界系
            self.data.qpos[0] = cx + nx * cosy - ny * siny
            self.data.qpos[1] = cy + nx * siny + ny * cosy
        # ★ 真碰撞判定: 本次移动被 clamp 挡住 (想走但顶墙/顶障碍 = 撞)
        #   before_wall = 积分后"想去的位姿"; clamp 后如果被推回 -> 顶墙/顶障碍 = 真碰撞
        self._hit_wall = float(np.linalg.norm(self.data.qpos[0:2] - before_wall)) > 1e-6 and \
                         float(np.linalg.norm(lin)) > 0.01 and self.step_count >= 50
        # 标量 yaw 积分 (修复原四元数点积 bug: qpos[3:7] 被压成标量广播)
        self._yaw += ang * dt
        self.data.qpos[3:7] = np.array([math.cos(self._yaw / 2), 0, 0, math.sin(self._yaw / 2)])
        self.data.qvel[0] = lin * math.cos(yaw)
        self.data.qvel[1] = lin * math.sin(yaw)
        self.data.qvel[5] = ang
        mujoco.mj_forward(self.model, self.data)

        pos = self.data.qpos[0:2]
        dist = float(np.linalg.norm(pos - self.goal))
        ranges = self._get_ranges()
        min_laser = float(ranges.min())

        # ★ 碰撞判定: 真实物理接触 (base_link 与墙/障碍接触 = 真碰, 对齐 gazebo).
        #   用户: 障碍物边缘+安全距离就判碰撞太严格 (有误差正常), 实际界定要准确(真碰了).
        #   激光阈值只用于奖励引导 (r_safe 惩罚贴墙), 不用于终止判定,
        #   这样"接近但没真碰"的局能继续跑完, 不会在边缘被误判终止.
        self._collision = self._is_collision() and self.step_count >= 50
        self._arrived = dist < self.GOAL_RADIUS
        out_of_bounds = (abs(pos[0]) > self.ROOM_HALF or abs(pos[1]) > self.ROOM_HALF)

        # ★ 奖励 (按用户 5 条 cost 设计):
        #   1. 不直接"靠近目标"加分 (否则策略只冲目标忽略障碍) -> 用沿 A* 绕障路径的
        #      引导点接近 (r_waypoint, 权重低: 引导不主导, 靠近障碍惩罚必须更大)
        #   2. 远离雷达障碍加分, 但有上限 (2.0m 内越远越好, 再远不加分)
        #   3. 步长扣分 (小但累积高, 逼策略走捷径不走冤枉路)
        #   4. 障碍碰撞重罚 (-200, 且终止)
        #   5. 到达给大奖励 (+100)
        wp = self._get_waypoint()
        wp_dist = float(np.linalg.norm(wp - pos))
        r_waypoint = float(self._prev_wp_dist - wp_dist) * 8.0   # 1: 跟随绕障路径引导点 (低权重)
        r_action = abs(self.lin_vel) * 2.0 - abs(self.ang_vel)   # 前进激励 + 压大转角
        r_clear = min(float(min_laser), 2.0) / 2.0 * 0.3          # 2: 远离障碍加分, 2m 封顶
        # ★ 贴障碍惩罚: 0.7m 梯度惩罚区, 越近二次陡增. 碰撞惩罚(-200) > 绕行收益,
        #   策略学"绕"而非"贴着障碍走/挤过去"
        SAFE = 0.5 + 2.0 * self.ROBOT_RADIUS   # 0.7 = 0.5 空隙 + 车直径 0.20
        pen = max(0.0, SAFE - min_laser)
        r_safe = -pen * pen * 60.0                     # 0.7m边缘≈0, 0.6m≈0.6, 0.5m≈2.4, 0.3m≈9.6
        r_step = -0.05                                  # 3: 步长扣分 (小但 1800 步累积 -90)
        r_target = 100.0 if self._arrived else 0.0      # 5: 到达大奖励
        r_collision = -200.0 if self._collision else 0.0  # 4: 碰撞重罚, 撞墙即死
        r_out = -10.0 if out_of_bounds else 0.0             # 温和出界惩罚
        reward = float(r_waypoint + r_action + r_clear + r_safe + r_step + r_target + r_collision + r_out)

        self._prev_dist = dist
        self._prev_wp_dist = wp_dist
        self._prev_act = np.array([a0, a1], dtype=np.float32)
        if self.obs_mode != "laser":
            self._frames.append(self._render_gray() / 255.0)
        self.step_count += 1

        # ★ 终止: 到达 或 碰撞 (撞墙即死, -100; 策略必须学会不撞/脱困)
        terminated = bool(self._arrived or self._collision)
        truncated = self.step_count >= self.MAX_STEPS
        info = {"arrived": bool(self._arrived), "collided": bool(self._collision),
                "out_of_bounds": bool(out_of_bounds), "dist": dist}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return self._renderer.render(self.data)
