#!/usr/bin/env python
"""MuJoCo true-dynamics tracked UGV navigation environment.

This file intentionally does not reuse the older env.py integration style. The
old environment changed qpos by hand, which is fast but bypasses contact
dynamics. Here actions command wheel velocity actuators and MuJoCo advances the
vehicle through mj_step, so walls and obstacles are physically meaningful.

URDF/Gazebo note:
URDF normally describes links/joints and Gazebo plugins/controllers provide the
simulation behavior. MJCF keeps geometry, contact parameters, actuators, cameras
and visualization helpers in one XML model, so we define the trainable dynamics
directly in assets/tracked_nav.xml.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
MODEL_PATH = os.path.join(ASSET_DIR, "tracked_nav.xml")


@dataclass
class RewardTerms:
    progress: float = 0.0
    velocity: float = 0.0
    safety: float = 0.0
    action: float = 0.0
    smooth: float = 0.0
    time: float = 0.0
    stuck: float = 0.0
    collision: float = 0.0
    goal: float = 0.0
    timeout: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(self.__dict__.values()))


class TrackedNavDynEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    N_RAYS = 72
    RAY_RANGE = 4.0
    MAX_LIN = 0.28
    MAX_ANG = 1.0
    TRACK_WIDTH = 0.40
    WHEEL_RADIUS = 0.055
    CONTROL_DT = 0.05
    GOAL_RADIUS = 0.35
    ROBOT_RADIUS = 0.32
    ROOM_HALF = 5.0
    MAX_STEPS = 700
    N_OBS = 12

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = None,
        stage: int = 1,
        scene: str = "mixed",
        log_reward_csv: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.stage = int(stage)
        self.scene = scene
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.np_random = np.random.default_rng(seed)
        self._viewer = None
        self._renderer = None
        self._reward_log_file = None
        self._reward_writer = None
        if log_reward_csv:
            os.makedirs(os.path.dirname(os.path.abspath(log_reward_csv)), exist_ok=True)
            self._reward_log_file = open(log_reward_csv, "w", newline="", encoding="utf-8")
            self._reward_writer = csv.DictWriter(
                self._reward_log_file,
                fieldnames=["episode", "step", "total", *RewardTerms().__dict__.keys()],
            )
            self._reward_writer.writeheader()

        self.root_x_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
        self.root_y_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_y")
        self.root_yaw_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_yaw")
        self.root_x_qpos = self.model.jnt_qposadr[self.root_x_jid]
        self.root_y_qpos = self.model.jnt_qposadr[self.root_y_jid]
        self.root_yaw_qpos = self.model.jnt_qposadr[self.root_yaw_jid]
        self.root_x_qvel = self.model.jnt_dofadr[self.root_x_jid]
        self.root_y_qvel = self.model.jnt_dofadr[self.root_y_jid]
        self.root_yaw_qvel = self.model.jnt_dofadr[self.root_yaw_jid]
        self.chassis_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.robot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "base_link",
                "front_bumper",
                "skid_pad",
                "lidar",
                "wheel_fl_geom",
                "wheel_rl_geom",
                "wheel_fr_geom",
                "wheel_rr_geom",
            )
        }
        self.obstacle_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in [f"obs_{i}" for i in range(self.N_OBS)] + ["wall_n", "wall_s", "wall_e", "wall_w"]
        }
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("motor_fl", "motor_rl", "motor_fr", "motor_rr")
        ]

        obs_dim = self.N_RAYS + 6
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._episode = 0
        self._step = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_path_progress = 0.0
        self._no_progress_steps = 0
        self._obs = []
        self._path = []
        self._path_s = np.zeros(1, dtype=float)
        self.start = np.zeros(2, dtype=float)
        self.goal = np.array([2.0, 0.0], dtype=float)
        self._last_terms = RewardTerms()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._episode += 1
        self._step = 0
        self._prev_action[:] = 0.0
        self._no_progress_steps = 0
        mujoco.mj_resetData(self.model, self.data)
        self._sample_layout()
        self._plan_path()
        yaw = self._initial_yaw()
        self._set_pose(self.start[0], self.start[1], yaw)
        self._prev_path_progress = self._path_progress(self.start)
        mujoco.mj_forward(self.model, self.data)
        self._update_visuals()
        return self._get_obs(), self._get_info(False, False, False)

    def set_goal(self, goal=None, min_goal_dist=1.5, reset_segment_clock=True):
        """Change only the navigation goal while keeping the current world/pose.

        This is used by continuous validation demos: the robot reaches one goal,
        then receives another goal in the same obstacle layout without resetting
        the simulation.
        """
        self.start = self._xy().copy()
        if goal is None:
            goal = self._sample_free_goal(self.start, min_goal_dist=min_goal_dist)
        self.goal = np.asarray(goal, dtype=float)
        self._plan_path()
        self._set_goal_marker()
        self._prev_path_progress = self._path_progress(self.start)
        self._no_progress_steps = 0
        self._prev_action[:] = 0.0
        self._last_terms = RewardTerms()
        if reset_segment_clock:
            self._step = 0
        self._update_visuals()
        mujoco.mj_forward(self.model, self.data)
        return self._get_info(False, False, False)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        lin_cmd = float(action[0]) * self.MAX_LIN
        ang_cmd = float(action[1]) * self.MAX_ANG
        left = (lin_cmd - 0.5 * self.TRACK_WIDTH * ang_cmd) / self.WHEEL_RADIUS
        right = (lin_cmd + 0.5 * self.TRACK_WIDTH * ang_cmd) / self.WHEEL_RADIUS
        self.data.ctrl[self.actuator_ids[0]] = left
        self.data.ctrl[self.actuator_ids[1]] = left
        self.data.ctrl[self.actuator_ids[2]] = right
        self.data.ctrl[self.actuator_ids[3]] = right

        n_substeps = max(1, int(round(self.CONTROL_DT / self.model.opt.timestep)))
        for _ in range(n_substeps):
            self._apply_velocity_servo(lin_cmd, ang_cmd)
            mujoco.mj_step(self.model, self.data)
        self._step += 1

        pos = self._xy()
        ranges = self._scan()
        collided = self._contact_collision()
        arrived = float(np.linalg.norm(pos - self.goal)) < self.GOAL_RADIUS
        truncated = self._step >= self.MAX_STEPS
        terms = self._reward_terms(action, ranges, collided, arrived, truncated)
        self._last_terms = terms
        obs = self._get_obs(ranges)
        terminated = bool(collided or arrived)
        if self._reward_writer:
            row = {"episode": self._episode, "step": self._step, "total": terms.total}
            row.update(terms.__dict__)
            self._reward_writer.writerow(row)
        if self.render_mode == "human":
            self.render()
        self._prev_action = action.copy()
        self._update_visuals()
        return obs, terms.total, terminated, bool(truncated and not terminated), self._get_info(collided, arrived, truncated)

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="top_view")
            return self._renderer.render()
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._reward_log_file is not None:
            self._reward_log_file.close()
            self._reward_log_file = None

    def _reward_terms(self, action, ranges, collided, arrived, truncated) -> RewardTerms:
        pos = self._xy()
        progress = self._path_progress(pos)
        delta = progress - self._prev_path_progress
        self._prev_path_progress = progress
        if delta < 0.002:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0

        vel_body = self._body_velocity()
        forward_vel = vel_body[0]
        min_range = float(np.min(ranges))
        safety = 0.0
        if min_range < 1.2:
            # NavRL-style dense safety gradient: gentle far away, steep near contact.
            safety = 0.30 * math.log(max(min_range, 1e-3) / 1.2)
        action_pen = -0.04 * (abs(float(action[1])) + max(0.0, -float(action[0])))
        smooth = -0.08 * float(np.linalg.norm(action - self._prev_action))
        stuck = -0.30 if self._no_progress_steps > 35 else 0.0
        terms = RewardTerms(
            progress=18.0 * max(-0.02, delta),
            velocity=0.50 * max(0.0, forward_vel),
            safety=safety,
            action=action_pen,
            smooth=smooth,
            time=-0.03,
            stuck=stuck,
            collision=-80.0 if collided else 0.0,
            goal=160.0 if arrived else 0.0,
            timeout=-20.0 if truncated and not arrived and not collided else 0.0,
        )
        return terms

    def _get_obs(self, ranges=None):
        if ranges is None:
            ranges = self._scan()
        pos = self._xy()
        yaw = self._yaw()
        guide = self._lookahead_point()
        guide_ang = _wrap(math.atan2(guide[1] - pos[1], guide[0] - pos[0]) - yaw)
        goal_ang = _wrap(math.atan2(self.goal[1] - pos[1], self.goal[0] - pos[0]) - yaw)
        remaining = max(0.0, self._path_s[-1] - self._path_progress(pos))
        vel_body = self._body_velocity()
        obs = np.concatenate(
            [
                ranges / self.RAY_RANGE,
                np.array(
                    [
                        min(remaining / 12.0, 1.0),
                        guide_ang / math.pi,
                        goal_ang / math.pi,
                        np.clip(vel_body[0] / self.MAX_LIN, -2.0, 2.0),
                        np.clip(self.data.qvel[self.root_yaw_qvel] / self.MAX_ANG, -2.0, 2.0),
                        np.clip(np.min(ranges) / self.RAY_RANGE, 0.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs.astype(np.float32)

    def _get_info(self, collided, arrived, truncated):
        terms = self._last_terms
        return {
            "episode_id": self._episode,
            "steps": self._step,
            "dist": float(np.linalg.norm(self._xy() - self.goal)),
            "path_progress": float(self._path_progress(self._xy())),
            "path_length": float(self._path_s[-1]),
            "collision": bool(collided),
            "arrived": bool(arrived),
            "timeout": bool(truncated),
            "min_laser": float(np.min(self._scan())),
            "reward_terms": terms.__dict__.copy(),
        }

    def _sample_layout(self):
        rng = self.np_random
        self._hide_obstacles()
        if self.stage <= 1:
            self.start = np.array([-2.0, rng.uniform(-0.8, 0.8)])
            self.goal = np.array([2.0, rng.uniform(-0.8, 0.8)])
        elif self.stage == 2:
            self.start = np.array([-3.0, rng.uniform(-1.0, 1.0)])
            self.goal = np.array([3.0, rng.uniform(-1.0, 1.0)])
            self._set_obstacle(0, [0.0, 0.0], [0.35, 0.75], [0.82, 0.54, 0.22, 1])
        else:
            self._build_mixed_scene(rng)
        self._set_goal_marker()

    def _build_mixed_scene(self, rng):
        scene = self.scene
        if scene == "mixed":
            scene = rng.choice(["corridor", "corner", "boxes"])
        if scene == "corridor":
            self.start = np.array([-3.7, rng.uniform(-1.0, 1.0)])
            self.goal = np.array([3.7, rng.uniform(-1.0, 1.0)])
            self._set_obstacle(0, [-0.2, 1.25], [2.2, 0.11], [0.62, 0.62, 0.60, 1])
            self._set_obstacle(1, [-0.2, -1.25], [2.2, 0.11], [0.62, 0.62, 0.60, 1])
        elif scene == "corner":
            self.start = np.array([-3.2, -2.8])
            self.goal = np.array([3.2, 3.0])
            self._set_obstacle(0, [1.0, 1.0], [0.12, 1.55], [0.62, 0.62, 0.60, 1])
            self._set_obstacle(1, [-0.15, 2.40], [1.25, 0.12], [0.62, 0.62, 0.60, 1])
            self.start += rng.uniform(-0.4, 0.4, 2)
            self.goal += rng.uniform(-0.35, 0.35, 2)
        else:
            self.start = np.array([-3.5, rng.uniform(-2.0, 2.0)])
            self.goal = np.array([3.5, rng.uniform(-2.0, 2.0)])
            for i, x in enumerate(np.linspace(-1.5, 1.8, 5)):
                y = rng.uniform(-2.3, 2.3)
                sx = rng.uniform(0.22, 0.45)
                sy = rng.uniform(0.18, 0.42)
                self._set_obstacle(i, [x, y], [sx, sy], [0.82, 0.54, 0.22, 1])

    def _hide_obstacles(self):
        self._obs = []
        for i in range(self.N_OBS):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obsb_{i}")
            self.model.body_pos[bid] = np.array([30.0 + i, 30.0, 0.3])
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}")
            self.model.geom_size[gid] = np.array([0.2, 0.2, 0.3])

    def _set_obstacle(self, idx, pos, half, rgba):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obsb_{idx}")
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{idx}")
        self.model.body_pos[bid] = np.array([pos[0], pos[1], 0.3])
        self.model.geom_size[gid] = np.array([half[0], half[1], 0.3])
        self.model.geom_rgba[gid] = np.array(rgba)
        self._obs.append((float(pos[0]), float(pos[1]), float(half[0]), float(half[1])))

    def _set_goal_marker(self):
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")
        self.model.geom_pos[gid] = np.array([self.goal[0], self.goal[1], 0.16])

    def _free_point(self, point, margin=None):
        margin = self.ROBOT_RADIUS + 0.25 if margin is None else margin
        x, y = float(point[0]), float(point[1])
        h = self.ROOM_HALF - margin - 0.2
        if x < -h or x > h or y < -h or y > h:
            return False
        return all(_point_box_dist(x, y, ox, oy, hx, hy) >= margin for ox, oy, hx, hy in self._all_boxes())

    def _sample_free_goal(self, origin, min_goal_dist=1.5, max_goal_dist=None, max_tries=500):
        origin = np.asarray(origin, dtype=float)
        lo = -self.ROOM_HALF + self.ROBOT_RADIUS + 0.55
        hi = self.ROOM_HALF - self.ROBOT_RADIUS - 0.55
        for _ in range(max_tries):
            goal = self.np_random.uniform(lo, hi, size=2)
            dist = float(np.linalg.norm(goal - origin))
            if dist < min_goal_dist:
                continue
            if max_goal_dist is not None and dist > max_goal_dist:
                continue
            if self._free_point(goal):
                return goal
        raise RuntimeError("failed to sample a free goal in the current scene")

    def _set_pose(self, x, y, yaw):
        self.data.qpos[self.root_x_qpos] = x
        self.data.qpos[self.root_y_qpos] = y
        self.data.qpos[self.root_yaw_qpos] = yaw
        self.data.qvel[self.root_x_qvel] = 0.0
        self.data.qvel[self.root_y_qvel] = 0.0
        self.data.qvel[self.root_yaw_qvel] = 0.0
        self.data.ctrl[:] = 0.0
        for _ in range(60):
            mujoco.mj_step(self.model, self.data)

    def _apply_velocity_servo(self, lin_cmd, ang_cmd):
        """Apply chassis-level force/torque servo for robust tracked UGV dynamics.

        This does not teleport qpos. MuJoCo still integrates velocities and
        resolves contacts against walls/obstacles. The small visual wheels spin,
        while chassis traction is represented by a bounded body force and yaw
        torque, which is a stable approximation for a tracked base when exact
        tread modeling is unnecessary.
        """
        self.data.qfrc_applied[:] = 0.0
        yaw = self._yaw()
        vel_body = self._body_velocity()
        yaw_rate = float(self.data.qvel[self.root_yaw_qvel])
        mass = float(self.model.body_mass[self.chassis_body])
        fwd_err = np.clip(lin_cmd - vel_body[0], -0.6, 0.6)
        lat_err = np.clip(0.0 - vel_body[1], -0.4, 0.4)
        force_body = np.array([mass * 8.0 * fwd_err, mass * 6.0 * lat_err, 0.0])
        c, s = math.cos(yaw), math.sin(yaw)
        force_world = np.array([
            c * force_body[0] - s * force_body[1],
            s * force_body[0] + c * force_body[1],
            0.0,
        ])
        torque_z = np.clip(0.9 * (ang_cmd - yaw_rate), -1.2, 1.2)
        self.data.qfrc_applied[self.root_x_qvel] = force_world[0]
        self.data.qfrc_applied[self.root_y_qvel] = force_world[1]
        self.data.qfrc_applied[self.root_yaw_qvel] = torque_z

    def _initial_yaw(self):
        direction = self.goal - self.start
        if len(self._path) > 1:
            direction = self._path[min(1, len(self._path) - 1)] - self.start
        return math.atan2(direction[1], direction[0])

    def _xy(self):
        return np.array([self.data.qpos[self.root_x_qpos], self.data.qpos[self.root_y_qpos]], dtype=float)

    def _yaw(self):
        return float(self.data.qpos[self.root_yaw_qpos])

    def _body_velocity(self):
        yaw = self._yaw()
        vx = float(self.data.qvel[self.root_x_qvel])
        vy = float(self.data.qvel[self.root_y_qvel])
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * vx + s * vy, -s * vx + c * vy], dtype=float)

    def _scan(self):
        pos = self._xy()
        yaw = self._yaw()
        out = np.full(self.N_RAYS, self.RAY_RANGE, dtype=np.float32)
        for i in range(self.N_RAYS):
            a = yaw + 2.0 * math.pi * i / self.N_RAYS
            d = self.RAY_RANGE
            dx, dy = math.cos(a), math.sin(a)
            for ox, oy, hx, hy in self._all_boxes():
                hit = _ray_aabb(pos[0], pos[1], dx, dy, ox, oy, hx, hy)
                if hit is not None and 0.0 < hit < d:
                    d = hit
            out[i] = d
        return out

    def _all_boxes(self):
        boxes = list(self._obs)
        h = self.ROOM_HALF
        boxes.extend([(0.0, h, h, 0.12), (0.0, -h, h, 0.12), (h, 0.0, 0.12, h), (-h, 0.0, 0.12, h)])
        return boxes

    def _contact_collision(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.robot_geom_ids and g2 in self.obstacle_geom_ids) or (
                g2 in self.robot_geom_ids and g1 in self.obstacle_geom_ids
            ):
                return True
        return False

    def _plan_path(self):
        res = 0.15
        lo, hi = -self.ROOM_HALF + 0.3, self.ROOM_HALF - 0.3
        n = int(round((hi - lo) / res)) + 1

        def to_idx(p):
            return (int(round((p[0] - lo) / res)), int(round((p[1] - lo) / res)))

        def to_xy(ix, iy):
            return np.array([lo + ix * res, lo + iy * res], dtype=float)

        occ = np.zeros((n, n), dtype=bool)
        # Use the same conservative clearance family as goal sampling. A lower
        # margin makes A*/smoothing accept paths that the true chassis cannot
        # reliably track after a continuous goal switch.
        margin = self.ROBOT_RADIUS + 0.25
        for ix in range(n):
            x = lo + ix * res
            for iy in range(n):
                y = lo + iy * res
                for ox, oy, hx, hy in self._all_boxes():
                    if _point_box_dist(x, y, ox, oy, hx, hy) < margin:
                        occ[ix, iy] = True
                        break
        s = to_idx(self.start)
        g = to_idx(self.goal)
        occ[s] = False
        occ[g] = False
        pq = [(0.0, s)]
        prev = {}
        cost = {s: 0.0}
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while pq:
            _, cur = heappop(pq)
            if cur == g:
                break
            for dx, dy in neighbors:
                nb = (cur[0] + dx, cur[1] + dy)
                if nb[0] < 0 or nb[0] >= n or nb[1] < 0 or nb[1] >= n or occ[nb]:
                    continue
                step = math.hypot(dx, dy) * res
                nc = cost[cur] + step
                if nc < cost.get(nb, 1e9):
                    cost[nb] = nc
                    prev[nb] = cur
                    h = math.hypot(nb[0] - g[0], nb[1] - g[1]) * res
                    heappush(pq, (nc + h, nb))
        if g not in prev:
            pts = [self.start.copy(), self.goal.copy()]
        else:
            cells = [g]
            while cells[-1] != s:
                cells.append(prev[cells[-1]])
            cells.reverse()
            pts = [self.start.copy()]
            pts.extend(to_xy(ix, iy) for ix, iy in cells[1:-1:3])
            pts.append(self.goal.copy())
        self._path = _smooth_path(pts, self._all_boxes(), margin)
        self._path_s = np.zeros(len(self._path), dtype=float)
        for i in range(1, len(self._path)):
            self._path_s[i] = self._path_s[i - 1] + float(np.linalg.norm(self._path[i] - self._path[i - 1]))

    def _path_progress(self, pos):
        pos = np.asarray(pos, dtype=float)
        best_s = 0.0
        best_d = 1e9
        for i in range(len(self._path) - 1):
            a, b = self._path[i], self._path[i + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            t = 0.0 if denom < 1e-9 else float(np.clip(np.dot(pos - a, ab) / denom, 0.0, 1.0))
            p = a + t * ab
            d = float(np.linalg.norm(pos - p))
            if d < best_d:
                best_d = d
                best_s = self._path_s[i] + t * float(np.linalg.norm(ab))
        return best_s

    def _lookahead_point(self, lookahead=0.8):
        target_s = min(self._path_s[-1], self._path_progress(self._xy()) + lookahead)
        for i in range(len(self._path_s) - 1):
            if self._path_s[i + 1] >= target_s:
                span = self._path_s[i + 1] - self._path_s[i]
                t = 0.0 if span < 1e-9 else (target_s - self._path_s[i]) / span
                return self._path[i] + t * (self._path[i + 1] - self._path[i])
        return self.goal.copy()

    def _update_visuals(self):
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "guide")
        guide = self._lookahead_point()
        self.model.geom_pos[gid] = np.array([guide[0], guide[1], 0.04])
        samples = _resample_polyline(self._path, 12)
        for i in range(12):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"path_{i}")
            if i < len(samples):
                self.model.geom_pos[gid] = np.array([samples[i][0], samples[i][1], 0.03])
            else:
                self.model.geom_pos[gid] = np.array([50.0, 50.0, 0.03])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ray_aabb(px, py, dx, dy, cx, cy, hx, hy):
    tmin, tmax = 0.0, 1e9
    for p, d, lo, hi in ((px, dx, cx - hx, cx + hx), (py, dy, cy - hy, cy + hy)):
        if abs(d) < 1e-12:
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
    return tmin if tmin >= 0 else tmax


def _point_box_dist(px, py, cx, cy, hx, hy):
    return math.hypot(max(abs(px - cx) - hx, 0.0), max(abs(py - cy) - hy, 0.0))


def _seg_clear(a, b, boxes, margin):
    steps = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / 0.08))
    for t in np.linspace(0.0, 1.0, steps):
        p = np.asarray(a) + t * (np.asarray(b) - np.asarray(a))
        for ox, oy, hx, hy in boxes:
            if _point_box_dist(p[0], p[1], ox, oy, hx, hy) < margin:
                return False
    return True


def _smooth_path(points: Iterable[np.ndarray], boxes, margin):
    pts = [np.asarray(p, dtype=float) for p in points]
    out = []
    i = 0
    while i < len(pts):
        out.append(pts[i])
        if i == len(pts) - 1:
            break
        j = len(pts) - 1
        while j > i + 1 and not _seg_clear(pts[i], pts[j], boxes, margin):
            j -= 1
        i = j
    return out


def _resample_polyline(points, count):
    pts = [np.asarray(p, dtype=float) for p in points]
    if not pts:
        return []
    if len(pts) == 1:
        return pts
    s = np.zeros(len(pts), dtype=float)
    for i in range(1, len(pts)):
        s[i] = s[i - 1] + float(np.linalg.norm(pts[i] - pts[i - 1]))
    if s[-1] < 1e-9:
        return pts[:1]
    out = []
    for target in np.linspace(0.0, s[-1], count):
        for i in range(len(s) - 1):
            if s[i + 1] >= target:
                span = s[i + 1] - s[i]
                t = 0.0 if span < 1e-9 else (target - s[i]) / span
                out.append(pts[i] + t * (pts[i + 1] - pts[i]))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--scene", default="mixed")
    args = ap.parse_args()
    env = TrackedNavDynEnv(stage=args.stage, scene=args.scene)
    obs, info = env.reset()
    print("obs", obs.shape, "info", info)
    for i in range(args.steps):
        obs, reward, term, trunc, info = env.step([0.8, 0.0])
        if i % 20 == 0 or term or trunc:
            print(i, "reward", round(reward, 3), "pos", env._xy().round(3), "info", info)
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    main()
