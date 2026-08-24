#!/usr/bin/env python
"""Continuous multi-goal GUI validation for the true-dynamics MuJoCo env."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_dyn import TrackedNavDynEnv
from view_dyn import draw_lasers, heuristic_action


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "runs_dyn"


def latest_best_model() -> Path | None:
    models = sorted(RUNS_DIR.glob("dyn_*/best_model.zip"), key=lambda p: p.stat().st_mtime)
    return models[-1] if models else None


def load_policy(path):
    from stable_baselines3 import PPO, SAC

    try:
        return PPO.load(str(path), device="cpu"), "PPO"
    except Exception as ppo_error:
        try:
            return SAC.load(str(path), device="cpu"), "SAC"
        except Exception:
            raise ppo_error


def draw_matplotlib_frame(ax, env, title, trail):
    pos = env._xy()
    yaw = env._yaw()
    ranges = env._scan()
    ax.clear()
    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-5.4, 5.4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    for ox, oy, hx, hy in env._all_boxes():
        color = "#c58a37" if hx < 4.5 and hy < 4.5 else "#9a9a9a"
        ax.add_patch(
            Rectangle(
                (ox - hx, oy - hy),
                2 * hx,
                2 * hy,
                facecolor=color,
                edgecolor="#4a4a4a",
                alpha=0.80,
            )
        )
    if len(env._path) > 1:
        pts = np.asarray(env._path)
        ax.plot(pts[:, 0], pts[:, 1], color="#15a04a", lw=1.6, ls="--", alpha=0.85)
    if len(trail) > 1:
        pts = np.asarray(trail)
        ax.plot(pts[:, 0], pts[:, 1], color="#2557d6", lw=1.5, alpha=0.8)
    guide = env._lookahead_point()
    ax.add_patch(Circle(guide, 0.08, facecolor="#00cfe8", edgecolor="black", zorder=5))
    ax.add_patch(Circle(env.goal, env.GOAL_RADIUS, facecolor="#ffe45c", edgecolor="#806600", alpha=0.8))
    for i, dist in enumerate(ranges):
        if i % 2:
            continue
        ang = yaw + 2.0 * math.pi * i / env.N_RAYS
        color = "#e34a33" if dist < env.RAY_RANGE - 0.02 else "#444444"
        ax.plot(
            [pos[0], pos[0] + dist * math.cos(ang)],
            [pos[1], pos[1] + dist * math.sin(ang)],
            color=color,
            lw=0.45,
            alpha=0.45,
        )
    ax.add_patch(Circle(pos, env.ROBOT_RADIUS, facecolor="#2b67c6", edgecolor="black", zorder=6))
    ax.arrow(
        pos[0],
        pos[1],
        0.32 * math.cos(yaw),
        0.32 * math.sin(yaw),
        head_width=0.12,
        head_length=0.12,
        fc="black",
        ec="black",
        zorder=7,
    )


def safety_shield_action(env, action, threshold=0.72, state=None):
    ranges = env._scan()
    front = float(min(np.min(ranges[:12]), np.min(ranges[-12:])))
    left_front = float(np.mean(ranges[8:28]))
    right_front = float(np.mean(ranges[-28:-8]))
    side_near = float(min(np.min(ranges[12:30]), np.min(ranges[-30:-12])))
    if state is None:
        state = {"steps": 0, "turn": 0.0}

    danger = front < threshold or side_near < threshold * 0.55
    if state["steps"] <= 0 and not danger:
        return action, False

    if state["steps"] <= 0:
        state["turn"] = 1.0 if left_front > right_front else -1.0
        state["steps"] = 10 if front < threshold * 0.70 else 6

    state["steps"] -= 1
    # Conservative recovery: create clearance only when blocked head-on; otherwise arc gently.
    if front < threshold * 0.58:
        lin = -0.30
    else:
        lin = 0.12
    return np.array([lin, state["turn"]], dtype=np.float32), True


def heading_guard_action(obs, action):
    action = np.asarray(action, dtype=np.float32).copy()
    guide_ang = abs(float(obs[73]) * math.pi)
    if guide_ang > 1.65:
        action[0] = min(float(action[0]), 0.0)
    elif guide_ang > 1.15:
        action[0] = min(float(action[0]), 0.20)
    elif guide_ang > 0.65:
        action[0] = min(float(action[0]), 0.45)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def choose_action(env, obs, policy, use_heuristic, use_shield=False, shield_threshold=0.72):
    if use_heuristic or policy is None:
        return heuristic_action(env)
    action, _ = policy.predict(obs, deterministic=True)
    if use_shield:
        action, _ = safety_shield_action(env, action, threshold=shield_threshold)
    return action


def draw_mujoco_debug(env, trail):
    viewer = env._viewer
    if viewer is None:
        return
    scn = viewer.user_scn
    scn.ngeom = 0
    pos = env._xy()
    yaw = env._yaw()
    ranges = env._scan()

    for i, dist in enumerate(ranges):
        if i % 2:
            continue
        ang = yaw + 2.0 * math.pi * i / env.N_RAYS
        start = np.array([pos[0], pos[1], 0.16], dtype=np.float64)
        end = np.array([pos[0] + dist * math.cos(ang), pos[1] + dist * math.sin(ang), 0.16], dtype=np.float64)
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            np.array([1.0, 0.85, 0.0, 0.45], dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.005, start, end)
        scn.ngeom += 1

    for a, b in zip(trail[-120:-1:3], trail[-119::3]):
        if scn.ngeom >= scn.maxgeom:
            break
        start = np.array([a[0], a[1], 0.06], dtype=np.float64)
        end = np.array([b[0], b[1], 0.06], dtype=np.float64)
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            np.array([0.10, 0.32, 1.0, 0.65], dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, start, end)
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="PPO model path. Default: latest runs_dyn/*/best_model.zip")
    ap.add_argument("--heuristic", action="store_true", help="Use guide controller instead of PPO.")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--scene", default="boxes", choices=["mixed", "corridor", "corner", "boxes"])
    ap.add_argument("--goals", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps-per-goal", type=int, default=700)
    ap.add_argument("--min-goal-dist", type=float, default=1.8)
    ap.add_argument("--sleep", type=float, default=0.03)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--continue-on-fail", action="store_true", help="Keep sampling goals after collision/timeout.")
    ap.add_argument("--goal-radius", type=float, default=None, help="Override arrival radius for stricter validation.")
    ap.add_argument("--safety-shield", action="store_true", help="Override network action only when the front lidar is dangerously close.")
    ap.add_argument("--shield-threshold", type=float, default=0.72)
    ap.add_argument("--heading-guard", action="store_true", help="Cap forward speed during large heading changes.")
    ap.add_argument("--gif-out", default=None)
    ap.add_argument("--gif-max-frames", type=int, default=260)
    args = ap.parse_args()

    policy = None
    model_path = None
    if not args.heuristic:
        model_path = Path(args.model).expanduser().resolve() if args.model else latest_best_model()
        if model_path is None:
            raise FileNotFoundError("No model was provided and no runs_dyn/*/best_model.zip exists")
        policy, policy_kind = load_policy(model_path)
        print(f"[multigoal] loaded {policy_kind}: {model_path}")

    render_mode = None if args.no_gui else "human"
    if render_mode == "human" and "DISPLAY" not in os.environ:
        print("[multigoal] DISPLAY not found; falling back to --no-gui mode.")
        render_mode = None

    env = TrackedNavDynEnv(render_mode=render_mode, stage=args.stage, scene=args.scene, seed=args.seed)
    env.MAX_STEPS = int(args.max_steps_per_goal)
    if args.goal_radius is not None:
        env.GOAL_RADIUS = float(args.goal_radius)
    obs, _ = env.reset()
    env.set_goal(min_goal_dist=args.min_goal_dist)
    trail = [env._xy().copy()]
    frames = []
    fig = ax = None
    if args.gif_out:
        fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=90)

    successes = collisions = timeouts = 0
    shield_count = 0
    shield_state = {"steps": 0, "turn": 0.0}
    total_goals = 0
    segment_steps = 0
    print(
        f"[multigoal] policy={model_path or 'heuristic'} scene={args.scene} "
        f"stage={args.stage} goals={args.goals}"
    )
    print(f"[multigoal] first goal={env.goal.round(2)}")

    try:
        while total_goals < args.goals:
            if args.safety_shield and not args.heuristic and policy is not None:
                raw_action, _ = policy.predict(obs, deterministic=True)
                if args.heading_guard:
                    raw_action = heading_guard_action(obs, raw_action)
                action, shielded = safety_shield_action(
                    env, raw_action, threshold=args.shield_threshold, state=shield_state
                )
                shield_count += int(shielded)
            else:
                action = choose_action(env, obs, policy, args.heuristic)
                if args.heading_guard and not args.heuristic:
                    action = heading_guard_action(obs, action)
            obs, reward, term, trunc, info = env.step(action)
            segment_steps += 1
            trail.append(env._xy().copy())

            if render_mode == "human":
                draw_mujoco_debug(env, trail)
            if fig is not None and len(frames) < args.gif_max_frames and segment_steps % 4 == 0:
                draw_matplotlib_frame(
                    ax,
                    env,
                    f"multi-goal | goal {total_goals + 1}/{args.goals} | step {segment_steps} | dist {info['dist']:.2f} m",
                    trail,
                )
                fig.canvas.draw()
                frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))

            reached = bool(info["arrived"])
            failed = bool(info["collision"] or segment_steps >= args.max_steps_per_goal)
            if reached or failed:
                total_goals += 1
                if reached:
                    successes += 1
                    result = "arrived"
                elif info["collision"]:
                    collisions += 1
                    result = "collision"
                else:
                    timeouts += 1
                    result = "timeout"
                print(
                    f"[multigoal] goal {total_goals}/{args.goals}: {result} "
                    f"steps={segment_steps} dist={info['dist']:.2f} "
                    f"success={successes} collision={collisions} timeout={timeouts}"
                )
                if total_goals >= args.goals:
                    break
                if result != "arrived" and not args.continue_on_fail:
                    print("[multigoal] stopped on failure; inspect the MuJoCo window, then Ctrl+C/close it.")
                    while render_mode == "human" and env._viewer is not None and env._viewer.is_running():
                        env.render()
                        draw_mujoco_debug(env, trail)
                        time.sleep(0.05)
                    break
                if info["collision"]:
                    obs, _ = env.reset()
                    trail = [env._xy().copy()]
                env.set_goal(min_goal_dist=args.min_goal_dist)
                obs = env._get_obs()
                shield_state = {"steps": 0, "turn": 0.0}
                segment_steps = 0
                print(f"[multigoal] next goal={env.goal.round(2)}")
            if args.sleep > 0:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("[multigoal] stopped by user")
    finally:
        env.close()
        if fig is not None:
            if frames:
                out = Path(args.gif_out).expanduser().resolve()
                out.parent.mkdir(parents=True, exist_ok=True)
                frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0)
                print(f"[multigoal] gif={out}")
            plt.close(fig)
    print(
        f"[multigoal] done goals={total_goals} success={successes} "
        f"collision={collisions} timeout={timeouts} shield={shield_count}"
    )


if __name__ == "__main__":
    main()
