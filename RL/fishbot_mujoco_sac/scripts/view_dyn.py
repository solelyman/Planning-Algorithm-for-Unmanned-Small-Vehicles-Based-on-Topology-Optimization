#!/usr/bin/env python
"""GUI viewer for the true-dynamics MuJoCo tracked navigation environment."""
import argparse
import math
import os
import time

import numpy as np
import mujoco

from env_dyn import TrackedNavDynEnv


def heuristic_action(env):
    pos = env._xy()
    yaw = env._yaw()
    guide = env._lookahead_point()
    ang = math.atan2(guide[1] - pos[1], guide[0] - pos[0]) - yaw
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    ranges = env._scan()
    front = float(min(np.min(ranges[:10]), np.min(ranges[-10:])))
    left = float(np.mean(ranges[8:28]))
    right = float(np.mean(ranges[-28:-8]))
    turn = float(np.clip(ang / 1.0, -1.0, 1.0))
    abs_ang = abs(ang)
    if getattr(env, "_no_progress_steps", 0) > 70 and front < 0.50:
        return np.array([-0.22, 1.0 if left > right else -1.0], dtype=np.float32)
    if abs_ang > 1.65:
        lin = 0.0
    elif abs_ang > 1.15:
        lin = 0.18
    elif abs_ang > 0.65:
        lin = 0.45
    else:
        lin = 0.8
    if front < 0.65:
        if front < 0.42:
            lin = -0.20
            turn = 1.0 if left > right else -1.0
        else:
            lin = min(lin, 0.12 if abs_ang < 1.15 else 0.0)
    return np.array([lin, turn], dtype=np.float32)


def draw_lasers(env):
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
        start = np.array([pos[0], pos[1], 0.14], dtype=np.float64)
        end = np.array([pos[0] + dist * math.cos(ang), pos[1] + dist * math.sin(ang), 0.14], dtype=np.float64)
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            np.array([1.0, 0.85, 0.0, 0.55], dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, start, end)
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--scene", default="mixed")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    if "DISPLAY" not in os.environ:
        print("[view_dyn] DISPLAY not found; run from the desktop terminal for GUI.")
        return
    policy = None
    if args.model:
        from stable_baselines3 import PPO

        policy = PPO.load(args.model, device="cpu")
        print(f"[view_dyn] loaded policy: {args.model}")
    env = TrackedNavDynEnv(render_mode="human", stage=args.stage, scene=args.scene, seed=0)
    obs, _ = env.reset()
    ep = 1
    print("[view_dyn] green path points, cyan guide, gold goal. Close window or Ctrl+C to stop.")
    try:
        while ep <= args.episodes:
            if policy is None:
                action = heuristic_action(env)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            draw_lasers(env)
            if info["steps"] % 50 == 0:
                print(
                    f"ep={ep} step={info['steps']} reward={reward:.2f} "
                    f"dist={info['dist']:.2f} min_laser={info['min_laser']:.2f}"
                )
            if term or trunc:
                end = "arrived" if info["arrived"] else ("collision" if info["collision"] else "timeout")
                print(f"Episode {ep}: {end} steps={info['steps']} terms={info['reward_terms']}")
                ep += 1
                if ep <= args.episodes:
                    obs, _ = env.reset()
            time.sleep(env.CONTROL_DT)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
