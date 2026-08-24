#!/usr/bin/env python
"""Render a MuJoCo 3D continuous-goal GIF for the SAC navigation demo."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_dyn import TrackedNavDynEnv
from view_multigoal_dyn import heading_guard_action, load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="docs/images/sac_mujoco_3d_seed2.gif")
    ap.add_argument("--snapshot-out", default="docs/images/sac_mujoco_3d_seed2.png")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--scene", default="boxes", choices=["mixed", "corridor", "corner", "boxes"])
    ap.add_argument("--goals", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--goal-radius", type=float, default=0.35)
    ap.add_argument("--max-steps-per-goal", type=int, default=900)
    ap.add_argument("--frame-skip", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=180)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    policy, policy_kind = load_policy(Path(args.model).expanduser().resolve())
    env = TrackedNavDynEnv(stage=args.stage, scene=args.scene, seed=args.seed)
    env.MAX_STEPS = int(args.max_steps_per_goal)
    env.GOAL_RADIUS = float(args.goal_radius)
    obs, _ = env.reset()
    env.set_goal(min_goal_dist=1.8)

    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    frames: list[Image.Image] = []
    snapshot = None
    successes = collisions = timeouts = 0
    goal_idx = 0
    segment_steps = 0

    try:
        while goal_idx < args.goals:
            if len(frames) < args.max_frames and segment_steps % args.frame_skip == 0:
                renderer.update_scene(env.data, camera="top_view")
                image = renderer.render()
                frame = Image.fromarray(image)
                frames.append(frame)
                if snapshot is None and goal_idx >= 2:
                    snapshot = frame.copy()

            action, _ = policy.predict(obs, deterministic=True)
            action = heading_guard_action(obs, action)
            obs, _, term, trunc, info = env.step(action)
            segment_steps += 1

            reached = bool(info["arrived"])
            failed = bool(info["collision"] or segment_steps >= args.max_steps_per_goal)
            if not reached and not failed:
                continue

            goal_idx += 1
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
                f"[render3d] goal {goal_idx}/{args.goals}: {result} "
                f"steps={segment_steps} dist={info['dist']:.2f}"
            )

            if goal_idx >= args.goals or result != "arrived":
                break
            env.set_goal(min_goal_dist=1.8)
            obs = env._get_obs()
            segment_steps = 0

        out = Path(args.out).expanduser().resolve()
        snap_out = Path(args.snapshot_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        snap_out.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            raise RuntimeError("No frames rendered")
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0)
        (snapshot or frames[min(len(frames) - 1, len(frames) // 2)]).save(snap_out)
        print(f"[render3d] policy={policy_kind} gif={out}")
        print(f"[render3d] snapshot={snap_out}")
        print(f"[render3d] done goals={goal_idx} success={successes} collision={collisions} timeout={timeouts}")
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
