#!/usr/bin/env python
"""Evaluate a PPO policy or the built-in guide controller in TrackedNavDynEnv."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_dyn import TrackedNavDynEnv
from view_dyn import heuristic_action


def run_episode(env, policy):
    obs, _ = env.reset()
    total = 0.0
    info = {}
    while True:
        if policy is None:
            action = heuristic_action(env)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return {
        "arrived": bool(info.get("arrived", False)),
        "collision": bool(info.get("collision", False)),
        "timeout": bool(
            info.get("timeout", False)
            and not info.get("arrived", False)
            and not info.get("collision", False)
        ),
        "steps": int(info.get("steps", 0)),
        "reward": total,
        "dist": float(info.get("dist", 0.0)),
        "min_laser": float(info.get("min_laser", 0.0)),
    }


def summarize(rows):
    n = max(1, len(rows))
    return {
        "episodes": len(rows),
        "arrived_rate": sum(r["arrived"] for r in rows) / n,
        "collision_rate": sum(r["collision"] for r in rows) / n,
        "timeout_rate": sum(r["timeout"] for r in rows) / n,
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_dist": float(np.mean([r["dist"] for r in rows])),
        "mean_min_laser": float(np.mean([r["min_laser"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="PPO .zip path. Omit for heuristic baseline.")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--scene", default="all", choices=["all", "mixed", "corridor", "corner", "boxes"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    policy = None
    if args.model:
        from stable_baselines3 import PPO

        policy = PPO.load(args.model, device="cpu")

    scenes = ["corridor", "corner", "boxes"] if args.scene == "all" else [args.scene]
    report = {}
    for scene in scenes:
        rows = []
        for ep in range(args.episodes):
            env = TrackedNavDynEnv(stage=args.stage, scene=scene, seed=args.seed + ep)
            rows.append(run_episode(env, policy))
            env.close()
        report[scene] = summarize(rows)

    print("\n===== TrackedNavDyn Eval =====")
    print(f"policy: {args.model or 'heuristic baseline'}")
    for scene, stats in report.items():
        print(
            f"{scene:8s} arrived={stats['arrived_rate']:.2f} "
            f"collision={stats['collision_rate']:.2f} timeout={stats['timeout_rate']:.2f} "
            f"steps={stats['mean_steps']:.1f} reward={stats['mean_reward']:.1f} "
            f"min_laser={stats['mean_min_laser']:.2f}"
        )

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
