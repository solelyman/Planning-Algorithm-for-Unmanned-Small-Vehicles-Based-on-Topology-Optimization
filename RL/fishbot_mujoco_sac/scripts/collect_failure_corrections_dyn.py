#!/usr/bin/env python
"""Collect small DAgger-style correction data from closed-loop failures.

The main BC dataset should still represent ordinary successful driving. This
script only records a capped window of risky states produced by the current
policy, then relabels them with the guide controller plus a conservative
near-obstacle override.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_dyn import TrackedNavDynEnv
from view_dyn import heuristic_action
from view_multigoal_dyn import load_policy


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "runs_dyn"


def conservative_teacher_action(env, threshold=0.80):
    action = heuristic_action(env).astype(np.float32)
    ranges = env._scan()
    front = float(min(np.min(ranges[:12]), np.min(ranges[-12:])))
    left_front = float(np.mean(ranges[8:30]))
    right_front = float(np.mean(ranges[-30:-8]))
    side_near = float(min(np.min(ranges[10:32]), np.min(ranges[-32:-10])))

    if front < threshold:
        action[0] = -0.28 if front < threshold * 0.65 else 0.10
        action[1] = 1.0 if left_front > right_front else -1.0
    elif side_near < threshold * 0.55:
        action[0] = min(action[0], 0.35)
        action[1] = 0.75 if left_front > right_front else -0.75
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def collect(args):
    policy, policy_kind = load_policy(Path(args.model).expanduser().resolve())
    print(f"[correct] loaded {policy_kind}: {args.model}")
    obs_rows, act_rows = [], []
    stats = {
        "episodes": args.episodes,
        "segments": 0,
        "arrived": 0,
        "collision": 0,
        "timeout": 0,
        "failure_windows": 0,
        "risk_samples": 0,
        "samples": 0,
    }
    for ep in range(args.episodes):
        env = TrackedNavDynEnv(stage=args.stage, scene=args.scene, seed=args.seed + ep)
        try:
            obs, _ = env.reset()
            env.set_goal(min_goal_dist=args.min_goal_dist)
            obs = env._get_obs()
            recent = deque(maxlen=args.pre_fail_window)
            goals_done = 0
            while goals_done < args.goals_per_episode and len(obs_rows) < args.max_samples:
                recent.clear()
                segment_added = False
                info = {}
                for step in range(args.max_steps_per_goal):
                    teacher_action = conservative_teacher_action(env, threshold=args.teacher_threshold)
                    action, _ = policy.predict(obs, deterministic=True)
                    min_laser = float(obs[77]) * env.RAY_RANGE
                    risky = min_laser < args.risk_laser or abs(float(obs[73])) > args.risk_guide_ang
                    recent.append((obs.copy(), teacher_action.copy(), risky))

                    next_obs, _, term, trunc, info = env.step(action)
                    obs = next_obs
                    if term or trunc or step + 1 >= args.max_steps_per_goal:
                        break

                arrived = bool(info.get("arrived", False))
                collision = bool(info.get("collision", False))
                timeout = not arrived and not collision
                stats["segments"] += 1
                stats["arrived"] += int(arrived)
                stats["collision"] += int(collision)
                stats["timeout"] += int(timeout)

                if collision or timeout:
                    window = list(recent)
                    stats["failure_windows"] += 1
                else:
                    window = [row for row in recent if row[2]]

                for row_obs, row_act, risky in window:
                    if len(obs_rows) >= args.max_samples:
                        break
                    if risky or collision or timeout:
                        obs_rows.append(row_obs)
                        act_rows.append(row_act)
                        stats["risk_samples"] += int(risky)
                        segment_added = True

                if segment_added:
                    stats["samples"] = len(obs_rows)
                goals_done += 1
                if goals_done >= args.goals_per_episode or len(obs_rows) >= args.max_samples:
                    break
                if not arrived:
                    obs, _ = env.reset()
                env.set_goal(min_goal_dist=args.min_goal_dist)
                obs = env._get_obs()
        finally:
            env.close()
        if (ep + 1) % max(1, args.report_every) == 0:
            print(
                f"[correct] ep={ep + 1}/{args.episodes} samples={len(obs_rows)} "
                f"fail_windows={stats['failure_windows']} collision={stats['collision']} timeout={stats['timeout']}"
            )
        if len(obs_rows) >= args.max_samples:
            break
    if not obs_rows:
        raise RuntimeError("No correction samples collected; try more episodes or looser risk thresholds.")
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(act_rows, dtype=np.float32), stats


def describe(obs, actions):
    extra = obs[:, 72:]
    return {
        "samples": int(len(obs)),
        "goal_ang_abs_gt_0_8": float(np.mean(np.abs(extra[:, 2]) > 0.8)),
        "guide_ang_abs_gt_0_8": float(np.mean(np.abs(extra[:, 1]) > 0.8)),
        "min_laser_lt_0_5": float(np.mean(extra[:, 5] < 0.5 / TrackedNavDynEnv.RAY_RANGE)),
        "min_laser_lt_0_3": float(np.mean(extra[:, 5] < 0.3 / TrackedNavDynEnv.RAY_RANGE)),
        "turn_abs_gt_0_8": float(np.mean(np.abs(actions[:, 1]) > 0.8)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--scene", default="boxes", choices=["mixed", "corridor", "corner", "boxes"])
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--goals-per-episode", type=int, default=8)
    ap.add_argument("--max-steps-per-goal", type=int, default=900)
    ap.add_argument("--min-goal-dist", type=float, default=1.8)
    ap.add_argument("--risk-laser", type=float, default=0.75)
    ap.add_argument("--risk-guide-ang", type=float, default=0.55)
    ap.add_argument("--teacher-threshold", type=float, default=0.80)
    ap.add_argument("--pre-fail-window", type=int, default=90)
    ap.add_argument("--max-samples", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_name = time.strftime("bc_corrections_%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out).expanduser().resolve() if args.out else out_dir / "bc_corrections_dataset.npz"

    obs, actions, stats = collect(args)
    summary = describe(obs, actions)
    np.savez_compressed(out, obs=obs, actions=actions)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "collect": stats, "summary": summary, "dataset": str(out)}, f, indent=2)
    print(f"[correct] saved {out}")
    print(f"[correct] stats={stats}")
    print(f"[correct] summary={summary}")


if __name__ == "__main__":
    main()
