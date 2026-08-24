#!/usr/bin/env python
"""Collect BC data focused on continuous-goal closed-loop states."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_dyn import TrackedNavDynEnv
from view_dyn import heuristic_action


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
RUNS_DIR = os.path.join(PROJECT_DIR, "runs_dyn")


def sample_goal(env, min_goal_dist, hard_turn_prob):
    pos = env._xy()
    yaw = env._yaw()
    if env.np_random.random() > hard_turn_prob:
        return env._sample_free_goal(pos, min_goal_dist=min_goal_dist)
    lo = -env.ROOM_HALF + env.ROBOT_RADIUS + 0.55
    hi = env.ROOM_HALF - env.ROBOT_RADIUS - 0.55
    for _ in range(500):
        dist = env.np_random.uniform(min_goal_dist, min(7.0, env.ROOM_HALF * 1.4))
        rel_ang = env.np_random.choice([-1.0, 1.0]) * env.np_random.uniform(0.65 * np.pi, np.pi)
        goal = pos + dist * np.array([np.cos(yaw + rel_ang), np.sin(yaw + rel_ang)])
        if lo <= goal[0] <= hi and lo <= goal[1] <= hi and env._free_point(goal):
            return goal
    return env._sample_free_goal(pos, min_goal_dist=min_goal_dist)


def sample_weight(obs, action, step_after_goal, boundary_steps, mode):
    guide_ang = abs(float(obs[73]))
    goal_ang = abs(float(obs[74]))
    min_laser = float(obs[77]) * TrackedNavDynEnv.RAY_RANGE
    turn = abs(float(action[1]))
    weight = 1
    if mode == "aggressive":
        if step_after_goal < boundary_steps:
            weight += 2
        if guide_ang > 0.45 or goal_ang > 0.55:
            weight += 3
        if min_laser < 0.8:
            weight += 3
        if turn > 0.7:
            weight += 1
        return min(weight, 8)

    # Balanced mode: normal driving remains the dominant distribution.
    # Goal-switch and large-angle states are useful, but near-obstacle samples
    # are capped instead of amplified so the student does not learn to drive
    # aggressively along walls.
    if step_after_goal < boundary_steps:
        weight += 1
    if guide_ang > 0.55 or goal_ang > 0.65:
        weight += 1
    if turn > 0.85 and min_laser > 0.75:
        weight += 1
    if min_laser < 0.55:
        weight = min(weight, 1)
    return min(weight, 3)


def collect(args):
    obs_rows, act_rows = [], []
    stats = {
        "episodes": args.episodes,
        "arrived": 0,
        "collision": 0,
        "timeout": 0,
        "kept_segments": 0,
        "dropped_segments": 0,
        "raw_samples": 0,
        "weighted_samples": 0,
    }
    for ep in range(args.episodes):
        env = TrackedNavDynEnv(stage=args.stage, scene=args.scene, seed=args.seed + ep)
        try:
            obs, _ = env.reset()
            env.set_goal(goal=sample_goal(env, args.min_goal_dist, args.hard_turn_prob), min_goal_dist=args.min_goal_dist)
            obs = env._get_obs()
            for goal_idx in range(args.goals_per_episode):
                seg_obs, seg_act = [], []
                info = {}
                min_seen = env.RAY_RANGE
                for step in range(args.max_steps_per_goal):
                    action = heuristic_action(env)
                    repeats = sample_weight(obs, action, step, args.boundary_steps, args.weight_mode)
                    for _ in range(repeats):
                        seg_obs.append(obs.copy())
                        seg_act.append(action.copy())
                    next_obs, _, term, trunc, info = env.step(action)
                    min_seen = min(min_seen, float(info.get("min_laser", env.RAY_RANGE)))
                    obs = next_obs
                    if term or trunc or step + 1 >= args.max_steps_per_goal:
                        break
                arrived = bool(info.get("arrived", False))
                collision = bool(info.get("collision", False))
                if arrived:
                    stats["arrived"] += 1
                elif collision:
                    stats["collision"] += 1
                else:
                    stats["timeout"] += 1
                keep = arrived and min_seen >= args.min_clearance
                if keep:
                    obs_rows.extend(seg_obs)
                    act_rows.extend(seg_act)
                    stats["kept_segments"] += 1
                    stats["weighted_samples"] += len(seg_obs)
                else:
                    stats["dropped_segments"] += 1
                stats["raw_samples"] += step + 1
                if goal_idx + 1 >= args.goals_per_episode:
                    break
                if not arrived:
                    obs, _ = env.reset()
                env.set_goal(goal=sample_goal(env, args.min_goal_dist, args.hard_turn_prob), min_goal_dist=args.min_goal_dist)
                obs = env._get_obs()
        finally:
            env.close()
        if (ep + 1) % max(1, args.report_every) == 0:
            print(
                f"[collect] ep={ep + 1}/{args.episodes} kept={stats['kept_segments']} "
                f"drop={stats['dropped_segments']} samples={stats['weighted_samples']}"
            )
    if not obs_rows:
        raise RuntimeError("No successful continuous-goal samples collected")
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
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--scene", default="boxes", choices=["mixed", "corridor", "corner", "boxes"])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--goals-per-episode", type=int, default=10)
    ap.add_argument("--max-steps-per-goal", type=int, default=900)
    ap.add_argument("--min-goal-dist", type=float, default=1.8)
    ap.add_argument("--hard-turn-prob", type=float, default=0.45)
    ap.add_argument("--boundary-steps", type=int, default=60)
    ap.add_argument("--min-clearance", type=float, default=0.05)
    ap.add_argument("--weight-mode", default="balanced", choices=["balanced", "aggressive"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_name = time.strftime("bc_multigoal_%Y%m%d_%H%M%S")
    out_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    out = args.out or os.path.join(out_dir, "bc_multigoal_dataset.npz")
    obs, actions, stats = collect(args)
    summary = describe(obs, actions)
    np.savez_compressed(out, obs=obs, actions=actions)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "collect": stats, "summary": summary, "dataset": out}, f, indent=2)
    print(f"[collect] saved {out}")
    print(f"[collect] stats={stats}")
    print(f"[collect] summary={summary}")


if __name__ == "__main__":
    main()
