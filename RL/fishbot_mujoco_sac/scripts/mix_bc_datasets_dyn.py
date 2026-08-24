#!/usr/bin/env python
"""Mix a stable BC dataset with a smaller corrective continuous-goal dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np


def load_npz(path):
    data = np.load(path)
    return data["obs"].astype(np.float32), data["actions"].astype(np.float32)


def summarize(obs, actions):
    extra = obs[:, 72:]
    return {
        "samples": int(len(obs)),
        "goal_ang_abs_gt_0_8": float(np.mean(np.abs(extra[:, 2]) > 0.8)),
        "guide_ang_abs_gt_0_8": float(np.mean(np.abs(extra[:, 1]) > 0.8)),
        "min_laser_lt_0_5": float(np.mean(extra[:, 5] < 0.5 / 4.0)),
        "min_laser_lt_0_3": float(np.mean(extra[:, 5] < 0.3 / 4.0)),
        "turn_abs_gt_0_8": float(np.mean(np.abs(actions[:, 1]) > 0.8)),
    }


def sample_rows(obs, actions, n, rng):
    if len(obs) <= n:
        return obs, actions
    idx = rng.choice(len(obs), size=n, replace=False)
    return obs[idx], actions[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-samples", type=int, default=80000)
    ap.add_argument("--extra-samples", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base_obs, base_act = load_npz(args.base)
    extra_obs, extra_act = load_npz(args.extra)
    base_obs, base_act = sample_rows(base_obs, base_act, args.base_samples, rng)
    extra_obs, extra_act = sample_rows(extra_obs, extra_act, args.extra_samples, rng)
    obs = np.concatenate([base_obs, extra_obs], axis=0)
    actions = np.concatenate([base_act, extra_act], axis=0)
    idx = rng.permutation(len(obs))
    obs = obs[idx]
    actions = actions[idx]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, obs=obs, actions=actions)
    stats = {
        "base": args.base,
        "extra": args.extra,
        "base_used": int(len(base_obs)),
        "extra_used": int(len(extra_obs)),
        "summary": summarize(obs, actions),
    }
    with open(os.path.splitext(args.out)[0] + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
