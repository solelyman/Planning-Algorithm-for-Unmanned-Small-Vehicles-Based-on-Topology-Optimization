#!/usr/bin/env python
"""Behavior-cloning warm start for TrackedNavDynEnv PPO policies."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_dyn import TrackedNavDynEnv
from eval_dyn import run_episode, summarize
from view_dyn import heuristic_action


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
RUNS_DIR = os.path.join(PROJECT_DIR, "runs_dyn")


def collect_dataset(
    stage,
    scene,
    episodes,
    max_steps,
    seed,
    continuous_goals,
    success_only=True,
    min_clearance=0.0,
    goals_per_reset=3,
):
    observations = []
    actions = []
    stats = {
        "arrived": 0,
        "collision": 0,
        "timeout": 0,
        "kept_segments": 0,
        "dropped_segments": 0,
        "kept_samples": 0,
        "dropped_samples": 0,
    }
    target_samples = episodes * max_steps
    for ep in range(episodes):
        env = TrackedNavDynEnv(stage=stage, scene=scene, seed=seed + ep)
        try:
            obs, _ = env.reset()
            if continuous_goals:
                env.set_goal(min_goal_dist=1.8)
                obs = env._get_obs()
            goals_done = 0
            while len(observations) < target_samples:
                segment_obs = []
                segment_actions = []
                min_seen = env.RAY_RANGE
                info = {}
                for _ in range(max_steps):
                    action = heuristic_action(env)
                    segment_obs.append(obs.copy())
                    segment_actions.append(action.copy())
                    obs, _, term, trunc, info = env.step(action)
                    min_seen = min(min_seen, float(info.get("min_laser", env.RAY_RANGE)))
                    if term or trunc:
                        break

                arrived = bool(info.get("arrived", False))
                collision = bool(info.get("collision", False))
                if arrived:
                    stats["arrived"] += 1
                elif collision:
                    stats["collision"] += 1
                else:
                    stats["timeout"] += 1

                keep = (arrived or not success_only) and min_seen >= min_clearance
                if keep:
                    observations.extend(segment_obs)
                    actions.extend(segment_actions)
                    stats["kept_segments"] += 1
                    stats["kept_samples"] += len(segment_obs)
                else:
                    stats["dropped_segments"] += 1
                    stats["dropped_samples"] += len(segment_obs)

                goals_done += 1
                if not continuous_goals or goals_done >= goals_per_reset or len(observations) >= target_samples:
                    break
                if not arrived:
                    obs, _ = env.reset()
                env.set_goal(min_goal_dist=1.8)
                obs = env._get_obs()
        finally:
            env.close()
    if not observations:
        raise RuntimeError("BC dataset is empty. Try more episodes, --keep-failures, or a lower --min-clearance.")
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32), stats


def actor_mean(policy, obs_tensor):
    features = policy.extract_features(obs_tensor, policy.pi_features_extractor)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    return torch.tanh(policy.action_net(latent_pi))


def eval_policy(model, stage, scene, episodes, seed):
    report = {}
    scenes = ["corridor", "corner", "boxes"] if scene == "all" else [scene]
    for scene_name in scenes:
        rows = []
        for ep in range(episodes):
            env = TrackedNavDynEnv(stage=stage, scene=scene_name, seed=seed + ep)
            rows.append(run_episode(env, model))
            env.close()
        report[scene_name] = summarize(rows)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="Existing PPO zip. Omit to initialize a new policy.")
    ap.add_argument("--dataset", default=None, help="Existing .npz with obs/actions. If set, skip collection.")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--scene", default="boxes", choices=["mixed", "corridor", "corner", "boxes", "all"])
    ap.add_argument("--episodes", type=int, default=160)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--continuous-goals", action="store_true")
    ap.add_argument("--keep-failures", action="store_true", help="Also train on collision/timeout teacher segments.")
    ap.add_argument("--min-clearance", type=float, default=0.0, help="Drop teacher segments whose min laser is below this value.")
    ap.add_argument("--goals-per-reset", type=int, default=3)
    ap.add_argument("--eval-episodes", type=int, default=8)
    args = ap.parse_args()

    run_name = time.strftime("bc_%Y%m%d_%H%M%S")
    out_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_scene = "mixed" if args.scene == "all" else args.scene
    env = TrackedNavDynEnv(stage=args.stage, scene=train_scene, seed=args.seed)
    if args.base:
        model = PPO.load(args.base, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
            seed=args.seed,
            device="cpu",
        )

    if args.dataset:
        data = np.load(args.dataset)
        obs = data["obs"].astype(np.float32)
        acts = data["actions"].astype(np.float32)
        teacher_stats = {"source": os.path.abspath(args.dataset), "samples": int(len(obs))}
    else:
        obs, acts, teacher_stats = collect_dataset(
            args.stage,
            train_scene,
            args.episodes,
            args.max_steps,
            args.seed + 1000,
            args.continuous_goals,
            success_only=not args.keep_failures,
            min_clearance=args.min_clearance,
            goals_per_reset=args.goals_per_reset,
        )
    np.savez_compressed(os.path.join(out_dir, "bc_dataset.npz"), obs=obs, actions=acts)
    print(f"[bc] collected obs={obs.shape} actions={acts.shape} teacher_stats={teacher_stats}")

    rng = np.random.default_rng(args.seed)
    opt = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
    losses = []
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    act_t = torch.as_tensor(acts, dtype=torch.float32, device=model.device)
    for epoch in range(args.epochs):
        idx = rng.permutation(len(obs))
        epoch_losses = []
        for start in range(0, len(idx), args.batch_size):
            batch = idx[start : start + args.batch_size]
            pred = actor_mean(model.policy, obs_t[batch])
            loss = torch.nn.functional.mse_loss(pred, act_t[batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(f"[bc] epoch={epoch + 1}/{args.epochs} loss={mean_loss:.5f}")

    model_path = os.path.join(out_dir, "bc_model")
    model.save(model_path)
    report = eval_policy(model, args.stage, args.scene, args.eval_episodes, args.seed + 5000)
    with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("actor MSE")
    ax.set_title("BC warm-start loss")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bc_loss.png"), dpi=140)
    plt.close(fig)
    env.close()

    print(f"[bc] saved model: {model_path}.zip")
    for scene_name, stats in report.items():
        print(
            f"[bc eval] {scene_name:8s} arrived={stats['arrived_rate']:.2f} "
            f"collision={stats['collision_rate']:.2f} timeout={stats['timeout_rate']:.2f} "
            f"reward={stats['mean_reward']:.1f}"
        )


if __name__ == "__main__":
    main()
