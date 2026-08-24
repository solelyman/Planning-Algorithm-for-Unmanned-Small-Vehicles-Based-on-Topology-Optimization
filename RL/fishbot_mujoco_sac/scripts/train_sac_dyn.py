#!/usr/bin/env python
"""SAC + expert replay training for the true-dynamics MuJoCo tracked robot."""
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
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import polyak_update

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_warmstart_dyn import collect_dataset
from env_dyn import TrackedNavDynEnv
from eval_dyn import run_episode, summarize
from view_dyn import heuristic_action


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
RUNS_DIR = os.path.join(PROJECT_DIR, "runs_dyn")


def make_env(seed, stage, scene):
    def _f():
        return TrackedNavDynEnv(seed=seed, stage=stage, scene=scene)

    return _f


def sac_actor_mean(model, obs_tensor):
    return model.actor(obs_tensor, deterministic=True)


def set_actor_log_std(model, value=-4.0):
    """Make SAC exploration conservative after BC pretraining."""
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(float(value))


def install_deterministic_rollout(model):
    """Use deterministic actor actions while collecting online SAC rollouts."""

    def deterministic_sample_action(learning_starts, action_noise=None, n_envs=1):
        assert model._last_obs is not None, "self._last_obs was not set"
        unscaled_action, _ = model.predict(model._last_obs, deterministic=True)
        scaled_action = model.policy.scale_action(unscaled_action)
        if action_noise is not None:
            scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
        return model.policy.unscale_action(scaled_action), scaled_action

    model._sample_action = deterministic_sample_action
    return model


def critic_only_update(model, gradient_steps, batch_size):
    """Warm the critic on replay data before allowing actor improvement."""
    model.policy.set_training_mode(True)
    lr = model.lr_schedule(1)
    for group in model.critic.optimizer.param_groups:
        group["lr"] = lr
    losses = []
    for gradient_step in range(gradient_steps):
        replay_data = model.replay_buffer.sample(batch_size, env=model._vec_normalize_env)
        discounts = replay_data.discounts if replay_data.discounts is not None else model.gamma
        with torch.no_grad():
            next_actions, next_log_prob = model.actor.action_log_prob(replay_data.next_observations)
            next_q_values = torch.cat(model.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            ent_coef = torch.tensor(0.0, device=model.device)
            if getattr(model, "ent_coef_optimizer", None) is None:
                ent_coef = model.ent_coef_tensor
            elif getattr(model, "log_ent_coef", None) is not None:
                ent_coef = torch.exp(model.log_ent_coef.detach())
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

        current_q_values = model.critic(replay_data.observations, replay_data.actions)
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        model.critic.optimizer.zero_grad()
        critic_loss.backward()
        model.critic.optimizer.step()
        if gradient_step % model.target_update_interval == 0:
            polyak_update(model.critic.parameters(), model.critic_target.parameters(), model.tau)
            polyak_update(model.batch_norm_stats, model.batch_norm_stats_target, 1.0)
        losses.append(float(critic_loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def install_guided_sac_train(model, bc_obs, bc_actions, bc_coef=1.0, actor_rl_coef=1.0, seed=0):
    """Patch this SAC instance so every actor update includes expert MSE.

    Keeping the saved object as plain SB3 SAC makes deployment/loading simpler;
    this guidance exists only while the training process is running.
    """
    rng = np.random.default_rng(seed)
    obs_t = torch.as_tensor(bc_obs, dtype=torch.float32, device=model.device)
    act_t = torch.as_tensor(bc_actions, dtype=torch.float32, device=model.device)

    def guided_train(gradient_steps: int, batch_size: int = 64) -> None:
        model.policy.set_training_mode(True)
        optimizers = [model.actor.optimizer, model.critic.optimizer]
        if model.ent_coef_optimizer is not None:
            optimizers += [model.ent_coef_optimizer]
        model._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, guidance_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = model.replay_buffer.sample(batch_size, env=model._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else model.gamma

            if model.use_sde:
                model.actor.reset_noise()

            actions_pi, log_prob = model.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if model.ent_coef_optimizer is not None and model.log_ent_coef is not None:
                ent_coef = torch.exp(model.log_ent_coef.detach())
                ent_coef_loss = -(model.log_ent_coef * (log_prob + model.target_entropy).detach()).mean()
                ent_coef_losses.append(float(ent_coef_loss.detach().cpu()))
            else:
                ent_coef = model.ent_coef_tensor

            ent_coefs.append(float(ent_coef.detach().cpu()))
            if ent_coef_loss is not None and model.ent_coef_optimizer is not None:
                model.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                model.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = model.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(model.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = model.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(float(critic_loss.detach().cpu()))

            model.critic.optimizer.zero_grad()
            critic_loss.backward()
            model.critic.optimizer.step()

            q_values_pi = torch.cat(model.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            expert_idx = rng.integers(0, len(obs_t), size=min(batch_size, len(obs_t)))
            expert_idx = torch.as_tensor(expert_idx, dtype=torch.long, device=model.device)
            pred_expert = model.actor(obs_t[expert_idx], deterministic=True)
            guidance_loss = F.mse_loss(pred_expert, act_t[expert_idx])
            total_actor_loss = actor_rl_coef * actor_loss + bc_coef * guidance_loss
            actor_losses.append(float(actor_loss.detach().cpu()))
            guidance_losses.append(float(guidance_loss.detach().cpu()))

            model.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            model.actor.optimizer.step()

            if gradient_step % model.target_update_interval == 0:
                polyak_update(model.critic.parameters(), model.critic_target.parameters(), model.tau)
                polyak_update(model.batch_norm_stats, model.batch_norm_stats_target, 1.0)

        model._n_updates += gradient_steps
        model.logger.record("train/n_updates", model._n_updates, exclude="tensorboard")
        model.logger.record("train/ent_coef", np.mean(ent_coefs))
        model.logger.record("train/actor_loss", np.mean(actor_losses))
        model.logger.record("train/critic_loss", np.mean(critic_losses))
        model.logger.record("train/guidance_loss", np.mean(guidance_losses))
        model.logger.record("train/actor_rl_coef", actor_rl_coef)
        if len(ent_coef_losses) > 0:
            model.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    model.train = guided_train
    return model


def bc_update_actor(model, obs, actions, epochs, batch_size, lr, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.actor.parameters(), lr=lr)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=model.device)
    losses = []
    model.actor.train()
    for _ in range(epochs):
        idx = rng.permutation(len(obs))
        epoch_losses = []
        for start in range(0, len(idx), batch_size):
            batch = idx[start : start + batch_size]
            pred = sac_actor_mean(model, obs_t[batch])
            loss = torch.nn.functional.mse_loss(pred, act_t[batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return float(np.mean(losses)) if losses else 0.0


def collect_expert_transitions(stage, scene, episodes, max_steps, seed, continuous_goals, goals_per_reset, min_clearance):
    obs_rows, next_rows, act_rows, rew_rows, done_rows = [], [], [], [], []
    stats = {"arrived": 0, "collision": 0, "timeout": 0, "kept_segments": 0, "dropped_segments": 0, "samples": 0}
    for ep in range(episodes):
        env = TrackedNavDynEnv(stage=stage, scene=scene, seed=seed + ep)
        try:
            obs, _ = env.reset()
            if continuous_goals:
                env.set_goal(min_goal_dist=1.8)
                obs = env._get_obs()
            goals_done = 0
            while True:
                seg_obs, seg_next, seg_act, seg_rew, seg_done = [], [], [], [], []
                min_seen = env.RAY_RANGE
                info = {}
                for _ in range(max_steps):
                    action = heuristic_action(env)
                    next_obs, reward, term, trunc, info = env.step(action)
                    done = bool(term or trunc)
                    seg_obs.append(obs.copy())
                    seg_next.append(next_obs.copy())
                    seg_act.append(action.copy())
                    seg_rew.append(float(reward))
                    seg_done.append(done)
                    min_seen = min(min_seen, float(info.get("min_laser", env.RAY_RANGE)))
                    obs = next_obs
                    if done:
                        break
                arrived = bool(info.get("arrived", False))
                collision = bool(info.get("collision", False))
                if arrived:
                    stats["arrived"] += 1
                elif collision:
                    stats["collision"] += 1
                else:
                    stats["timeout"] += 1
                if arrived and min_seen >= min_clearance:
                    obs_rows.extend(seg_obs)
                    next_rows.extend(seg_next)
                    act_rows.extend(seg_act)
                    rew_rows.extend(seg_rew)
                    done_rows.extend(seg_done)
                    stats["kept_segments"] += 1
                    stats["samples"] += len(seg_obs)
                else:
                    stats["dropped_segments"] += 1

                goals_done += 1
                if not continuous_goals or goals_done >= goals_per_reset:
                    break
                if not arrived:
                    obs, _ = env.reset()
                env.set_goal(min_goal_dist=1.8)
                obs = env._get_obs()
        finally:
            env.close()
    if not obs_rows:
        raise RuntimeError("No successful expert transitions collected")
    return (
        np.asarray(obs_rows, dtype=np.float32),
        np.asarray(next_rows, dtype=np.float32),
        np.asarray(act_rows, dtype=np.float32),
        np.asarray(rew_rows, dtype=np.float32),
        np.asarray(done_rows, dtype=bool),
        stats,
    )


def add_to_replay(model, obs, next_obs, actions, rewards, dones):
    infos = [{}]
    for o, no, a, r, d in zip(obs, next_obs, actions, rewards, dones):
        model.replay_buffer.add(
            o.reshape((1, -1)),
            no.reshape((1, -1)),
            a.reshape((1, -1)),
            np.asarray([r], dtype=np.float32),
            np.asarray([d], dtype=bool),
            infos,
        )


def evaluate_policy(model, stage, scene, episodes, seed):
    rows = []
    for ep in range(episodes):
        env = TrackedNavDynEnv(stage=stage, scene=scene, seed=seed + ep)
        try:
            rows.append(run_episode(env, model))
        finally:
            env.close()
    return summarize(rows)


def write_curve(out_dir, history):
    if not history:
        return
    xs = [h["timesteps"] for h in history]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(xs, [h["arrived_rate"] for h in history], label="arrived")
    axs[0].plot(xs, [h["collision_rate"] for h in history], label="collision")
    axs[0].plot(xs, [h["timeout_rate"] for h in history], label="timeout")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].legend()
    axs[0].set_title("Eval rates")
    axs[1].plot(xs, [h["mean_reward"] for h in history], label="reward")
    axs[1].plot(xs, [h["bc_loss"] for h in history], label="bc_loss")
    axs[1].legend()
    axs[1].set_title("SAC + guidance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sac_curve.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--scene", default="boxes", choices=["mixed", "corridor", "corner", "boxes"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=None, help="Existing BC .npz with obs/actions for actor guidance.")
    ap.add_argument("--transition-dataset", default=None, help="Existing SAC transition .npz.")
    ap.add_argument("--expert-episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--continuous-goals", action="store_true")
    ap.add_argument("--goals-per-reset", type=int, default=3)
    ap.add_argument("--min-clearance", type=float, default=0.08)
    ap.add_argument("--bc-pretrain-epochs", type=int, default=8)
    ap.add_argument("--bc-guidance-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--bc-lr", type=float, default=3e-4)
    ap.add_argument("--sac-lr", type=float, default=3e-4)
    ap.add_argument("--actor-bc-coef", type=float, default=1.0)
    ap.add_argument("--actor-rl-coef", type=float, default=1.0)
    ap.add_argument("--critic-warmup-steps", type=int, default=0)
    ap.add_argument("--log-std-init", type=float, default=-4.0)
    ap.add_argument("--deterministic-rollout", action="store_true")
    ap.add_argument("--ent-coef", default="auto_0.2")
    ap.add_argument("--target-entropy", type=float, default=-1.0)
    ap.add_argument("--buffer-size", type=int, default=200000)
    ap.add_argument("--learning-starts", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--rl-chunk", type=int, default=12000)
    ap.add_argument("--gradient-steps", type=int, default=1)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument(
        "--save-round-models",
        action="store_true",
        help="Also save every round model; disabled by default to reduce disk use.",
    )
    ap.add_argument(
        "--save-final-model",
        action="store_true",
        help="Also save final_model.zip; best_model.zip is always saved.",
    )
    args = ap.parse_args()

    run_name = time.strftime("sac_%Y%m%d_%H%M%S")
    out_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    env = DummyVecEnv([make_env(args.seed, args.stage, args.scene)])
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.sac_lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.995,
        train_freq=(1, "step"),
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        seed=args.seed,
        device="cpu",
    )

    if args.dataset:
        d = np.load(args.dataset)
        bc_obs = d["obs"].astype(np.float32)
        bc_actions = d["actions"].astype(np.float32)
        dataset_stats = {"source": os.path.abspath(args.dataset), "samples": int(len(bc_obs))}
    else:
        bc_obs, bc_actions, dataset_stats = collect_dataset(
            args.stage,
            args.scene,
            args.expert_episodes,
            args.max_steps,
            args.seed + 1000,
            args.continuous_goals,
            success_only=True,
            min_clearance=args.min_clearance,
            goals_per_reset=args.goals_per_reset,
        )
    np.savez_compressed(os.path.join(out_dir, "bc_dataset.npz"), obs=bc_obs, actions=bc_actions)

    if args.transition_dataset:
        td = np.load(args.transition_dataset)
        tr_obs = td["obs"].astype(np.float32)
        tr_next = td["next_obs"].astype(np.float32)
        tr_actions = td["actions"].astype(np.float32)
        tr_rewards = td["rewards"].astype(np.float32)
        tr_dones = td["dones"].astype(bool)
        transition_stats = {"source": os.path.abspath(args.transition_dataset), "samples": int(len(tr_obs))}
    else:
        tr_obs, tr_next, tr_actions, tr_rewards, tr_dones, transition_stats = collect_expert_transitions(
            args.stage,
            args.scene,
            args.expert_episodes,
            args.max_steps,
            args.seed + 3000,
            args.continuous_goals,
            args.goals_per_reset,
            args.min_clearance,
        )
    np.savez_compressed(
        os.path.join(out_dir, "expert_transitions.npz"),
        obs=tr_obs,
        next_obs=tr_next,
        actions=tr_actions,
        rewards=tr_rewards,
        dones=tr_dones,
    )
    add_to_replay(model, tr_obs, tr_next, tr_actions, tr_rewards, tr_dones)

    with open(os.path.join(out_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"bc": dataset_stats, "transitions": transition_stats}, f, indent=2)
    print(f"[sac] BC dataset obs={bc_obs.shape} actions={bc_actions.shape} stats={dataset_stats}")
    print(f"[sac] expert replay obs={tr_obs.shape} actions={tr_actions.shape} stats={transition_stats}")
    print(f"[sac] replay size={model.replay_buffer.size()}")

    pre_loss = bc_update_actor(model, bc_obs, bc_actions, args.bc_pretrain_epochs, args.batch_size, args.bc_lr, args.seed)
    print(f"[sac] actor BC pretrain loss={pre_loss:.5f}")
    set_actor_log_std(model, args.log_std_init)
    print(f"[sac] actor log_std initialized to {args.log_std_init}")
    if args.critic_warmup_steps > 0:
        critic_loss = critic_only_update(model, args.critic_warmup_steps, args.batch_size)
        print(f"[sac] critic-only warmup steps={args.critic_warmup_steps} loss={critic_loss:.5f}")
    if args.deterministic_rollout:
        install_deterministic_rollout(model)
        print("[sac] deterministic rollout enabled")
    install_guided_sac_train(
        model,
        bc_obs,
        bc_actions,
        bc_coef=args.actor_bc_coef,
        actor_rl_coef=args.actor_rl_coef,
        seed=args.seed + 12345,
    )
    print(
        f"[sac] installed in-update actor guidance "
        f"bc_coef={args.actor_bc_coef} actor_rl_coef={args.actor_rl_coef}"
    )

    history = []
    best_score = -1e9
    try:
        for round_idx in range(args.rounds):
            print(f"[sac] round {round_idx + 1}/{args.rounds}: SAC chunk {args.rl_chunk} steps")
            model.learn(total_timesteps=args.rl_chunk, reset_num_timesteps=False, log_interval=10)
            bc_loss = bc_update_actor(
                model,
                bc_obs,
                bc_actions,
                args.bc_guidance_epochs,
                args.batch_size,
                args.bc_lr * 0.3,
                args.seed + 7000 + round_idx,
            )
            stats = evaluate_policy(model, args.stage, args.scene, args.eval_episodes, args.seed + 9000 + round_idx * 100)
            stats["round"] = round_idx + 1
            stats["timesteps"] = int(model.num_timesteps)
            stats["bc_loss"] = bc_loss
            score = stats["arrived_rate"] - stats["collision_rate"] - 0.2 * stats["timeout_rate"]
            stats["selection_score"] = score
            history.append(stats)
            with open(os.path.join(out_dir, "eval_history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            write_curve(out_dir, history)
            if args.save_round_models:
                model.save(os.path.join(out_dir, f"round_{round_idx + 1:02d}_model"))
            if score > best_score:
                best_score = score
                model.save(os.path.join(out_dir, "best_model"))
            print(
                f"[sac eval] round={round_idx + 1} arrived={stats['arrived_rate']:.2f} "
                f"collision={stats['collision_rate']:.2f} timeout={stats['timeout_rate']:.2f} "
                f"reward={stats['mean_reward']:.1f} bc_loss={bc_loss:.5f} score={score:.2f}"
            )
    finally:
        env.close()
    if args.save_final_model:
        model.save(os.path.join(out_dir, "final_model"))
    print(f"[sac] saved run to {out_dir}")


if __name__ == "__main__":
    main()
