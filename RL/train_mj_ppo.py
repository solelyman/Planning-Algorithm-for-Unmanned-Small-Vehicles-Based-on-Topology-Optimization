#!/usr/bin/env python3
"""
MuJoCo 纯激光 PPO 课程训练 (SB3) — 收敛配方
============================================
  obs     : 激光 90 束 + [dist/15, beta/pi]  (92 维向量, MlpPolicy)
  action  : (a0, a1) in [-1,1] -> lin in [0,1]m/s, ang in [-2,2]rad/s
  reward  : GTRL = (distOld-dist)*20 + v*2 - |w| - |Δw|/4 + 到达+100 - 碰撞-100
  课程    : 空地图(n_obstacles=None) -> 随机 n 障碍 -> 目标距离从近到远
  PPO     : lr 3e-4, clip 0.2, ent 0.005, gamma 0.99, gae 0.95,
            n_envs=32, n_steps=2048, batch 2048, epochs 10, net [256,256]
用法:
  # 阶段1: 空地图, 目标 2~3m (先学会直奔目标)
  python train_mj_ppo.py --tag stage1_empty --goal-dist 2,3
  # 阶段2: 1 个障碍
  python train_mj_ppo.py --tag stage2_obs1 --n-obs 1 --goal-dist 2,3 --load 模型.zip
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from end2end_env import End2EndNavEnv

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ppo_curriculum")


def make_env(seed, n_obs, goal_dist, goal_margin, detour_ratio=0.6):
    def _f():
        return End2EndNavEnv(seed=seed, n_obstacles=n_obs, goal_dist=goal_dist,
                             obs_mode="laser", goal_margin=goal_margin, detour_ratio=detour_ratio)
    return _f


class ArrivalEvalCallback(BaseCallback):
    """每 eval_every 步, 在主进程用当前策略跑 n_episodes 评估到达率/碰撞率.
    记录最近评估结果供门槛判定; 若 min_arrival 指定, 到达率达标才会停止 (配合 learn 循环)
    每次评估后实时保存模型到 eval_save_path (供 live GUI 边训练边看最新策略)"""

    def __init__(self, n_obs, goal_dist, goal_margin=0.15, eval_every=98304, n_episodes=20, seed=9999, verbose=0, detour_ratio=0.6, eval_save_path=None):
        super().__init__(verbose)
        self.n_obs = n_obs
        self.goal_dist = goal_dist
        self.goal_margin = goal_margin
        self.detour_ratio = detour_ratio
        self.eval_every = eval_every
        self.n_episodes = n_episodes
        self.seed = seed
        self.eval_env = None
        self.last_eval_step = 0
        self.last_arrival = 0.0
        self.last_collision = 0.0
        self.eval_save_path = eval_save_path

    def _on_training_start(self):
        self.eval_env = DummyVecEnv([make_env(self.seed, self.n_obs, self.goal_dist, self.goal_margin, self.detour_ratio)])

    def _on_step(self):
        steps = int(self.model.num_timesteps)
        if steps - self.last_eval_step < self.eval_every:
            return True
        self.last_eval_step = steps
        arr = col = out = 0
        rews, lens = [], []
        for i in range(self.n_episodes):
            obs = self.eval_env.reset()
            ep_r, s, done, ep_collided = 0.0, 0, False, False
            while not np.any(done) and s < 1800:   # 与 MAX_STEPS 一致 (4-7m+绕障需要更长)
                act, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, info = self.eval_env.step(act)
                ep_r += float(r[0])
                s += 1
                inf = info[0]
                if "arrived" in inf:
                    ep_collided |= bool(inf["collided"])  # episode 是否撞过 (非次数)
            arr += int(np.any(inf["arrived"]))
            col += int(ep_collided)
            out += int(np.any(inf["out_of_bounds"]))
            rews.append(ep_r)
            lens.append(s)
        n = max(len(rews), 1)
        self.last_arrival = arr / n
        self.last_collision = col / n
        print(f"[eval] 步数={steps} 到达率={self.last_arrival:.2f} 碰撞={self.last_collision:.2f} "
              f"出界={out / n:.2f} 均长={np.mean(lens):.0f} 均奖={np.mean(rews):.1f}", flush=True)
        # 实时保存最新模型, live GUI 每几秒 reload 就能看到训练中策略
        if self.eval_save_path:
            try:
                self.model.save(self.eval_save_path)
                print(f"[eval] 已实时保存 {self.eval_save_path}", flush=True)
            except Exception as e:
                print(f"[eval] 实时保存失败: {e}", flush=True)
        return True

    def _on_training_end(self):
        if self.eval_env is not None:
            self.eval_env.close()


def show_rollout_gui(model, args, lo, hi):
    """训练完弹 MuJoCo GUI 实时展示当前模型 rollout (阻塞直到关窗).
    固定俯视可缩放视角 (TRACKING); 目标金色球实时标记; 每局结束自动 reset 新起点"""
    import mujoco
    import mujoco.viewer as mjv
    env = End2EndNavEnv(seed=42, n_obstacles=args.n_obs, goal_dist=(lo, hi),
                        goal_margin=args.goal_margin, detour_ratio=args.detour_ratio, obs_mode="laser")
    obs, _ = env.reset()
    ep_r, ep_len, done = 0.0, 0, False
    print(f"[gui] 展示训练结果 | 场景={args.n_obs} 障碍数={len(env._obs)} "
          f"起点={env.data.qpos[0:2].round(2)} 目标(金球)={env.goal.round(2)}", flush=True)
    print("[gui] 关闭 MuJoCo 窗口继续训练; 鼠标滚轮缩放/拖拽旋转", flush=True)
    with mjv.launch_passive(env.model, env.data) as viewer:
        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 0
        cam.lookat = np.array([0.0, 0.0, 0.5])
        cam.distance = 13.0
        cam.azimuth = 90.0
        cam.elevation = 68.0
        while viewer.is_running():
            if done:
                inf = env.last_info if hasattr(env, "last_info") else {}
                print(f"[gui] 局结束: 步数={ep_len} 奖励={ep_r:.1f} 到达={inf.get('arrived', False)} "
                      f"碰撞={inf.get('collided', False)}", flush=True)
                obs, _ = env.reset()
                ep_r, ep_len, done = 0.0, 0, False
                print(f"[gui] 新起点={env.data.qpos[0:2].round(2)} 目标(金球)={env.goal.round(2)}", flush=True)
                continue
            act, _ = model.predict(obs, deterministic=True)
            obs, r, ter, trunc, inf = env.step(act)
            ep_r += float(r)
            ep_len += 1
            env.last_info = inf
            done = ter or trunc
            viewer.sync()
            time.sleep(0.03)
    print("[gui] 窗口已关闭", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--n-obs", default=None, help="None=空地图; nav2=固定nav2布局; 数字=随机n个box")
    ap.add_argument("--goal-dist", default="2,3")
    ap.add_argument("--goal-margin", type=float, default=0.15, help="目标距障碍最小距离 (0.15=允许贴障碍)")
    ap.add_argument("--detour-ratio", type=float, default=0.6,
                    help="目标被障碍直线遮挡(必须绕路)的比例: 0.6=60%局要绕路, 对齐 gz 部署分布")
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--load", default=None, help="续训模型 zip 路径")
    ap.add_argument("--init-bc", default=None, help="BC 权重 pt 路径 (注入 PPO 策略初始化)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-arrival", type=float, default=0.0,
                    help="到达率达标门槛: 每轮 learn 结束后若到达率<此值则用最近模型继续练, 达标才退出")
    ap.add_argument("--gui", action="store_true",
                    help="训练时开 MuJoCo GUI 实时展示策略 rollout (训练子进程同步推进)")
    ap.add_argument("--gui-speed", type=float, default=1.0, help="GUI 展示仿真倍率 (默认实时)")
    args = ap.parse_args()
    if args.n_obs is not None:
        try:
            args.n_obs = int(args.n_obs)
        except ValueError:
            pass  # 保留字符串 (如 'nav2')

    lo, hi = [float(x) for x in args.goal_dist.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    vec_env = SubprocVecEnv([make_env(100 + i, args.n_obs, (lo, hi), args.goal_margin, args.detour_ratio) for i in range(args.n_envs)])

    kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=os.path.join(OUT_DIR, "tb"),
        verbose=0,
        device=args.device,
    )
    if args.load and os.path.exists(args.load):
        model = PPO.load(args.load, **kwargs)
        print(f"[train] 续训: {args.load}", flush=True)
    else:
        model = PPO(**kwargs)

    if args.init_bc:
        import torch
        bc = torch.load(args.init_bc, map_location="cpu")["state_dict"]
        sd = model.policy.state_dict()
        # BC: net.0(92,256) -> policy_net.0; net.2(256,256) -> policy_net.2; net.4(256,2) -> action_net
        mapping = {
            "net.0.weight": "mlp_extractor.policy_net.0.weight",
            "net.0.bias": "mlp_extractor.policy_net.0.bias",
            "net.2.weight": "mlp_extractor.policy_net.2.weight",
            "net.2.bias": "mlp_extractor.policy_net.2.bias",
            "net.4.weight": "action_net.weight",
            "net.4.bias": "action_net.bias",
        }
        for bk, pk in mapping.items():
            if bk in bc and pk in sd and bc[bk].shape == sd[pk].shape:
                sd[pk].copy_(bc[bk])
            else:
                print(f"[train] 跳过注入 {bk} -> {pk}", flush=True)
        print(f"[train] BC 权重已注入 PPO 策略: {args.init_bc}", flush=True)

    eval_cb = ArrivalEvalCallback(args.n_obs, (lo, hi), goal_margin=args.goal_margin, seed=8888,
                                  detour_ratio=args.detour_ratio,
                                  eval_save_path=os.path.join(OUT_DIR, f"{args.tag}_ppo.zip"))
    print(f"[train] {args.tag} | n_obs={args.n_obs} goal_dist=({lo},{hi}) "
          f"n_envs={args.n_envs} lr={args.lr} min_arrival={args.min_arrival}", flush=True)
    try:
        round_no = 0
        while True:
            round_no += 1
            # 每轮 learn 前重置 eval 计数, 使每轮都完整评估一次
            eval_cb.last_eval_step = -1
            eval_cb.last_arrival = 0.0
            model.learn(total_timesteps=args.timesteps, callback=eval_cb)
            path = os.path.join(OUT_DIR, f"{args.tag}_ppo.zip")
            model.save(path)
            arr = eval_cb.last_arrival
            print(f"[round{round_no}] 保存 {path} | 到达率={arr:.2f} | 累计用时 {(time.time() - t0) / 60:.1f}min", flush=True)
            if args.gui:
                # 训练后开 GUI 实时展示当前模型 rollout (阻塞直到关窗, 可看清训练效果)
                show_rollout_gui(model, args, lo, hi)
            if arr >= args.min_arrival:
                print(f"[train] 到达率 {arr:.2f} >= 门槛 {args.min_arrival:.2f}, 达标进入下一阶段", flush=True)
                break
            if args.min_arrival <= 0:
                break
            print(f"[train] 到达率 {arr:.2f} < 门槛 {args.min_arrival:.2f}, 继续训练 {args.timesteps} 步...", flush=True)
            # 用刚保存的最新模型续训下一轮 (模型对象不变, 继续 learn 即可)
    finally:
        path = os.path.join(OUT_DIR, f"{args.tag}_ppo.zip")
        model.save(path)
        vec_env.close()
        print(f"[train] 已保存 {path} | 用时 {(time.time() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
