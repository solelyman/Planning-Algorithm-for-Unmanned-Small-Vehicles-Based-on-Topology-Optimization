#!/usr/bin/env python3
"""
BC 预训练 (UGV, 空地图) — teacher = A*(vprm_plan) + pure pursuit(_follow_path_action)
=============================================================================================
流程:
  1) 用 env 自带的 V-PRM(A*) 全局路径 + pure pursuit 老师跑 N 条轨迹, 收集 (obs, action)
  2) MLP 回归 (92 -> 256 -> 256 -> 2, tanh 输出), MSE loss
  3) 保存 BC 权重 + 测试到达率 (BC 直接部署)
用法:
  python bc_pretrain.py --trajs 200 --goal-dist 2,3
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from end2end_env import End2EndNavEnv

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OBS_DIM = 90 + 4   # laser 90 + [dist/15, beta/pi, lin/MAX, ang/MAX]


class BCMlp(nn.Module):
    """与 SB3 PPO MlpPolicy 完全同构 (Tanh 激活, 3层):
    net.0(92,256) Tanh net.2(256,256) Tanh net.4(256,2) -> 可直接注入 PPO policy_net"""
    def __init__(self, n_in=OBS_DIM, hidden=256, n_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_out), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def collect_demos(seed, n_trajs, n_obs, goal_dist, goal_margin=0.15):
    """A* + pure pursuit 老师演示数据"""
    env = End2EndNavEnv(seed=seed, n_obstacles=n_obs, goal_dist=goal_dist,
                        obs_mode="laser", goal_margin=goal_margin)
    obs_all, act_all = [], []
    arr = 0
    for _ in range(n_trajs):
        env.reset(seed=seed + _)
        path = env.vprm_demo_path()
        if path is None or len(path) < 2:
            continue
        for _s in range(env.MAX_STEPS):
            act = env._follow_path_action(path)
            obs_all.append(env._get_obs().astype(np.float32))
            act_all.append(act)
            _, _, term, trunc, _ = env.step(act)
            if term:
                arr += 1
                break
            if trunc:
                break
    env.close()
    return np.stack(obs_all), np.stack(act_all), arr / max(n_trajs, 1)


def evaluate(model, n=15, seed=7777, n_obs=None, goal_dist=(2.0, 3.0), goal_margin=0.15):
    env = End2EndNavEnv(seed=seed, n_obstacles=n_obs, goal_dist=goal_dist,
                        obs_mode="laser", goal_margin=goal_margin)
    arr = col = out = 0
    lens = []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            obs, _ = env.reset(seed=seed + i)
            s, done, trunc = 0, False, False
            while not done and not trunc and s < env.MAX_STEPS:
                o = torch.from_numpy(obs).float().unsqueeze(0)
                a = model(o).squeeze(0).numpy()
                obs, r, done, trunc, info = env.step(a)
                s += 1
            lens.append(s)
            arr += int(info["arrived"])
            col += int(info["collided"])
            out += int(info["out_of_bounds"])
    env.close()
    return arr / n, col / n, out / n, float(np.mean(lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajs", type=int, default=300)
    ap.add_argument("--n-obs", default=None, help="None=空地图; nav2=固定nav2布局; 数字=随机n个box")
    ap.add_argument("--goal-dist", default="2,3")
    ap.add_argument("--goal-margin", type=float, default=0.15, help="目标距障碍最小距离")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tag", default="bc_ugv_empty")
    ap.add_argument("--data", default=None, help="从已收集 npz 数据训练 (跳过在线收集)")
    args = ap.parse_args()
    if args.n_obs is not None:
        try:
            args.n_obs = int(args.n_obs)
        except ValueError:
            pass  # 保留字符串 (如 'nav2')
    lo, hi = [float(x) for x in args.goal_dist.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[bc] 收集 teacher 演示 (A* + pure pursuit)...", flush=True)
    t0 = time.time()
    if args.data and os.path.exists(args.data):
        d = np.load(args.data)
        obs, act = d["obs"], d["act"]
        arr = float(d.get("n_trajs", 0)) if "n_trajs" in d.files else 0.0
        print(f"[bc] 从 {args.data} 加载: {obs.shape[0]} 样本 ({time.time()-t0:.0f}s)", flush=True)
    else:
        obs, act, arr = collect_demos(seed=1000, n_trajs=args.trajs, n_obs=args.n_obs,
                                      goal_dist=(lo, hi), goal_margin=args.goal_margin)
        print(f"[bc] 演示: {obs.shape[0]} 样本, teacher 到达率={arr:.2f} ({time.time()-t0:.0f}s)", flush=True)

    model = BCMlp()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    n = obs.shape[0]
    for ep in range(args.epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            x = torch.from_numpy(obs[idx]).float()
            y = torch.from_numpy(act[idx]).float()
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 10 == 0:
            rate, c, o, ml = evaluate(model, n=10, n_obs=args.n_obs, goal_dist=(lo, hi), goal_margin=args.goal_margin)
            print(f"[bc] epoch {ep+1:3d} loss={tot / n:.5f} 到达率={rate:.2f} "
                  f"碰撞={c:.2f} 出界={o:.2f} 均长={ml:.0f}", flush=True)

    rate, c, o, ml = evaluate(model, n=20, n_obs=args.n_obs, goal_dist=(lo, hi), goal_margin=args.goal_margin)
    print(f"[bc] 最终: 到达率={rate:.2f} 碰撞={c:.2f} 出界={o:.2f} 均长={ml:.0f}", flush=True)
    path = os.path.join(OUT_DIR, f"{args.tag}.pt")
    torch.save({"state_dict": model.state_dict(), "config": dict(trajs=args.trajs,
               n_obs=args.n_obs, goal_dist=[lo, hi])}, path)
    print(f"[bc] 已保存 {path}", flush=True)


if __name__ == "__main__":
    main()
