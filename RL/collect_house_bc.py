#!/usr/bin/env python3
"""
红房子专项 BC 数据收集 — 只留成功轨迹
  teacher = LOS/lookahead (_dwa_action) 沿 A* 大膨胀路径跟踪
  安全机制:
    - 卡死检测: 连续 40 步位移 < 0.02m -> 用当前位置重规划 A* (带 re-plan 缓存)
    - 安全倒车: _dwa_action 内前方过近时减速/倒车脱困
  只收集 arrived=True 的轨迹 (obs[94], act[2]) -> npz
用法: python collect_house_bc.py --n-episodes 60 --out ../models/bc_house.npz
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from end2end_env import End2EndNavEnv

MAX_STEPS = 1600


def run_episode(env, seed):
    """LOS 老师跑一局, 返回 (obs_list, act_list, info) 或 None(失败)"""
    env.reset(seed=seed)
    path = env.vprm_demo_path(start=env.data.qpos[0:2].copy())
    if path is None or len(path) < 2:
        return None
    obs_list, act_list = [], []
    stuck = 0
    prev_pos = env.data.qpos[0:2].copy()
    done = False
    for _s in range(MAX_STEPS):
        act = env._dwa_action(path)
        obs_list.append(env._get_obs().astype(np.float32))
        act_list.append(act.astype(np.float32))
        _, _, ter, trunc, inf = env.step(act)
        if inf["arrived"] or inf["collided"]:
            done = True
            break
        # 卡死检测: 连续 40 步基本不动 -> 重规划 (以当前位置为新起点)
        moved = float(np.linalg.norm(env.data.qpos[0:2] - prev_pos))
        prev_pos = env.data.qpos[0:2].copy()
        if moved < 0.02:
            stuck += 1
        else:
            stuck = 0
        if stuck >= 40:
            path = env.vprm_demo_path(start=env.data.qpos[0:2].copy())
            stuck = 0
            if path is None or len(path) < 2:
                break
        if trunc:
            done = True
            break
    if not done or not inf["arrived"]:
        return None
    return np.array(obs_list, np.float32), np.array(act_list, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bc_house.npz"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-obs", default="nav2_house")
    ap.add_argument("--goal-dist", default="3,5")
    ap.add_argument("--detour-ratio", type=float, default=1.0)
    args = ap.parse_args()
    lo, hi = [float(x) for x in args.goal_dist.split(",")]

    all_obs, all_act = [], []
    n_succ = 0
    t0 = time.time()
    for ep in range(args.n_episodes):
        env = End2EndNavEnv(seed=args.seed + ep * 37, n_obstacles=args.n_obs,
                            goal_dist=(lo, hi), detour_ratio=args.detour_ratio,
                            goal_margin=0.35, obs_mode="laser")
        res = run_episode(env, seed=args.seed + ep * 37)
        if res is not None:
            o, a = res
            all_obs.append(o)
            all_act.append(a)
            n_succ += 1
            print(f"[ep{ep}] 成功 轨迹{o.shape[0]}步", flush=True)
        else:
            print(f"[ep{ep}] 失败", flush=True)
        env.close()
        if (ep + 1) % 10 == 0:
            print(f"... {ep+1}/{args.n_episodes} 成功累计={n_succ} ({time.time()-t0:.0f}s)", flush=True)

    if not all_obs:
        print("[collect] 无成功轨迹! 降低难度或加 teacher 兜底")
        return
    obs_all = np.concatenate(all_obs)
    act_all = np.concatenate(all_act)
    np.savez(args.out, obs=obs_all, act=act_all, n_trajs=n_succ,
             n_obs=args.n_obs, goal_dist=[lo, hi], detour_ratio=args.detour_ratio)
    print(f"[collect] 完成: {n_succ}/{args.n_episodes} 成功轨迹, {obs_all.shape[0]} 步样本 "
          f"obs={obs_all.shape} act={act_all.shape}")
    print(f"[collect] 保存到 {args.out}")


if __name__ == "__main__":
    main()
