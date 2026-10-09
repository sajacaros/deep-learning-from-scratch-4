"""저장한 PPO 체크포인트마다 여러 판 싸워 승률을 잰다. evaluate.py의 PPO판이다.

  uv run python sf2/evaluate_ppo.py runs/ppo --episodes 200 --workers 60 --out runs/eval_ppo.csv
  uv run python sf2/evaluate_ppo.py runs/ppo/ppo_2500000_steps.zip --deterministic   # 파일을 직접 줘도 된다

PPO의 정책은 원래 확률적이라 기본은 정책에서 행동을 뽑는다(참고 프로젝트의 test.py와 같음).
--deterministic을 주면 버튼마다 누를 확률이 0.5를 넘을 때만 누른다(DQN의 ε=0에 해당).
evaluate.py와 같이 판마다 처음 0~30걸음은 무작위 행동을 해서 출발 상황을 흔든다.
결과 csv의 열도 evaluate.py와 같아서 같은 방식으로 그래프를 그릴 수 있다.
결과 파일 옆에 <out>_buttons.csv로 체크포인트별 버튼을 누른 비율을 남긴다(무작위 시작 구간은 빼고).
"""
import argparse
import glob
import math
import multiprocessing as mp
import os
import re
import zlib

import numpy as np


def evaluate(job):
    import torch
    from stable_baselines3 import PPO
    from sf2_env import make_env

    run, ckpt, episodes, max_random, seed, deterministic = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(discrete=False)
    # 학습률·클리핑 스케줄은 평가에 필요 없으므로 상수로 바꿔서 읽는다
    model = PPO.load(ckpt, device='cpu', custom_objects={
        'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0})

    m = re.search(r'ppo_(\d+)_steps\.zip', ckpt)
    step = int(m.group(1)) if m else os.path.splitext(os.path.basename(ckpt))[0].replace('ppo_', '')
    rows = []
    for episode in range(episodes):
        state, info = env.reset()
        done, total_reward, length = False, 0.0, 0
        for _ in range(rng.integers(0, max_random + 1)):  # random start
            state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            length += 1
            done = terminated or truncated
            if done:
                break
        presses = np.zeros(env.action_space.n, dtype=np.int64)
        n_actions, repeats, prev = 0, 0, None
        while not done:
            action, _ = model.predict(state, deterministic=deterministic)
            action = action.astype(np.int8)  # 버튼마다 0 또는 1
            presses += action
            n_actions += 1
            repeats += prev is not None and np.array_equal(action, prev)
            prev = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1
        # 마지막 셋은 csv에 쓰지 않고 버튼 통계로만 쓴다
        rows.append((run, step, episode, int(info['enemy_hp'] < 0), total_reward,
                     info['agent_hp'], info['enemy_hp'], length, presses, n_actions, repeats))
    env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max-random', type=int, default=30)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--workers', type=int, default=40)
    p.add_argument('--out', default='runs/eval_ppo.csv')
    args = p.parse_args()

    jobs = []
    for run in args.runs:
        if run.endswith('.zip'):
            ckpts, run = [run], os.path.dirname(run)
        else:
            ckpts = glob.glob(os.path.join(run, 'ppo_*.zip'))
        for ckpt in ckpts:
            # 한 체크포인트를 여러 조각으로 나눠 프로세스를 고르게 쓴다
            n_chunks = max(1, args.episodes // 10)
            for c in range(n_chunks):
                n = args.episodes // n_chunks + (c < args.episodes % n_chunks)
                jobs.append((os.path.basename(run.rstrip('/')), ckpt, n, args.max_random,
                             zlib.crc32(f'{ckpt}:{c}'.encode()), args.deterministic))
    print(f'{len(jobs)} jobs, {args.workers} workers', flush=True)

    rows = []
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(evaluate, jobs), 1):
            rows += r
            if i % 20 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)}', flush=True)

    rows.sort(key=lambda r: (r[0], str(r[1]).zfill(9), r[2]))
    with open(args.out, 'w') as f:
        f.write('run,step,episode,win,reward,agent_hp,enemy_hp,length\n')
        for r in rows:
            f.write(','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r[:8]) + '\n')

    # 체크포인트별 버튼을 누른 비율과 직전과 같은 조합을 고른 비율
    from sf2_env import make_env
    env = make_env(discrete=False)
    buttons = env.unwrapped.buttons
    env.close()
    acts = {}
    for r in rows:
        a = acts.setdefault((r[0], r[1]), [np.zeros(len(buttons)), 0, 0])
        a[0] += r[8]
        a[1] += r[9]
        a[2] += r[10]
    btn_out = os.path.splitext(args.out)[0] + '_buttons.csv'
    with open(btn_out, 'w') as f:
        f.write('run,step,' + ','.join(buttons) + ',same_as_prev\n')
        for (run, step), (presses, n, rep) in acts.items():
            f.write(f'{run},{step},' + ','.join(f'{x / n:.4f}' for x in presses) + f',{rep / n:.4f}\n')

    # 체크포인트별 승률 ± 표준오차
    stats = {}
    for run, step, _, win, reward, *_ in rows:  # noqa
        stats.setdefault((run, step), []).append((win, reward))
    print(f'\n{"run":16s}{"step":>9s}{"win":>14s}{"reward":>9s}{"same":>7s}')
    for (run, step), v in sorted(stats.items(), key=lambda kv: (kv[0][0], str(kv[0][1]).zfill(9))):
        w = np.mean([x[0] for x in v])
        se = math.sqrt(w * (1 - w) / len(v))
        _, n, rep = acts[(run, step)]
        print(f'{run:16s}{str(step):>9s}{w:8.3f}±{se:.3f}{np.mean([x[1] for x in v]):9.3f}{rep / n:7.2f}')


if __name__ == '__main__':
    main()
