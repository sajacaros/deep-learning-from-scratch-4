"""저장한 체크포인트마다 여러 판 싸워 승률을 잰다(기본 ε=0).

  uv run python sf2/evaluate.py runs/dqn runs/double runs/dueling runs/double_dueling --episodes 100
  uv run python sf2/evaluate.py runs/dueling_lrdecay/qnet_final.pt ...   # 파일을 직접 줘도 된다

결과 파일 옆에 <out>_actions.csv로 체크포인트별 행동 횟수도 남긴다(무작위 시작 구간은 빼고 모델이 고른 것만).

에뮬레이터도 정책도 결정적이라 같은 지점에서 시작하면 매 판이 똑같이 흘러간다.
그래서 판마다 처음 0~30걸음은 무작위 행동을 해서(random start) 출발 상황을 흔든다.
(아무것도 누르지 않고 기다리기만 하면 기다린 걸음 수만큼, 즉 31가지 판밖에 나오지 않는다)
stable-retro는 한 프로세스에 에뮬레이터를 하나만 띄울 수 있어 체크포인트마다 프로세스를 따로 쓴다.
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
    from dqn import QNet
    from sf2_env import make_env

    run, ckpt, episodes, max_random, device, seed, epsilon = job
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    env = make_env()
    params = torch.load(ckpt, map_location=device)
    qnet = QNet(env.action_space.n, dueling=any(k.startswith('v.') for k in params)).to(device)
    qnet.load_state_dict(params)
    qnet.eval()

    m = re.search(r'qnet_(\d+)\.pt', ckpt)
    step = int(m.group(1)) if m else os.path.splitext(os.path.basename(ckpt))[0].replace('qnet_', '')
    rows = []
    for episode in range(episodes):
        state, info = env.reset()
        done, total_reward, length = False, 0.0, 0
        for _ in range(rng.integers(0, max_random + 1)):  # random start
            state, reward, terminated, truncated, info = env.step(int(rng.integers(env.action_space.n)))
            total_reward += reward
            length += 1
            done = terminated or truncated
            if done:
                break
        counts = np.zeros(env.action_space.n, dtype=np.int64)
        repeats, prev = 0, None
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                with torch.no_grad():
                    action = qnet(torch.from_numpy(state).unsqueeze(0).to(device)).argmax().item()
            counts[action] += 1
            repeats += action == prev
            prev = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1
        # 마지막 둘은 csv에 쓰지 않고 행동 통계로만 쓴다
        rows.append((run, step, episode, int(info['enemy_hp'] < 0), total_reward,
                     info['agent_hp'], info['enemy_hp'], length, counts, repeats))
    env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max-random', type=int, default=30)
    p.add_argument('--epsilon', type=float, default=0.0)  # Nature DQN의 평가는 0.05
    p.add_argument('--workers', type=int, default=40)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='runs/eval.csv')
    args = p.parse_args()

    jobs = []
    for run in args.runs:
        if run.endswith('.pt'):
            ckpts, run = [run], os.path.dirname(run)
        else:
            ckpts = glob.glob(os.path.join(run, 'qnet_[0-9]*.pt'))
        for ckpt in ckpts:
            # 한 체크포인트를 여러 조각으로 나눠 프로세스를 고르게 쓴다
            n_chunks = max(1, args.episodes // 10)
            for c in range(n_chunks):
                n = args.episodes // n_chunks + (c < args.episodes % n_chunks)
                jobs.append((os.path.basename(run.rstrip('/')), ckpt, n, args.max_random, args.device,
                             zlib.crc32(f'{ckpt}:{c}'.encode()), args.epsilon))
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

    # 체크포인트별 행동 분포: 이긴 판·진 판을 나눠 센다
    from sf2_env import ACTIONS
    names = ['+'.join(a) if a else 'NOOP' for a in ACTIONS]
    acts = {}
    for r in rows:
        a = acts.setdefault((r[0], r[1]), {'win': np.zeros(len(names)), 'lose': np.zeros(len(names)), 'rep': 0})
        a['win' if r[3] else 'lose'] += r[8]
        a['rep'] += r[9]
    act_out = os.path.splitext(args.out)[0] + '_actions.csv'
    with open(act_out, 'w') as f:
        f.write('run,step,action,win_count,lose_count\n')
        for (run, step), a in acts.items():
            for i, n in enumerate(names):
                f.write(f'{run},{step},{n},{int(a["win"][i])},{int(a["lose"][i])}\n')
    print('\n행동 분포(모델이 고른 것, 상위 5개)와 직전과 같은 행동을 고른 비율')
    for (run, step), a in sorted(acts.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        tot = a['win'] + a['lose']
        top = np.argsort(-tot)[:5]
        print(f'{run:16s}{str(step):>9s}  same-as-prev {a["rep"] / tot.sum():.2f}  '
              + '  '.join(f'{names[i]} {tot[i] / tot.sum():.0%}' for i in top))

    # 체크포인트별 승률 ± 표준오차
    stats = {}
    for run, step, _, win, reward, *_ in rows:  # noqa
        stats.setdefault((run, step), []).append((win, reward))
    print(f'\n{"run":16s}{"step":>9s}{"win":>14s}{"reward":>9s}')
    for (run, step), v in sorted(stats.items(), key=lambda kv: (kv[0][0], str(kv[0][1]).zfill(9))):
        w = np.mean([x[0] for x in v])
        se = math.sqrt(w * (1 - w) / len(v))
        print(f'{run:16s}{str(step):>9s}{w:9.3f}±{se:.3f}{np.mean([x[1] for x in v]):9.3f}')
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
