"""저장한 체크포인트마다 여러 판 싸워 승률을 잰다. DQN(qnet_*.pt)은 기본 ε=0, PPO(ppo_*.zip)는 정책에서 행동을 뽑는다.

  uv run python sf2/evaluate.py runs/dqn runs/double runs/dueling runs/double_dueling --episodes 200
  uv run python sf2/evaluate.py runs/dueling/qnet_final.pt ...   # 파일을 직접 줘도 된다
  uv run python sf2/evaluate.py runs/ppo --episodes 200          # PPO도 같은 방식으로(--deterministic이면 확률이 가장 큰 행동)
  uv run python sf2/evaluate.py runs/ppo/ppo_final.zip --blind runs/eval_actions.csv   # 화면을 안 보고 모델의 행동 비율대로만 뽑기

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
    from dqn import load_qnet, mask_specials, random_action
    from sf2_env import STATE, action_mask, make_env

    run, ckpt, episodes, max_random, device, seed, epsilon, deterministic, *rest = job
    blind = rest[:1]
    state = rest[1] if len(rest) > 1 else STATE
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # 모델이 본 상태와 같은 환경을 띄운다(쿨타임 채널이 3개면 필살기별 쿨타임). 보너스·벌점은 평가에 쓰지 않는다
    if ckpt.endswith('.zip'):  # PPO(stable-baselines3). 학습률·클리핑 스케줄은 평가에 필요 없어 상수로 바꿔 읽는다
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(ckpt, device=device, custom_objects={
            'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0})
        separate = min(model.observation_space.shape) > 4  # sb3는 (C, H, W)로 바꿔 저장한다
    else:
        qnet = load_qnet(ckpt, device)
        separate = qnet.conv[0].in_channels > 4
    env = make_env(separate=separate, state=state)
    if blind and blind[0] is not None:  # 화면을 보지 않고 정해 둔 행동 비율대로 뽑는다(쿨타임 중이면 필살기를 빼고 다시 맞춤)
        def policy(state, probs=blind[0]):
            p = probs * action_mask(state)
            return int(rng.choice(len(p), p=p / p.sum()))
    elif ckpt.endswith('.zip'):
        policy = lambda state: int(model.predict(state, deterministic=deterministic, action_masks=action_mask(state))[0])
    else:
        def policy(state):
            with torch.no_grad():
                x = torch.from_numpy(state).unsqueeze(0).to(device)
                return mask_specials(qnet(x), x).argmax().item()

    m = re.search(r'(?:qnet_|ppo_)(\d+)(?:_steps)?\.(?:pt|zip)$', ckpt)
    step = int(m.group(1)) if m else re.sub(r'^(qnet|ppo)_', '', os.path.splitext(os.path.basename(ckpt))[0])
    rows = []
    for episode in range(episodes):
        state, info = env.reset()
        done, total_reward, length = False, 0.0, 0
        for _ in range(rng.integers(0, max_random + 1)):  # random start
            state, reward, terminated, truncated, info = env.step(random_action(state, rng))
            total_reward += reward
            length += 1
            done = terminated or truncated
            if done:
                break
        counts = np.zeros(env.action_space.n, dtype=np.int64)
        repeats, prev = 0, None
        while not done:
            if rng.random() < epsilon:
                action = random_action(state, rng)
            else:
                action = policy(state)
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
    p.add_argument('--deterministic', action='store_true')  # PPO만: 뽑지 않고 확률이 가장 큰 행동
    p.add_argument('--workers', type=int, default=40)
    p.add_argument('--device', default='cpu')
    p.add_argument('--blind', help='<out>_actions.csv. 모델 대신 이 파일에 적힌 그 모델의 행동 비율대로만 뽑는다')
    p.add_argument('--state', nargs='+', default=[None])  # 시작 저장 상태(여럿이면 상태마다 따로 평가, run 이름 뒤에 @상태)
    p.add_argument('--out', default='runs/eval.csv')
    args = p.parse_args()

    freq = {}
    if args.blind:
        from sf2_env import ACTION_NAMES
        for line in open(args.blind).read().splitlines()[1:]:
            run, step, action, win, lose = line.split(',')
            freq.setdefault((run, step), np.zeros(len(ACTION_NAMES)))[ACTION_NAMES.index(action)] = int(win) + int(lose)

    from sf2_env import STATE
    jobs = []
    for run in args.runs:
        if run.endswith(('.pt', '.zip')):
            ckpts, run = [run], os.path.dirname(run)
        else:
            ckpts = glob.glob(os.path.join(run, 'qnet_[0-9]*.pt')) + glob.glob(os.path.join(run, 'ppo_*_steps.zip'))
        for ckpt in ckpts:
            probs, name = None, os.path.basename(run.rstrip('/'))
            if args.blind:
                m = re.search(r'(\d+)(?:_steps)?\.(?:pt|zip)$', ckpt)
                step = m.group(1) if m else re.sub(r'^(qnet|ppo)_', '', os.path.splitext(os.path.basename(ckpt))[0])
                probs, name = freq[(name, step)] / freq[(name, step)].sum(), name + '_blind'
            for state in args.state:
                label = name if state is None else f"{name}@{state.removeprefix('Champion.')}"
                # 한 체크포인트를 여러 조각으로 나눠 프로세스를 고르게 쓴다
                n_chunks = max(1, args.episodes // 10)
                for c in range(n_chunks):
                    n = args.episodes // n_chunks + (c < args.episodes % n_chunks)
                    jobs.append((label, ckpt, n, args.max_random, args.device, zlib.crc32(f'{ckpt}:{c}'.encode()),
                                 args.epsilon, args.deterministic, probs, state or STATE))
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
    from sf2_env import ACTION_NAMES as names
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
