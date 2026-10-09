"""8장 DQN으로 스트리트 파이터 2 학습하기.

pytorch/s07_dqn.py와 구조는 같고, 화면 입력에 맞춰 다음을 바꿨다.
  - 신경망: 완전연결층 → CNN(DQN 논문의 구조)
  - 경험 재생 버퍼: deque → 미리 잡아둔 uint8 배열(화면이라 메모리가 크다)
  - 목표 신경망 동기화·학습 주기: 에피소드 단위 → 걸음 단위
  - ε: 고정 0.1 → 1.0에서 0.05로 줄여감
  - 손실: MSE → Huber(큰 오차에 덜 민감)
8.4절의 확장은 옵션으로 켠다.
  --double   8.4.1 Double DQN: 다음 행동은 원본 신경망이 고르고, 그 값은 목표 신경망으로 평가
  --dueling  8.4.3 Dueling DQN: Q = V + A - mean(A)
학습 안정화와 모델 고르기도 옵션이다.
  --lr-end      학습률을 --lr에서 이 값까지 직선으로 줄인다(후반에 정책이 크게 흔들리지 않게)
  --adam-eps    Adam의 eps. Rainbow의 1.5e-4는 보상을 ±1로 자른 경우의 값이다.
                보상에 0.001을 곱하는 이 환경은 기울기가 1e-6 수준이라 1.5e-4면 학습 폭이 1/100로 줄어 학습이 거의 안 된다
  --eval-interval  이 걸음마다 따로 띄운 프로세스에서 평가하고, 가장 좋은 모델을 qnet_best.pt로 남긴다

  uv run python sf2/dqn.py --steps 1000000 --device cuda --out runs/dqn
  uv run python sf2/dqn.py --double --dueling --out runs/double_dueling
  uv run python sf2/dqn.py --lr-end 1e-5 --eval-interval 50000 --out runs/dqn_lrdecay
"""
import argparse
import multiprocessing as mp
import os
import shutil
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sf2_env import make_env


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_shape):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state = np.zeros((buffer_size, *state_shape), dtype=np.uint8)
        self.next_state = np.zeros((buffer_size, *state_shape), dtype=np.uint8)
        self.action = np.zeros(buffer_size, dtype=np.int64)
        self.reward = np.zeros(buffer_size, dtype=np.float32)
        self.done = np.zeros(buffer_size, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        i = self.pos
        self.state[i], self.action[i], self.reward[i] = state, action, reward
        self.next_state[i], self.done[i] = next_state, done
        self.pos = (self.pos + 1) % self.buffer_size  # 가득 차면 가장 오래된 것부터 덮어씀
        self.size = min(self.size + 1, self.buffer_size)

    def __len__(self):
        return self.size

    def get_batch(self, device):
        idx = np.random.randint(0, self.size, self.batch_size)
        to = lambda x: torch.from_numpy(x[idx]).to(device)
        return to(self.state), to(self.action), to(self.reward), to(self.next_state), to(self.done)


class QNet(nn.Module):  # 신경망 클래스
    def __init__(self, action_size, in_shape=(100, 128, 3), dueling=False):
        super().__init__()
        h, w, c = in_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, c, h, w)).shape[1]  # 100×128 입력 → 6912
        self.dueling = dueling
        if dueling:  # 출력 직전에 두 갈래로 나눈다
            self.v = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, 1))            # V(s)
            self.a = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, action_size))  # A(s, a)
        else:
            self.fc = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, action_size))

    def forward(self, x):
        # (N, H, W, C) uint8 → (N, C, H, W) 0~1 실수
        x = x.permute(0, 3, 1, 2).float() / 255.0
        h = self.conv(x)
        if self.dueling:
            a = self.a(h)
            # A의 평균을 빼서 V와 A가 하나로 정해지게 한다(빼지 않으면 V에 상수를 더하고 A에서 빼도 Q가 같음)
            return self.v(h) + a - a.mean(dim=1, keepdim=True)
        return self.fc(h)


class DQNAgent:  # 에이전트 클래스
    def __init__(self, action_size, device, buffer_size=100_000, double=False, dueling=False,
                 lr=1e-4, adam_eps=1e-8):
        self.gamma = 0.94          # 참고 프로젝트(PPO)와 같은 값
        self.lr = lr
        self.epsilon = 1.0
        self.batch_size = 32
        self.action_size = action_size
        self.device = device
        self.double = double

        self.replay_buffer = ReplayBuffer(buffer_size, self.batch_size, (100, 128, 3))
        self.qnet = QNet(action_size, dueling=dueling).to(device)         # 원본 신경망
        self.qnet_target = QNet(action_size, dueling=dueling).to(device)  # 목표 신경망
        self.sync_qnet()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr, eps=adam_eps)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)  # 배치 처리용 차원 추가
        with torch.no_grad():
            qs = self.qnet(state)
        return qs.argmax().item()

    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.get_batch(self.device)

        qs = self.qnet(state)
        q = qs.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_qs = self.qnet_target(next_state)
            if self.double:  # 고르는 신경망과 평가하는 신경망을 나눠 과대평가를 줄인다
                next_action = self.qnet(next_state).argmax(dim=1, keepdim=True)
                next_q = next_qs.gather(1, next_action).squeeze(1)
            else:
                next_q = next_qs.max(dim=1).values
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def sync_qnet(self):  # 두 신경망 동기화
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group['lr'] = lr


class Evaluator:
    """저장한 체크포인트를 다른 프로세스에서 평가하고(evaluate.py), 승률이 가장 높은 것을 qnet_best.pt로 복사한다.
    stable-retro는 한 프로세스에 에뮬레이터를 하나만 띄울 수 있어 학습 프로세스 안에서는 평가할 수 없다."""
    def __init__(self, out, episodes, epsilon, workers=10):
        self.out, self.episodes, self.epsilon, self.workers = out, episodes, epsilon, workers
        self.pool = mp.get_context('spawn').Pool(workers)
        self.lock = threading.Lock()
        self.pending = {}  # 걸음 → 아직 안 끝난 조각 수, 지금까지의 승패
        self.best = (-1.0, None)
        self.log = open(os.path.join(out, 'eval.csv'), 'w')
        self.log.write('step,episodes,win_rate,reward\n')

    def submit(self, step, ckpt):
        from evaluate import evaluate
        n = self.episodes // self.workers
        with self.lock:
            self.pending[step] = [self.workers, []]
        for c in range(self.workers):
            job = ('', ckpt, n, 30, 'cpu', step * 100 + c, self.epsilon)
            self.pool.apply_async(evaluate, (job,), callback=lambda rows, s=step, k=ckpt: self._done(s, k, rows))

    def _done(self, step, ckpt, rows):
        with self.lock:
            left, acc = self.pending[step]
            acc += rows
            self.pending[step][0] = left - 1
            if left > 1:
                return
            del self.pending[step]
            win = np.mean([r[3] for r in acc])
            reward = np.mean([r[4] for r in acc])
            self.log.write(f'{step},{len(acc)},{win:.4f},{reward:.4f}\n')
            self.log.flush()
            best = ''
            if win > self.best[0]:
                self.best = (win, step)
                shutil.copy(ckpt, os.path.join(self.out, 'qnet_best.pt'))
                best = ' ← best'
            print(f'[eval] step {step:8d} | win {win:.3f} | reward {reward:.3f}{best}', flush=True)

    def close(self):
        self.pool.close()
        self.pool.join()
        self.log.close()
        print(f'[eval] best: step {self.best[1]}, win {self.best[0]:.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1_000_000)       # 총 걸음 수(1걸음 = 6프레임)
    p.add_argument('--buffer', type=int, default=100_000)        # 경험 재생 버퍼 크기
    p.add_argument('--learning-starts', type=int, default=10_000)
    p.add_argument('--train-freq', type=int, default=4)          # 몇 걸음마다 한 번 학습할지
    p.add_argument('--sync-interval', type=int, default=10_000)  # 목표 신경망 동기화 주기(걸음)
    p.add_argument('--eps-decay', type=float, default=0.3)       # 전체의 몇 %에 걸쳐 ε을 줄일지
    p.add_argument('--save-interval', type=int, default=100_000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-end', type=float, default=None)          # 주면 --lr에서 이 값까지 직선으로 줄임
    p.add_argument('--adam-eps', type=float, default=1e-8)
    p.add_argument('--eval-interval', type=int, default=0)        # 0이면 학습 중 평가 안 함
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--eval-epsilon', type=float, default=0.05)    # Nature DQN의 평가 방식
    p.add_argument('--double', action='store_true')
    p.add_argument('--dueling', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default='runs/dqn')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = make_env()
    agent = DQNAgent(env.action_space.n, args.device, args.buffer, args.double, args.dueling,
                     args.lr, args.adam_eps)
    evaluator = Evaluator(args.out, args.eval_episodes, args.eval_epsilon) if args.eval_interval else None
    eps_end, eps_steps = 0.05, int(args.steps * args.eps_decay)
    print(f'actions={env.action_space.n}, {vars(args)}')

    log = open(os.path.join(args.out, 'episodes.csv'), 'w')
    log.write('episode,step,reward,win,length,epsilon\n')

    state, info = env.reset()
    episode, ep_reward, ep_len = 0, 0.0, 0
    recent = []  # 최근 에피소드의 승패
    t0 = time.time()

    for step in range(1, args.steps + 1):
        agent.epsilon = max(eps_end, 1.0 - (1.0 - eps_end) * step / eps_steps)
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, terminated)
        state = next_state
        ep_reward += reward
        ep_len += 1

        if step > args.learning_starts and step % args.train_freq == 0:
            if args.lr_end is not None:
                agent.set_lr(args.lr + (args.lr_end - args.lr) * step / args.steps)
            agent.update()
        if step % args.sync_interval == 0:
            agent.sync_qnet()
        if step % args.save_interval == 0 or (evaluator and step % args.eval_interval == 0):
            ckpt = os.path.join(args.out, f'qnet_{step}.pt')
            torch.save(agent.qnet.state_dict(), ckpt)
            if evaluator and step % args.eval_interval == 0:
                evaluator.submit(step, ckpt)

        if terminated or truncated:
            win = int(info['enemy_hp'] < 0)
            recent = (recent + [win])[-100:]
            log.write(f'{episode},{step},{ep_reward:.4f},{win},{ep_len},{agent.epsilon:.3f}\n')
            log.flush()
            if episode % 10 == 0:
                sps = step / (time.time() - t0)
                print(f'episode {episode:5d} | step {step:8d} | reward {ep_reward:7.3f} | '
                      f'win rate(100) {np.mean(recent):.2f} | eps {agent.epsilon:.3f} | {sps:.0f} steps/s')
            episode += 1
            ep_reward, ep_len = 0.0, 0
            state, info = env.reset()

    torch.save(agent.qnet.state_dict(), os.path.join(args.out, 'qnet_final.pt'))
    log.close()
    env.close()
    if evaluator:
        evaluator.close()  # 남은 평가가 끝날 때까지 기다린다


if __name__ == '__main__':
    main()
