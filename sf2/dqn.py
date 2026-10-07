"""8장 DQN으로 스트리트 파이터 2 학습하기.

pytorch/s07_dqn.py와 구조는 같고, 화면 입력에 맞춰 다음을 바꿨다.
  - 신경망: 완전연결층 → CNN(DQN 논문의 구조)
  - 경험 재생 버퍼: deque → 미리 잡아둔 uint8 배열(화면이라 메모리가 크다)
  - 목표 신경망 동기화·학습 주기: 에피소드 단위 → 걸음 단위
  - ε: 고정 0.1 → 1.0에서 0.05로 줄여감
  - 손실: MSE → Huber(큰 오차에 덜 민감)

  uv run python sf2/dqn.py --steps 1000000 --device cuda --out runs/dqn
"""
import argparse
import os
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
    def __init__(self, action_size, in_shape=(100, 128, 3)):
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
        self.fc = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, action_size))

    def forward(self, x):
        # (N, H, W, C) uint8 → (N, C, H, W) 0~1 실수
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.fc(self.conv(x))


class DQNAgent:  # 에이전트 클래스
    def __init__(self, action_size, device, buffer_size=100_000):
        self.gamma = 0.94          # 참고 프로젝트(PPO)와 같은 값
        self.lr = 1e-4
        self.epsilon = 1.0
        self.batch_size = 32
        self.action_size = action_size
        self.device = device

        self.replay_buffer = ReplayBuffer(buffer_size, self.batch_size, (100, 128, 3))
        self.qnet = QNet(action_size).to(device)         # 원본 신경망
        self.qnet_target = QNet(action_size).to(device)  # 목표 신경망
        self.sync_qnet()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

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
            next_q = self.qnet_target(next_state).max(dim=1).values
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def sync_qnet(self):  # 두 신경망 동기화
        self.qnet_target.load_state_dict(self.qnet.state_dict())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1_000_000)       # 총 걸음 수(1걸음 = 6프레임)
    p.add_argument('--buffer', type=int, default=100_000)        # 경험 재생 버퍼 크기
    p.add_argument('--learning-starts', type=int, default=10_000)
    p.add_argument('--train-freq', type=int, default=4)          # 몇 걸음마다 한 번 학습할지
    p.add_argument('--sync-interval', type=int, default=10_000)  # 목표 신경망 동기화 주기(걸음)
    p.add_argument('--eps-decay', type=float, default=0.3)       # 전체의 몇 %에 걸쳐 ε을 줄일지
    p.add_argument('--save-interval', type=int, default=100_000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default='runs/dqn')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = make_env()
    agent = DQNAgent(env.action_space.n, args.device, args.buffer)
    eps_end, eps_steps = 0.05, int(args.steps * args.eps_decay)
    print(f'device={args.device}, actions={env.action_space.n}, steps={args.steps}')

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
            agent.update()
        if step % args.sync_interval == 0:
            agent.sync_qnet()
        if step % args.save_interval == 0:
            torch.save(agent.qnet.state_dict(), os.path.join(args.out, f'qnet_{step}.pt'))

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


if __name__ == '__main__':
    main()
