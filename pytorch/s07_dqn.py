"""ch08/s03_dqn.py 의 파이토치 + Gymnasium 버전."""
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import deque
import random
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]), dtype=torch.float32)
        action = torch.tensor(np.array([x[1] for x in data]), dtype=torch.int64)
        reward = torch.tensor(np.array([x[2] for x in data]), dtype=torch.float32)
        next_state = torch.tensor(np.stack([x[3] for x in data]), dtype=torch.float32)
        done = torch.tensor(np.array([x[4] for x in data]), dtype=torch.float32)
        return state, action, reward, next_state, done


class QNet(nn.Module):  # 신경망 클래스
    def __init__(self, state_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:  # 에이전트 클래스
    def __init__(self, state_size=4, action_size=2):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000  # 경험 재생 버퍼 크기
        self.batch_size = 32      # 미니배치 크기
        self.state_size = state_size
        self.action_size = action_size

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(state_size, action_size)         # 원본 신경망
        self.qnet_target = QNet(state_size, action_size)  # 목표 신경망
        self.sync_qnet()  # 두 신경망을 같은 값에서 출발시킨다
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)  # 배치 처리용 차원 추가
            with torch.no_grad():
                qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        # 경험 재생 버퍼에 경험 데이터 추가
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return  # 데이터가 미니배치 크기만큼 쌓이지 않았다면 여기서 끝

        # 미니배치 크기 이상이 쌓이면 미니배치 생성
        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        # 실제로 선택한 행동의 Q 값만 골라낸다
        qs = self.qnet(state)
        q = qs.gather(1, action.unsqueeze(1)).squeeze(1)

        # 목표 신경망으로 다음 상태의 최대 Q 값을 구한다.
        # no_grad로 감싸 목표값이 역전파 대상에서 제외되도록 한다.
        with torch.no_grad():
            next_q = self.qnet_target(next_state).max(dim=1).values
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):  # 두 신경망 동기화
        self.qnet_target.load_state_dict(self.qnet.state_dict())


if __name__ == '__main__':
    episodes = 300      # 에피소드 수
    sync_interval = 20  # 신경망 동기화 주기(20번째 에피소드마다 동기화)
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_size=env.observation_space.shape[0],
                     action_size=env.action_space.n)
    reward_history = []  # 에피소드별 보상 기록

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 부트스트랩에는 terminated만 쓴다.
            # 제한 시간에 걸려 잘린 것(truncated)은 '가치가 0인 종료'가 아니라
            # 에피소드를 여기서 끊었을 뿐이므로 다음 상태의 가치를 그대로 살려야 한다.
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()

        reward_history.append(total_reward)
        if episode % 10 == 0:
            print("episode :{}, total reward : {}".format(episode, total_reward))

    env.close()


    # [그림 8-8] 「카트 폴」에서 에피소드별 보상 총합의 추이
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()


    # 학습이 끝난 에이전트에 탐욕 행동을 선택하도록 하여 플레이
    # 창을 띄우려면 render_mode='human'으로 환경을 다시 만든다.
    agent.epsilon = 0  # 탐욕 정책(무작위로 행동할 확률 ε을 0으로 설정)
    env = gym.make('CartPole-v1', render_mode='human')
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    env.close()
    print('Total Reward:', total_reward)
