"""ch08/replay_buffer.py 의 파이토치 + Gymnasium 버전.

경험 재생(experience replay) 버퍼는 넘파이로 모아두었다가
학습에 넣기 직전에 텐서로 바꾼다. 이때 자료형을 명시하는 것이 중요하다.
  - 상태/보상 : float32 (신경망 입력과 손실 계산에 쓰인다)
  - 행동      : int64   (gather의 인덱스로 쓰이므로 정수여야 한다)
"""
from collections import deque
import random
import gymnasium as gym
import numpy as np
import torch


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


env = gym.make('CartPole-v1')
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):  # 에피소드 10회 수행
    state, info = env.reset()
    done = False

    while not done:
        action = 0  # 항상 0번째 행동만 수행
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.add(state, action, reward, next_state, done)  # 버퍼에 추가
        state = next_state

# 경험 데이터 버퍼로부터 미니배치 생성
state, action, reward, next_state, done = replay_buffer.get_batch()
print(state.shape, state.dtype)            # torch.Size([32, 4]) torch.float32
print(action.shape, action.dtype)          # torch.Size([32]) torch.int64
print(reward.shape, reward.dtype)          # torch.Size([32]) torch.float32
print(next_state.shape, next_state.dtype)  # torch.Size([32, 4]) torch.float32
print(done.shape, done.dtype)              # torch.Size([32]) torch.float32
