"""ch09/simple_pg.py(가장 간단한 정책 경사법)의 파이토치 + Gymnasium 버전."""
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from common.utils import plot_total_reward


class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)  # 첫 번째 계층
        self.l2 = nn.Linear(128, action_size)  # 두 번째 계층

    def forward(self, x):
        x = F.relu(self.l1(x))              # 첫 번째 계층에서는 ReLU 함수 사용
        x = F.softmax(self.l2(x), dim=1)    # 두 번째 계층에서는 소프트맥스 함수 사용
        return x


class Agent:
    def __init__(self, state_size=4, action_size=2):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = action_size

        self.memory = []
        self.pi = Policy(state_size, action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)  # 배치 처리용 축 추가
        probs = self.pi(state)  # 순전파 수행
        probs = probs[0]
        m = Categorical(probs)            # 확률분포에서
        action = m.sample().item()        # 행동을 샘플링
        return action, probs[action]      # 선택된 행동과 그 확률 반환

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):  # 수익 G 계산
            G = reward + self.gamma * G

        for reward, prob in self.memory:  # 손실 함수 계산
            loss += -torch.log(prob) * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []  # 메모리 초기화


episodes = 3000
env = gym.make('CartPole-v1')
agent = Agent(state_size=env.observation_space.shape[0],
              action_size=env.action_space.n)
reward_history = []

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)  # 행동 선택
        next_state, reward, terminated, truncated, info = env.step(action)  # 행동 수행
        done = terminated or truncated

        agent.add(reward, prob)  # 보상과 행동의 확률을 에이전트에 추가
        state = next_state       # 상태 전이
        total_reward += reward   # 보상 총합 계산

    agent.update()  # 정책 갱신

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

env.close()

# [그림 9-2] 에피소드별 보상 합계 추이
plot_total_reward(reward_history)
