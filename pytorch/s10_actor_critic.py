"""ch09/actor_critic.py(행위자-비평자)의 파이토치 + Gymnasium 버전."""
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


class PolicyNet(nn.Module):  # 행위자(actor)
    def __init__(self, state_size, action_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):  # 비평자(critic)
    def __init__(self, state_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self, state_size=4, action_size=2):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = action_size

        self.pi = PolicyNet(state_size, action_size)
        self.v = ValueNet(state_size)

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)

        # ① 비평자(V) 갱신: TD 목표를 향해 V(s)를 맞춘다.
        #    목표값은 학습 대상이 아니므로 no_grad로 역전파에서 끊는다.
        with torch.no_grad():
            target = reward + self.gamma * self.v(next_state) * (1 - done)
        v = self.v(state)
        loss_v = F.mse_loss(v, target)

        # ② 행위자(pi) 갱신: TD 오차 delta를 REINFORCE의 G 대신 사용한다.
        #    delta는 '가중치'일 뿐이므로 상수로 취급한다(.item()).
        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()

        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()


episodes = 2000
env = gym.make('CartPole-v1')
agent = Agent(state_size=env.observation_space.shape[0],
              action_size=env.action_space.n)
reward_history = []

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 부트스트랩에는 terminated만 쓴다.
        # 제한 시간에 걸려 잘린 경우(truncated)는 다음 상태의 가치를 살려야 한다.
        agent.update(state, prob, reward, next_state, terminated)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

env.close()

# [그림 9-9] 에피소드별 보상 합계 추이
plot_total_reward(reward_history)
