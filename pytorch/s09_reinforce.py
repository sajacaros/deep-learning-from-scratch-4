"""ch09/s02_reinforce.py(REINFORCE)의 파이토치 + Gymnasium 버전.

simple_pg.py와의 차이는 update() 하나뿐이다.
simple_pg는 모든 시각의 손실에 '에피소드 전체의 수익 G_0'를 곱하지만,
REINFORCE는 각 시각 t마다 '그 시점 이후의 수익 G_t'를 곱한다.
과거의 보상은 지금의 행동과 무관하므로, 이렇게 하면 분산이 줄어든다.
"""
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
        self.l1 = nn.Linear(state_size, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
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
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        G, loss = 0, 0
        # 뒤에서부터 거슬러 올라가며 G_t를 만들고, 그때그때 손실에 더한다
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -torch.log(prob) * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


if __name__ == '__main__':
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
            action, prob = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        agent.update()

        reward_history.append(total_reward)
        if episode % 100 == 0:
            print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

    env.close()

    # [그림 9-5] 에피소드별 보상 합계 추이
    plot_total_reward(reward_history)
