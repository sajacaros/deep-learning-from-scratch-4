"""ch07/q_learning_nn.py(신경망을 이용한 Q 러닝)의 파이토치 버전."""
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.gridworld import GridWorld


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return torch.from_numpy(vec).unsqueeze(0)  # 배치 처리용 차원 추가


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 100)  # 입력은 원핫 벡터(3*4), 중간층의 크기는 100
        self.l2 = nn.Linear(100, 4)   # 행동의 크기(가능한 행동의 개수)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()                                        # 신경망 초기화
        self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():  # 행동 선택에는 역전파가 필요 없다
                qs = self.qnet(state_vec)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        # 다음 상태에서 최대가 되는 Q 함수의 값(next_q) 계산
        if done:  # 목표 상태에 도달(목표 상태에서의 Q 함수는 항상 0)
            next_q = torch.zeros(1)
        else:
            with torch.no_grad():  # next_q를 역전파 대상에서 제외
                next_q = self.qnet(next_state).max(dim=1).values

        # 목표 (reward는 float으로 변환해 float32 텐서를 유지한다)
        target = self.gamma * next_q + float(reward)
        # 현재 상태에서의 Q 함수 값(q) 계산
        qs = self.qnet(state)
        q = qs[:, action]
        # 목표(target)와 q의 오차 계산
        loss = F.mse_loss(q, target)

        # 역전파 → 매개변수 갱신
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


env = GridWorld()
agent = QLearningAgent()

episodes = 1000  # 에피소드 수
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state

    average_loss = total_loss / cnt
    loss_history.append(average_loss)


# [그림 7-14] 에피소드별 손실 추이
plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# [그림 7-15] 신경망을 이용한 Q 러닝으로 얻은 Q 함수와 정책
Q = {}
for state in env.states():
    with torch.no_grad():
        qs = agent.qnet(one_hot(state))
    for action in env.action_space:
        Q[state, action] = float(qs[0, action])
env.render_q(Q)
