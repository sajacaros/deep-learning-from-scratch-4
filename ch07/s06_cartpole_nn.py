import sys
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(2)    # 행동의 개수(왼쪽, 오른쪽)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.action_size = 2

        self.qnet = QNet()
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state[np.newaxis, :])  # 원핫 대신 실수 4개를 그대로 넣음
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = np.zeros(1)
        else:
            next_qs = self.qnet(next_state[np.newaxis, :])
            next_q = next_qs.max(axis=1)
            next_q.unchain()

        target = self.gamma * next_q + reward
        qs = self.qnet(state[np.newaxis, :])
        q = qs[:, action]
        loss = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()


def play(agent, render_env):
    # 탐욕 정책으로 한 에피소드를 화면에 보여줌(학습하지 않음)
    state = render_env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = agent.qnet(state[np.newaxis, :]).data.argmax()
        state, reward, terminated, truncated, info = render_env.step(action)
        done = terminated | truncated
        total_reward += reward
    return total_reward


if __name__ == '__main__':
    np.random.seed(0)
    episodes = 300
    watch_interval = 50  # --watch: 이 주기마다 학습 중인 에이전트의 플레이를 화면에 보여줌
    watch = '--watch' in sys.argv
    live = '--live' in sys.argv  # --live: 학습 중인 카트 폴과 보상 그래프를 한 창에 실시간으로 보여줌
    env = gym.make('CartPole-v1', render_mode='rgb_array' if live else None)
    render_env = gym.make('CartPole-v1', render_mode='human') if watch else None
    agent = QLearningAgent()
    reward_history = []

    if live:
        plt.ion()
        fig, (ax_env, ax_reward) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax_env.axis('off')
        img = ax_env.imshow(np.zeros((400, 600, 3), dtype=np.uint8))
        ax_reward.set_xlim(0, episodes)
        ax_reward.set_ylim(0, 200)
        ax_reward.set_xlabel('Episode')
        ax_reward.set_ylabel('Total Reward')
        line, = ax_reward.plot([], [])
        best_text = ax_env.text(0.5, -0.05, '', transform=ax_env.transAxes, ha='center', va='top')
        best_reward, best_episode = 0, 0

    for episode in range(episodes):
        if watch and episode % watch_interval == 0:
            print('[watch] episode {} → {}점'.format(episode, play(agent, render_env)))

        state = env.reset(seed=episode)[0]
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if live:
                img.set_data(env.render())
                ax_env.set_title('episode {}  step {}'.format(episode, int(total_reward)))
                plt.pause(0.001)

        reward_history.append(total_reward)
        if episode % 10 == 0:
            print('episode :{}, total reward : {}'.format(episode, total_reward))

        if live:
            line.set_data(range(len(reward_history)), reward_history)
            if total_reward > best_reward:
                best_reward, best_episode = total_reward, episode
            best_text.set_text('best: {} steps (episode {})'.format(int(best_reward), best_episode))
            if total_reward > ax_reward.get_ylim()[1]:
                ax_reward.set_ylim(0, total_reward * 1.1)

    if watch:
        print('[watch] 학습 끝 → {}점'.format(play(agent, render_env)))
        render_env.close()

    if live:
        plt.ioff()
        plt.show()
        sys.exit()

    # 에피소드별 보상 총합의 추이
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    if '--no-show' in sys.argv:
        plt.savefig('images/s06_cartpole_nn.png')
    else:
        plt.show()
