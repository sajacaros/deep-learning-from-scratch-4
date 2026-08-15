import numpy as np
import matplotlib.pyplot as plt
from s02_bandit import Agent


class NonStatBandit:
    def __init__(self, arms=10, sigma=0.1):
        self.arms = arms
        # 참값 q(= 진짜 승률)가 한 걸음마다 흔들리는 정도. 환경의 성질이지
        # 에이전트가 고르는 값이 아니다. 0이면 q가 변하지 않는 정상 문제가 된다.
        self.sigma = sigma
        self.rates = np.random.rand(arms)  # 각 슬롯머신의 참값 q

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += self.sigma * np.random.randn(self.arms)  # 노이즈 추가
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)  # 참값 q에 대한 추정치 Q
        self.alpha = alpha  # 고정값 α

    def update(self, action, reward):
        # α로 갱신
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


if __name__ == '__main__':
    runs = 200
    steps = 1000
    epsilon = 0.1
    alpha = 0.8
    agent_types = ['sample average', 'alpha const update']
    results = {}

    for agent_type in agent_types:
        all_rates = np.zeros((runs, steps))  # (200, 1000)

        for run in range(runs):
            if agent_type == 'sample average':
                agent = Agent(epsilon)
            else:
                agent = AlphaAgent(epsilon, alpha)

            bandit = NonStatBandit()
            total_reward = 0
            rates = []

            for step in range(steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step + 1))

            all_rates[run] = rates

        avg_rates = np.average(all_rates, axis=0)
        results[agent_type] = avg_rates

    # [그림 1-20] 표본 평균과 고정값 α에 의한 갱신 비교
    plt.figure()
    plt.ylabel('Average Rates')
    plt.xlabel('Steps')
    for key, avg_rates in results.items():
        plt.plot(avg_rates, label=key)
    plt.legend()
    plt.show()
