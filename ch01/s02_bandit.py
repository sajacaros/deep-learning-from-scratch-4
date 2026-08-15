import numpy as np
import matplotlib.pyplot as plt


def place_window(x=60, y=60):
    """그래프 창을 현재 화면 왼쪽 위로 옮기고 맨 앞으로 올린다.

    확장 모니터를 쓰다가 노트북 화면만 남으면, 창 관리자가 예전 좌표를 그대로
    써서 이제는 존재하지 않는 모니터 자리에 창을 띄우는 일이 있다. 그러면
    plt.show()는 정상인데 화면에는 아무것도 안 보인다. 그래서 창을 띄우기 전에
    위치를 직접 잡아준다.
    """
    try:
        window = plt.get_current_fig_manager().window
        if hasattr(window, 'wm_geometry'):  # TkAgg
            window.wm_geometry('+{}+{}'.format(x, y))
            window.lift()
        elif hasattr(window, 'move'):       # QtAgg
            window.move(x, y)
            window.raise_()
    except Exception:
        pass  # 위치 조정은 부가 기능이므로 실패해도 그림 보는 데 지장은 없다


class Bandit:
    def __init__(self, arms=10):  # arms = 슬롯머신 대수
        self.rates = np.random.rand(arms)  # 슬롯머신 각각의 승률 설정(무작위) = 참값 q

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon  # 무작위로 행동할 확률(탐색 확률)
        self.Qs = np.zeros(action_size)  # 참값 q에 대한 추정치 Q
        self.ns = np.zeros(action_size)

    # 슬롯머신의 가치 추정
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    # 행동 선택(ε-탐욕 정책)
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))  # 무작위 행동 선택
        return np.argmax(self.Qs)  # 탐욕 행동 선택


if __name__ == '__main__':
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []  # 보상 합
    rates = []          # 승률

    for step in range(steps):
        action = agent.get_action()   # 행동 선택
        reward = bandit.play(action)  # 실제로 플레이하고 보상을 받음
        agent.update(action, reward)  # 행동과 보상을 통해 학습
        total_reward += reward

        total_rewards.append(total_reward)       # 현재까지의 보상 합 저장
        rates.append(total_reward / (step + 1))  # 현재까지의 승률 저장

    print(total_reward)

    # [그림 1-12] 단계별 보상 총합
    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewards)
    place_window()  # 창이 화면 밖에 뜨지 않도록 위치를 잡아준다
    plt.show()

    # [그림 1-13] 단계별 승률
    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    place_window()
    plt.show()
