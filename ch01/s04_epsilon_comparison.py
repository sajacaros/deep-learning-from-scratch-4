if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
from s02_bandit import Bandit, Agent


# =============================================================================
# 1.4절 보충 - ε를 바꿔가며 비교하기 (원서에 없는 추가분)
#
# s03_bandit_avg.py는 ε = 0.1 하나만 쓴다. 여기서는 ε를 바꿔가며
# '탐색(exploration)과 활용(exploitation)의 균형'을 눈으로 확인한다.
#
#   ε = 0     : 한 번 좋아 보인 슬롯머신만 계속 당긴다(활용만)
#               → 진짜 최고를 못 찾고 엉뚱한 곳에 머무를 수 있다
#   ε = 1     : 항상 무작위(탐색만) → 배운 것을 쓰지 않으니 평균 승률에 머문다
#   그 사이 값 : 처음에는 탐색이 손해처럼 보이지만 길게 보면 이득이다
# =============================================================================

if __name__ == '__main__':
    runs = 200    # 실험 횟수(평균을 내기 위해)
    steps = 1000  # 한 실험당 플레이 횟수
    epsilons = [0.0, 0.01, 0.1, 0.3, 1.0]

    results = {}

    for epsilon in epsilons:
        all_rates = np.zeros((runs, steps))

        for run in range(runs):
            bandit = Bandit()      # 승률이 변하지 않는 정상 문제
            agent = Agent(epsilon)
            total_reward = 0
            rates = []

            for step in range(steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step + 1))

            all_rates[run] = rates

        results[epsilon] = np.average(all_rates, axis=0)

    print('정상 문제에서 {}회 실험 평균 (steps={})'.format(runs, steps))
    print('-' * 44)
    print('{:<20}{:>12}{:>12}'.format('epsilon', 'step 100', 'step 1000'))
    print('-' * 44)
    for epsilon in epsilons:
        rates = results[epsilon]
        print('{:<20}{:>12.4f}{:>12.4f}'.format(epsilon, rates[99], rates[-1]))

    early = max(epsilons, key=lambda e: results[e][99])
    late = max(epsilons, key=lambda e: results[e][-1])
    print('-' * 44)
    print('100 걸음 시점 1위: epsilon =', early)
    print('1000 걸음 시점 1위: epsilon =', late)
    print('\nε=0은 처음 좋아 보인 슬롯머신에 그대로 갇혀 승률이 거의 오르지 않는다.')
    print('ε=1은 배운 것을 전혀 쓰지 않으므로 역시 전체 평균 근처에 머문다.')
    print('탐색을 너무 많이 해도(0.3) 손해다. 좋은 걸 알고도 30%는 딴 데를 당기기 때문이다.')

    # -------------------------------------------------------------------------
    # 그런데 ε는 작을수록 늦게, 그러나 더 높이 올라간다.
    # 1000 걸음에서는 ε=0.01이 한참 뒤처져 보이지만 정말 그럴까?
    # 걸음 수를 20배로 늘려 직접 확인해본다.
    # -------------------------------------------------------------------------
    long_steps = 20000
    long_runs = 100
    long_results = {}

    print('\n걸음 수를 {}로 늘려 ε=0.01과 ε=0.1을 다시 비교한다...'.format(long_steps))
    for epsilon in [0.01, 0.1]:
        all_rates = np.zeros((long_runs, long_steps))
        for run in range(long_runs):
            bandit = Bandit()
            agent = Agent(epsilon)
            total_reward = 0
            rates = []
            for step in range(long_steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step + 1))
            all_rates[run] = rates
        long_results[epsilon] = np.average(all_rates, axis=0)

    print('-' * 56)
    print('{:<12}{:>11}{:>11}{:>11}{:>11}'.format(
        'epsilon', 'step 1k', 'step 5k', 'step 10k', 'step 20k'))
    print('-' * 56)
    for epsilon, rates in long_results.items():
        print('{:<12}{:>11.4f}{:>11.4f}{:>11.4f}{:>11.4f}'.format(
            epsilon, rates[999], rates[4999], rates[9999], rates[-1]))
    print('-' * 56)

    # 역전이 일어나는 지점: ε=0.01이 앞선 뒤로 다시 뒤집히지 않는 첫 걸음
    diff = long_results[0.01] - long_results[0.1]
    behind = np.flatnonzero(diff <= 0)
    crossover = int(behind[-1]) + 1 if len(behind) and behind[-1] + 1 < len(diff) else None
    if crossover is not None:
        print('ε=0.01이 ε=0.1을 추월해 계속 앞서는 시점: 약 {} 걸음'.format(crossover))
    else:
        print('이번 실행에서는 {} 걸음 안에 역전이 확정되지 않았다.'.format(long_steps))
    print('\n탐색에 드는 비용은 매 걸음 일정한데(ε의 확률로 손해),')
    print('탐색으로 찾아낸 이득은 남은 걸음 내내 쌓인다.')
    print('그래서 오래 할수록 작은 ε가 유리해진다. "좋은 ε"는 문제가 아니라')
    print('얼마나 오래 플레이하느냐에 달려 있다.')

    plt.figure(figsize=(11, 4))

    plt.subplot(1, 2, 1)
    for epsilon in epsilons:
        plt.plot(results[epsilon], label='epsilon = {}'.format(epsilon))
    plt.xlabel('Steps')
    plt.ylabel('Average Rates')
    plt.title('Effect of epsilon ({} steps)'.format(steps))
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for epsilon, rates in long_results.items():
        plt.plot(rates, label='epsilon = {}'.format(epsilon))
    if crossover:
        plt.axvline(crossover, color='gray', linestyle='--', linewidth=1,
                    label='crossover ~{}'.format(crossover))
    plt.xlabel('Steps')
    plt.ylabel('Average Rates')
    plt.title('Small epsilon wins in the long run')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
