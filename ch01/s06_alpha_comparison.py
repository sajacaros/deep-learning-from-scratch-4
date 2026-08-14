if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
from s02_bandit import Agent
from s05_non_stationary import NonStatBandit, AlphaAgent


# =============================================================================
# 1.5절 보충 - 고정값 α를 바꿔가며 비교하기 (원서에 없는 추가분)
#
# s05_non_stationary.py는 α = 0.8 하나만 써서 '표본 평균 vs 고정값 α'를 비교한다.
# 여기서는 α 자체를 여러 값으로 바꿔가며 무엇이 달라지는지 본다.
#
#   Q_n = Q_{n-1} + α (R_n - Q_{n-1})       [식 1.9]
#
# 이 식을 풀어 쓰면 k걸음 전의 보상에 α(1-α)^k 의 가중치가 붙는다.
#   - α가 작으면 가중치가 천천히 줄어 과거를 오래 기억한다
#     → 추정이 안정적이지만 승률이 변하면 느리게 따라간다
#   - α가 크면 최근 몇 번의 보상이 Q를 지배한다
#     → 빠르게 따라가지만 우연한 결과에 출렁인다
#
# 즉 α는 '느리고 안정적' 과 '빠르고 불안정' 사이의 손잡이다.
# =============================================================================

if __name__ == '__main__':
    runs = 200    # 실험 횟수(평균을 내기 위해)
    steps = 1000  # 한 실험당 플레이 횟수
    epsilon = 0.1
    alphas = [0.01, 0.1, 0.3, 0.8]

    # -------------------------------------------------------------------------
    # 1) 비정상 밴디트에서의 승률 비교
    # -------------------------------------------------------------------------
    results = {}
    labels = ['sample average'] + ['alpha = {}'.format(a) for a in alphas]

    for label in labels:
        all_rates = np.zeros((runs, steps))

        for run in range(runs):
            if label == 'sample average':
                agent = Agent(epsilon)  # 표본 평균 방식(α = 1/n 과 같다)
            else:
                agent = AlphaAgent(epsilon, float(label.split('=')[1]))

            bandit = NonStatBandit()  # 승률이 조금씩 변하는 비정상 문제
            total_reward = 0
            rates = []

            for step in range(steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step + 1))

            all_rates[run] = rates

        results[label] = np.average(all_rates, axis=0)

    print('비정상 문제에서 {}회 실험 평균 (steps={}, epsilon={})'.format(
        runs, steps, epsilon))
    print('-' * 44)
    print('{:<20}{:>12}{:>12}'.format('', 'step 100', 'step 1000'))
    print('-' * 44)
    for label in labels:
        rates = results[label]
        print('{:<20}{:>12.4f}{:>12.4f}'.format(label, rates[99], rates[-1]))
    print('-' * 44)
    print('최종 승률 1위:', max(labels, key=lambda k: results[k][-1]))
    print('\nα=0.01은 갱신이 너무 느려 {}걸음 안에 학습을 끝내지 못한다.'.format(steps))
    print('표본 평균 방식도 n이 커질수록 갱신 폭이 1/n로 줄어들어')
    print('나중에는 사실상 학습을 멈춘다.')

    # -------------------------------------------------------------------------
    # 2) 그런데 승률만 봐서는 α의 역할이 잘 안 보인다.
    #    Q가 '진짜 승률'을 얼마나 잘 따라가는지(추적 오차)를 직접 재본다.
    #
    #    승률이 sigma 크기로 무작위 보행하는 팔 하나를 두고,
    #    받은 보상만으로 Q를 갱신하며 |Q - 진짜 승률| 의 평균을 측정한다.
    # -------------------------------------------------------------------------
    def tracking_error(alpha, sigma, steps=4000, runs=500, seed=0):
        """고정값 α로 추정한 Q와 진짜 승률의 평균 절대 오차"""
        rng = np.random.default_rng(seed)
        q_true = rng.random(runs)   # 팔의 진짜 승률(실험마다 다름)
        Q = np.zeros(runs)
        # 초기 수렴 구간은 빼고 잰다. α가 작을수록 자리를 잡는 데 1/α 걸음쯤
        # 걸리므로, 이 구간을 넉넉히 잡지 않으면 작은 α가 억울하게 손해를 본다.
        burn_in = steps // 4
        error = np.zeros(runs)

        for t in range(steps):
            reward = (rng.random(runs) < q_true).astype(float)
            Q += alpha * (reward - Q)                       # [식 1.9]
            if t >= burn_in:
                error += np.abs(Q - q_true)
            # 진짜 승률이 조금씩 흘러간다(sigma=0이면 정상 문제)
            q_true = np.clip(q_true + sigma * rng.standard_normal(runs), 0.05, 0.95)

        return float(np.mean(error / (steps - burn_in)))

    fine_alphas = [0.005, 0.01, 0.03, 0.1, 0.3, 0.8]
    sigmas = [0.0, 0.005, 0.02, 0.08]  # 승률이 변하는 속도: 안 변함 ~ 빠르게 변함

    print('\n\n추적 오차 |Q - 진짜 승률| (작을수록 좋음)')
    print('-' * 62)
    print('{:<10}'.format('sigma')
          + ''.join('{:>10}'.format('a=' + str(a)) for a in fine_alphas))
    print('-' * 62)

    grid = {}
    for sigma in sigmas:
        row = [tracking_error(a, sigma) for a in fine_alphas]
        grid[sigma] = row
        mark = ['  '] * len(row)
        mark[int(np.argmin(row))] = ' *'
        print('{:<10}'.format(sigma)
              + ''.join('{:>8.4f}{}'.format(v, m) for v, m in zip(row, mark)))
    print('-' * 62)
    print('(* 는 그 행에서 오차가 가장 작은 α)')

    best_by_sigma = [fine_alphas[int(np.argmin(grid[s]))] for s in sigmas]
    print('\n최적 α:', ', '.join('sigma {} -> {}'.format(s, a)
                                for s, a in zip(sigmas, best_by_sigma)))
    if best_by_sigma == sorted(best_by_sigma) and best_by_sigma[0] != best_by_sigma[-1]:
        print('승률이 빨리 변할수록 최적 α가 커진다.')
        print('빨리 변하는 환경에서는 오래된 보상이 쓸모없어지기 때문이다.')
    print('\n반대로 승률이 아예 안 변하면(sigma=0) α는 작을수록 좋다.')
    print('이때는 α를 1/n로 줄여가는 표본 평균 방식이 이론적으로 최선이다.')

    # -------------------------------------------------------------------------
    # 3) 그래프
    # -------------------------------------------------------------------------
    plt.figure(figsize=(15, 8))

    # 승률 추이
    plt.subplot(2, 2, 1)
    for label in labels:
        plt.plot(results[label], label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average Rates')
    plt.title('Non-stationary bandit')
    plt.legend()
    plt.grid(True)

    # 과거 보상에 붙는 가중치 α(1-α)^k
    plt.subplot(2, 2, 2)
    ks = np.arange(50)
    for alpha in alphas:
        plt.plot(ks, alpha * (1 - alpha) ** ks, label='alpha = {}'.format(alpha))
    plt.xlabel('k (how many steps ago)')
    plt.ylabel('weight on that reward')
    plt.title('Weight of past rewards: alpha(1-alpha)^k')
    plt.legend()
    plt.grid(True)

    # 추적 오차 곡선
    plt.subplot(2, 2, 3)
    for sigma in sigmas:
        plt.plot(fine_alphas, grid[sigma], marker='o', label='sigma = {}'.format(sigma))
    plt.xscale('log')
    plt.xlabel('alpha (log scale)')
    plt.ylabel('mean |Q - true rate|')
    plt.title('Best alpha grows as the world changes faster')
    plt.legend()
    plt.grid(True)

    # Q가 진짜 승률을 따라가는 모습
    plt.subplot(2, 2, 4)
    rng = np.random.default_rng(1)
    T, sigma = 600, 0.02
    q_true = np.empty(T)
    q = 0.5
    for t in range(T):
        q_true[t] = q
        q = float(np.clip(q + sigma * rng.standard_normal(), 0.05, 0.95))
    rewards = (rng.random(T) < q_true).astype(float)

    plt.plot(q_true, color='black', linewidth=2, label='true rate')
    for alpha in [0.01, 0.1, 0.8]:
        Q, trace = 0.0, []
        for r in rewards:
            Q += alpha * (r - Q)
            trace.append(Q)
        plt.plot(trace, alpha=0.8, label='alpha = {}'.format(alpha))
    plt.xlabel('Steps')
    plt.ylabel('Q')
    plt.title('Small alpha lags, large alpha jitters')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
