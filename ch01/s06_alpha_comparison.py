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
    env_sigma = 0.1  # 환경이 변하는 속도. epsilon(탐색), alpha(기억)와 달리
                     # 에이전트가 고르는 값이 아니라 문제가 주어지는 방식이다.

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

            bandit = NonStatBandit(sigma=env_sigma)  # 승률이 조금씩 변하는 비정상 문제
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
    # 2) 위 실험은 sigma가 0.1 하나로 고정되어 있다. 그래서 'α는 클수록 좋다'로
    #    읽히기 쉽지만, 그건 그 환경이 빨리 변하기 때문이다.
    #    같은 밴디트에서 sigma만 바꿔가며 다시 재본다. 1)과 완전히 같은
    #    환경·같은 잣대(승률)를 쓴다.
    #
    #    주의: 승률은 α를 고르는 잣대로는 무디다. ε=0.1 이므로 승률의 천장이
    #    0.9*(최고 승률) + 0.1*(평균 승률) 로 막혀 있고, α가 0.1만 넘으면
    #    이미 천장에 붙어 그 위로는 차이가 노이즈에 묻힌다. Q의 순위만 맞으면
    #    Q가 부정확해도 보상은 만점이기 때문이다.
    #    대신 '표본 평균 vs 고정값 α'의 역전은 아주 선명하게 나온다.
    # -------------------------------------------------------------------------
    def steady_rate(agent_kind, sigma, runs=200, steps=1000):
        """비정상 밴디트에서 뒤쪽 절반 구간의 평균 보상.

        인수:
            agent_kind: 'sample average' 이거나 고정값 α(float).
            sigma (float): 진짜 승률이 한 걸음마다 흔들리는 정도.
                0이면 승률이 변하지 않는 정상 문제가 된다.
            runs (int): 평균을 낼 독립 실험 수.
            steps (int): 한 실험당 플레이 횟수.

        반환값:
            float: 앞쪽 절반(학습이 자리를 잡는 구간)을 빼고 잰 평균 보상.
                이 구간을 빼지 않으면 초반이 느린 작은 α가 억울하게 손해를
                본다. 클수록 좋다.
        """
        burn_in = steps // 2
        total = 0.0

        for _ in range(runs):
            if agent_kind == 'sample average':
                agent = Agent(epsilon)
            else:
                agent = AlphaAgent(epsilon, agent_kind)

            bandit = NonStatBandit(sigma=sigma)
            reward_sum = 0

            for step in range(steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                if step >= burn_in:
                    reward_sum += reward

            total += reward_sum / (steps - burn_in)

        return total / runs

    fine_alphas = [0.01, 0.03, 0.1, 0.3, 0.8]
    sigmas = [0.0, 0.01, 0.03, 0.1]  # 승률이 변하는 속도: 안 변함 ~ 빠르게 변함

    print('\n\nsigma별 정상 구간 평균 보상 (클수록 좋음)')
    print('-' * 68)
    print('{:<10}'.format('sigma')
          + ''.join('{:>9}'.format('a=' + str(a)) for a in fine_alphas)
          + '{:>13}'.format('sample avg'))
    print('-' * 68)

    grid = {}
    baseline = {}
    for sigma in sigmas:
        row = [steady_rate(a, sigma) for a in fine_alphas]
        grid[sigma] = row
        baseline[sigma] = steady_rate('sample average', sigma)
        mark = ['  '] * len(row)
        mark[int(np.argmax(row))] = ' *'
        print('{:<10}'.format(sigma)
              + ''.join('{:>7.4f}{}'.format(v, m) for v, m in zip(row, mark))
              + '{:>13.4f}'.format(baseline[sigma]))
    print('-' * 68)
    print('(* 는 그 행에서 승률이 가장 높은 α)')

    print('\n표본 평균 vs 그 sigma에서 가장 좋았던 고정값 α')
    for sigma in sigmas:
        best = fine_alphas[int(np.argmax(grid[sigma]))]
        gap = max(grid[sigma]) - baseline[sigma]
        winner = '고정값 α={}'.format(best) if gap > 0 else '표본 평균'
        print('  sigma {:<6} 승자: {:<14} (차이 {:+.4f})'.format(
            sigma, winner, gap))

    print('\n승률이 아예 안 변하면(sigma=0) α를 1/n로 줄여가는 표본 평균 방식이')
    print('모든 고정값 α를 앞선다. 세상이 변하기 시작해야 비로소 고정값 α가 이긴다.')
    print('1)에서 α=0.8이 이긴 것은 그 환경이 sigma={}로 빠르게 변하기 때문이지,'
          .format(env_sigma))
    print('α가 클수록 좋아서가 아니다.')
    print('\n고정값 α들끼리의 우열은 승률로는 잘 안 보인다(위 표의 오른쪽 세 열이')
    print('거의 붙어 있다). α 자체를 고르려면 승률이 아니라 |Q - 진짜 승률| 같은')
    print('추정 오차를 재야 한다.')

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
    plt.title('Non-stationary bandit (sigma = {})'.format(env_sigma))
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

    # sigma별 최적 α
    plt.subplot(2, 2, 3)
    for i, sigma in enumerate(sigmas):
        row = grid[sigma]
        line, = plt.plot(fine_alphas, row, marker='o',
                         label='sigma = {}'.format(sigma))
        # 그 행에서 가장 좋은 α를 별로 표시한다
        best = int(np.argmax(row))
        plt.plot(fine_alphas[best], row[best], marker='*', markersize=16,
                 color=line.get_color(), linestyle='none', label='_nolegend_')
        # 같은 sigma에서 표본 평균 방식이 낸 승률(점선)
        plt.axhline(baseline[sigma], color=line.get_color(), linestyle=':',
                    linewidth=1,
                    label='sample average (dotted)'
                    if i == len(sigmas) - 1 else '_nolegend_')
    plt.xscale('log')
    plt.xlabel('alpha (log scale)')
    plt.ylabel('mean reward (2nd half)')
    plt.title('Fixed alpha beats sample average only when sigma > 0')
    plt.legend(fontsize=8)
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
