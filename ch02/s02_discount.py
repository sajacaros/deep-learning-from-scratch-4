if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import os
import matplotlib.pyplot as plt
from common.two_square_grid import TwoSquareGrid


if __name__ == '__main__':
    # =============================================================================
    # 2.3절 - 할인율 gamma는 무엇을 하는가?
    #
    #   G_t = R_t + gamma*R_{t+1} + gamma^2*R_{t+2} + ...
    #
    # gamma가 작으면 눈앞의 보상만 보고, gamma가 1에 가까우면 먼 미래까지 내다본다.
    # 또한 gamma < 1 이어야 연속적 과제에서 수익이 무한대로 발산하지 않는다.
    # =============================================================================
    env = TwoSquareGrid()

    # 비교할 두 정책
    policies = {
        'mu1 (L1:Right, L2:Left)': {'L1': 1, 'L2': 0},   # 사과를 계속 먹으러 오간다
        'mu2 (L1:Right, L2:Right)': {'L1': 1, 'L2': 1},  # 사과를 한 번 먹고 벽에 계속 부딪힌다
    }


    def rewards_of(env, policy, steps):
        """정책대로 steps번 진행하며 받은 보상들을 순서대로 반환"""
        state = env.reset()
        rewards = []
        for _ in range(steps):
            next_state, reward, done = env.step(policy[state])
            rewards.append(reward)
            state = next_state
        return rewards


    def calc_return(rewards, gamma):
        G = 0.0
        for reward in reversed(rewards):
            G = reward + gamma * G
        return G


    # -----------------------------------------------------------------------------
    # 1) gamma에 따라 어떤 정책이 좋아 보이는지가 달라진다
    # -----------------------------------------------------------------------------
    gammas = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]
    steps = 1000  # gamma=0.99 에서도 충분히 수렴하는 길이

    print('gamma에 따른 수익 G_0 (시작 상태 L1)')
    print('-' * 58)
    print('{:>7}{:>25}{:>25}'.format('gamma', *policies.keys()))
    print('-' * 58)
    for gamma in gammas:
        values = [calc_return(rewards_of(env, policy, steps), gamma)
                  for policy in policies.values()]
        print('{:>7.2f}{:>25.4f}{:>25.4f}'.format(gamma, *values))

    print('\ngamma=0 이면 바로 다음 보상만 보므로 두 정책이 똑같아 보인다.')
    print('gamma가 커질수록 "L2에서 계속 벽에 부딪히는" mu2의 손해가 드러난다.')


    # -----------------------------------------------------------------------------
    # 2) 할인 가중치 gamma^k 자체를 그려본다
    # -----------------------------------------------------------------------------
    ks = list(range(30))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for gamma in [0.5, 0.9, 0.99]:
        plt.plot(ks, [gamma ** k for k in ks], label='gamma={}'.format(gamma))
    plt.xlabel('steps ahead (k)')  # 그래프 라벨은 한글 폰트 문제를 피해 영문으로 표기
    plt.ylabel('weight  gamma^k')
    plt.title('Discount weight')
    plt.legend()
    plt.grid(True)

    # -----------------------------------------------------------------------------
    # 3) 수익의 부분합이 수렴하는 모습
    # -----------------------------------------------------------------------------
    plt.subplot(1, 2, 2)
    for gamma in [0.5, 0.9, 0.99]:
        rewards = rewards_of(env, policies['mu1 (L1:Right, L2:Left)'], 200)
        partial = []
        G = 0.0
        for k, reward in enumerate(rewards):
            G += (gamma ** k) * reward
            partial.append(G)
        plt.plot(partial, label='gamma={}'.format(gamma))
    plt.xlabel('steps included')
    plt.ylabel('partial sum of G_0')
    plt.title('Return converges (policy mu1)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # README에 싣는 그림이므로 화면에 띄우기 전에 파일로도 남긴다.
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    os.makedirs(image_dir, exist_ok=True)
    save_path = os.path.join(image_dir, 's02_discount.png')
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    print('\n그래프 저장:', save_path)

    plt.show()
