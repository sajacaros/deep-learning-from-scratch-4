if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
from common.two_square_grid import TwoSquareGrid


# =============================================================================
# 3.1절 - 벨만 방정식을 '반복 대입'으로 풀기
#
# 벨만 방정식은 v_pi = r_pi + gamma * P_pi @ v_pi 라는 고정점 방정식이다.
# 아무 값에서 시작해 우변에 대입하기를 반복하면 참값으로 수렴한다.
#
#   v_{k+1} = r_pi + gamma * P_pi @ v_k
#
# 한 번 대입할 때마다 오차가 gamma배로 줄어들기 때문이다(축소 사상).
# 이것이 4장 '반복 정책 평가(iterative policy evaluation)'의 정체다.
# =============================================================================


if __name__ == '__main__':
    env = TwoSquareGrid()
    pi = {state: {0: 0.5, 1: 0.5} for state in env.states()}  # 무작위 정책

    plt.figure(figsize=(10, 4))

    for plot_i, gamma in enumerate([0.9, 0.5]):
        P, r = env.policy_matrices(pi, gamma)
        I = np.eye(len(P))

        # 정확해 (s01_bellman_linear.py 와 같은 방법)
        v_exact = np.linalg.solve(I - gamma * P, r)

        # 반복 대입
        v = np.zeros(len(P))
        errors = []
        for k in range(60):
            errors.append(np.max(np.abs(v - v_exact)))
            v = r + gamma * P @ v

        print('gamma = {}'.format(gamma))
        print('  정확해   : v(L1)={:.6f}, v(L2)={:.6f}'.format(*v_exact))
        print('  60회 반복: v(L1)={:.6f}, v(L2)={:.6f}'.format(*v))
        print('  최종 오차: {:.3e}'.format(np.max(np.abs(v - v_exact))))

        # 오차가 매 반복마다 정확히 gamma배로 줄어드는지 확인
        # (k=0은 초깃값에서 출발하는 첫 걸음이라 예외적으로 더 많이 줄어든다)
        ratios = [errors[k + 1] / errors[k] for k in range(1, 11)]
        print('  반복당 오차 감소 비율: {:.4f}  (gamma = {})\n'.format(
            float(np.mean(ratios)), gamma))
        assert np.allclose(ratios, gamma, atol=1e-6)

        plt.subplot(1, 2, plot_i + 1)
        plt.semilogy(errors, marker='.')
        plt.xlabel('iteration k')
        plt.ylabel('max |v_k - v_pi|')
        plt.title('gamma = {}'.format(gamma))
        plt.grid(True)

    print('gamma가 작을수록 빨리 수렴한다. 대신 먼 미래를 보지 못한다.')

    plt.tight_layout()
    plt.show()
