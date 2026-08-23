"""논문 그림 생성 스크립트 (ch04_dp.tex의 그림 하나를 만든다).

왼쪽 : 두 칸짜리 그리드 월드에서 야코비(배열 둘)와 가우스-자이델(제자리) 갱신의
       오차 감소를 비교한다. 점선은 각 반복 행렬의 스펙트럼 반지름이 예측하는 기울기다.
오른쪽: 3x4 그리드 월드의 반복적 정책 평가에서 '실제 오차'와 정지 규칙이 보는
       'delta x gamma/(1-gamma)' 상한을 나란히 그린다.

실행:  python ch04/paper/fig_convergence.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common.gridworld import GridWorld
from ch04.s04_policy_eval import eval_onestep

GAMMA = 0.9
N_ITER = 70


def two_square_curves():
    """두 칸 세계에서 야코비/가우스-자이델의 오차 열을 만든다."""
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    r = np.array([0.0, -0.5])
    v_exact = np.linalg.solve(np.eye(2) - GAMMA * P, r)

    jacobi, gauss = [], []
    vj = np.zeros(2)
    vg = {'L1': 0.0, 'L2': 0.0}
    for _ in range(N_ITER):
        jacobi.append(np.max(np.abs(vj - v_exact)))
        gauss.append(max(abs(vg['L1'] - v_exact[0]), abs(vg['L2'] - v_exact[1])))
        vj = r + GAMMA * P @ vj
        vg['L1'] = 0.5 * (-1 + GAMMA * vg['L1']) + 0.5 * (1 + GAMMA * vg['L2'])
        vg['L2'] = 0.5 * (0 + GAMMA * vg['L1']) + 0.5 * (-1 + GAMMA * vg['L2'])

    # 반복 행렬의 스펙트럼 반지름
    L = np.tril(P, -1)
    DU = P - L
    rho_j = max(abs(np.linalg.eigvals(GAMMA * P)))
    rho_g = max(abs(np.linalg.eigvals(np.linalg.inv(np.eye(2) - GAMMA * L) @ (GAMMA * DU))))
    return jacobi, gauss, rho_j, rho_g


def grid_curves():
    """3x4 그리드에서 스윕마다 실제 오차와 delta 기반 상한을 잰다."""
    env = GridWorld()
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

    exact = defaultdict(lambda: 0)
    for _ in range(400):  # 충분히 수렴시킨 값을 참값으로 삼는다
        exact = eval_onestep(pi, exact, env, GAMMA)

    V = defaultdict(lambda: 0)
    errors, bounds = [], []
    for _ in range(N_ITER):
        old = V.copy()
        V = eval_onestep(pi, V, env, GAMMA)
        delta = max(abs(V[s] - old[s]) for s in V.keys())
        errors.append(max(abs(V[s] - exact[s]) for s in env.states()))
        bounds.append(delta * GAMMA / (1 - GAMMA))
    return errors, bounds


def main():
    jacobi, gauss, rho_j, rho_g = two_square_curves()
    errors, bounds = grid_curves()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ks = np.arange(N_ITER)
    ax1.semilogy(ks, jacobi, label=f'Jacobi (two arrays), rho={rho_j:.4f}')
    ax1.semilogy(ks, gauss, label=f'Gauss-Seidel (in-place), rho={rho_g:.4f}')
    ax1.semilogy(ks, jacobi[1] * rho_j ** (ks - 1.0), 'k:', lw=0.8)
    ax1.semilogy(ks, gauss[1] * rho_g ** (ks - 1.0), 'k:', lw=0.8)
    ax1.set_xlabel('sweep k')
    ax1.set_ylabel(r'max $|V_k - v_\pi|$')
    ax1.set_title('two-square grid world')
    ax1.set_ylim(1e-4, 5)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(ks, errors, label=r'actual  max$|V_k - v_\pi|$')
    ax2.semilogy(ks, bounds, '--', label=r'bound  $\delta_k\,\gamma/(1-\gamma)$')
    ax2.set_xlabel('sweep k')
    ax2.set_ylabel('error')
    ax2.set_title(r'$3\times4$ grid world, policy evaluation')
    ax2.set_ylim(1e-7, 5)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_convergence.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('그래프 저장:', out)


if __name__ == '__main__':
    main()
