if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
import numpy as np
from common.two_square_grid import TwoSquareGrid
from common.gridworld import GridWorld


# =============================================================================
# 3.1~3.2절 - 벨만 방정식을 '연립일차방정식'으로 풀기
#
#   v_pi(s) = sum_a pi(a|s) sum_s' p(s'|s,a) { r(s,a,s') + gamma * v_pi(s') }
#
# 이 식은 v_pi에 대해 '일차식'이다. 상태 수만큼의 미지수와 방정식이 있으므로
# 행렬로 묶으면 다음과 같이 한 방에 풀린다.
#
#   v = r_pi + gamma * P_pi @ v   =>   (I - gamma*P_pi) v = r_pi
#
# 반복 계산(4장의 정책 평가) 없이 정확한 해를 얻을 수 있다.
# =============================================================================


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # 1) 두 칸짜리 그리드 월드 (책 3.2절의 예)
    # -----------------------------------------------------------------------------
    env = TwoSquareGrid()
    gamma = 0.9

    # 무작위 정책: 각 상태에서 Left/Right를 0.5씩
    pi = {state: {0: 0.5, 1: 0.5} for state in env.states()}

    P, r = env.policy_matrices(pi, gamma)
    print('무작위 정책의 상태 전이 행렬 P_pi =\n', P)
    print('무작위 정책의 기대 보상 r_pi =', r)

    I = np.eye(len(P))
    v = np.linalg.solve(I - gamma * P, r)  # (I - gamma*P) v = r 를 푼다

    print('\n[무작위 정책]  연립방정식의 해')
    for state in env.states():
        print('  v_pi({}) = {:.4f}'.format(state, v[env.state_index(state)]))
    print('  (책 3.2절의 v_pi(L1)=-2.25, v_pi(L2)=-2.75 와 같다)')

    # ch04/s01_dp.py 가 반복 계산으로 구한 값과 일치하는지 확인
    assert np.allclose(v, [-2.25, -2.75], atol=1e-6)

    # 정책을 바꾸면 해도 바뀐다: mu = {L1: Right, L2: Left}
    mu = {'L1': {0: 0.0, 1: 1.0}, 'L2': {0: 1.0, 1: 0.0}}
    P_mu, r_mu = env.policy_matrices(mu, gamma)
    v_mu = np.linalg.solve(I - gamma * P_mu, r_mu)
    print('\n[정책 mu = {L1:Right, L2:Left}]')
    for state in env.states():
        print('  v_mu({}) = {:.4f}'.format(state, v_mu[env.state_index(state)]))
    print('  (ch02/s03_policy_bruteforce.py 가 500걸음 시뮬레이션으로 구한 값과 같다)')


    # -----------------------------------------------------------------------------
    # 2) 3x4 그리드 월드 (4장에서 반복 계산으로 푸는 그 문제)
    #
    # 목표 상태는 종료 상태이므로 v(goal) = 0 이라는 식을 그대로 넣어준다.
    # -----------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('3x4 그리드 월드를 연립방정식으로 풀기')
    print('=' * 60)

    env = GridWorld()
    gamma = 0.9
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})  # 무작위 정책

    states = list(env.states())
    index = {state: i for i, state in enumerate(states)}
    n = len(states)

    A = np.zeros((n, n))
    b = np.zeros(n)

    for state in states:
        i = index[state]
        if state == env.goal_state:
            A[i, i] = 1.0  # v(goal) = 0
            b[i] = 0.0
            continue

        A[i, i] = 1.0  # 좌변의 v(s)
        for action, action_prob in pi[state].items():
            next_state = env.next_state(state, action)
            reward = float(env.reward(state, action, next_state))
            b[i] += action_prob * reward
            A[i, index[next_state]] -= action_prob * gamma  # 우변의 gamma*v(s')를 이항

    v = np.linalg.solve(A, b)
    V = {state: v[index[state]] for state in states}

    print('무작위 정책의 가치 함수 (연립방정식의 정확해)')
    for h in range(env.height):
        print('  ' + ' '.join('{:>7.2f}'.format(V[(h, w)]) for w in range(env.width)))

    # 4장의 반복 계산(정책 평가) 결과와 비교한다.
    from ch04.s04_policy_eval import policy_eval

    V_iter = policy_eval(pi, defaultdict(lambda: 0), env, gamma, threshold=1e-10)
    max_diff = max(abs(V[state] - V_iter[state]) for state in states)
    print('\nch04/s04_policy_eval.py 의 반복 계산 결과와의 최대 오차: {:.3e}'.format(max_diff))
    assert max_diff < 1e-6, '두 방법의 해가 일치해야 한다'
    print('두 방법의 해가 일치한다. 즉 4장의 반복 계산은 이 연립방정식을 푸는 다른 방법이다.')

    # [그림] 무작위 정책의 가치 함수
    env.render_v(V, pi)
