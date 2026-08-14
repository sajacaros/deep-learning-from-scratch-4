if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
import numpy as np
from common.two_square_grid import TwoSquareGrid
from common.gridworld import GridWorld


# =============================================================================
# 3.3절 - 행동 가치 함수(Q 함수)와 벨만 방정식
#
# 상태 가치 함수 v_pi(s)는 "상태 s에서 정책 pi를 따를 때"의 기대 수익이고,
# 행동 가치 함수 q_pi(s, a)는 "상태 s에서 행동 a를 하고, 그다음부터 pi를 따를 때"의
# 기대 수익이다. 둘은 다음 관계로 이어져 있다.
#
#   q_pi(s,a) = sum_s' p(s'|s,a) { r(s,a,s') + gamma * v_pi(s') }   ... (A) V -> Q
#   v_pi(s)   = sum_a pi(a|s) q_pi(s,a)                             ... (B) Q -> V
#
# 그리고 Q 함수 자신에 대한 벨만 방정식은 다음과 같다.
#
#   q_pi(s,a) = sum_s' p(s'|s,a) { r(s,a,s')
#                                  + gamma * sum_a' pi(a'|s') q_pi(s',a') }
# =============================================================================


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # 1) 두 칸짜리 그리드 월드
    # -----------------------------------------------------------------------------
    env = TwoSquareGrid()
    gamma = 0.9
    pi = {state: {0: 0.5, 1: 0.5} for state in env.states()}  # 무작위 정책

    # 먼저 v_pi를 연립방정식으로 구한다 (s01_bellman_linear.py와 같은 방법)
    P, r = env.policy_matrices(pi, gamma)
    v_vec = np.linalg.solve(np.eye(len(P)) - gamma * P, r)
    V = {state: v_vec[env.state_index(state)] for state in env.states()}

    # (A) V -> Q
    Q = {}
    for state in env.states():
        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            Q[state, action] = reward + gamma * V[next_state]

    print('상태 가치 함수 v_pi')
    for state in env.states():
        print('  v_pi({}) = {:.4f}'.format(state, V[state]))

    print('\n행동 가치 함수 q_pi  (V로부터 계산)')
    for state in env.states():
        for action in env.actions():
            print('  q_pi({}, {:<5}) = {:.4f}'.format(
                state, env.action_meaning[action], Q[state, action]))

    # (B) Q -> V 로 되돌아오는지 확인
    print('\nQ에서 V를 복원: v_pi(s) = sum_a pi(a|s) q_pi(s,a)')
    for state in env.states():
        restored = sum(prob * Q[state, action] for action, prob in pi[state].items())
        print('  {} : {:.6f}  (원래 {:.6f})'.format(state, restored, V[state]))
        assert abs(restored - V[state]) < 1e-9

    # Q에 대한 벨만 방정식이 실제로 성립하는지 확인
    print('\nQ 함수의 벨만 방정식 검증')
    for state in env.states():
        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            rhs = reward + gamma * sum(
                prob * Q[next_state, a] for a, prob in pi[next_state].items())
            assert abs(rhs - Q[state, action]) < 1e-9
            print('  q({}, {:<5}) : 좌변 {:.6f} = 우변 {:.6f}'.format(
                state, env.action_meaning[action], Q[state, action], rhs))


    # -----------------------------------------------------------------------------
    # 2) 3x4 그리드 월드에서도 같은 관계가 성립한다
    # -----------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('3x4 그리드 월드의 Q 함수')
    print('=' * 60)

    env = GridWorld()
    gamma = 0.9
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

    from ch04.s04_policy_eval import policy_eval
    V = policy_eval(pi, defaultdict(lambda: 0), env, gamma, threshold=1e-10)

    Q = {}
    for state in env.states():
        for action in env.actions():
            if state == env.goal_state:
                Q[state, action] = 0.0  # 종료 상태 이후로는 보상이 없다
                continue
            next_state = env.next_state(state, action)
            reward = float(env.reward(state, action, next_state))
            Q[state, action] = reward + gamma * V[next_state]

    # Q -> V 복원 확인
    max_diff = 0.0
    for state in env.states():
        restored = sum(prob * Q[state, action] for action, prob in pi[state].items())
        max_diff = max(max_diff, abs(restored - V[state]))
    print('Q에서 복원한 V와 원래 V의 최대 오차: {:.3e}'.format(max_diff))
    assert max_diff < 1e-6

    # 각 상태에서 Q가 가장 큰 행동이 곧 '탐욕 정책'이다 (4장 정책 개선의 씨앗)
    print('\n각 상태에서 Q가 최대인 행동 (탐욕 정책)')
    for h in range(env.height):
        row = []
        for w in range(env.width):
            state = (h, w)
            if state == env.wall_state:
                row.append('  WALL')
                continue
            if state == env.goal_state:
                row.append('  GOAL')
                continue
            best = max(env.actions(), key=lambda a: Q[state, a])
            row.append('{:>6}'.format(env.action_meaning[best]))
        print('  ' + ' '.join(row))

    # [그림] 무작위 정책의 Q 함수
    env.render_q(Q)
