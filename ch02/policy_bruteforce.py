if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import itertools
from common.two_square_grid import TwoSquareGrid


# =============================================================================
# 2.4절 - MDP 예제: 최적 정책은 무엇인가?
#
# MDP의 목표는 '수익의 기댓값을 최대로 만드는 정책'을 찾는 것이다.
# 상태가 2개, 행동이 2개뿐인 이 문제에서 결정적 정책은 2^2 = 4가지밖에 없다.
# 그러니 전부 만들어보고 가장 좋은 것을 고르면 된다(전수 탐색).
#
# 아직 벨만 방정식(3장)을 배우지 않았으므로,
# 가치는 '수익의 정의' 그대로 충분히 긴 걸음 수의 할인 합으로 구한다.
# =============================================================================
env = TwoSquareGrid()
gamma = 0.9
HORIZON = 500  # gamma^500 ~ 1e-23 이므로 사실상 무한 걸음과 같다


def value_of(env, policy, start_state, gamma, horizon=HORIZON):
    """정책을 start_state에서 실행했을 때의 상태 가치 v_pi(start_state)"""
    state = start_state
    G = 0.0
    for k in range(horizon):
        next_state = env.next_state(state, policy[state])
        reward = env.reward(state, policy[state], next_state)
        G += (gamma ** k) * reward
        state = next_state
    return G


# 결정적 정책을 전부 만든다: {L1: a1, L2: a2} 조합 4가지
states = list(env.states())
all_policies = [dict(zip(states, actions))
                for actions in itertools.product(env.actions(), repeat=len(states))]

print('결정적 정책 {}가지를 모두 평가 (gamma={})'.format(len(all_policies), gamma))
print('-' * 56)
print('{:<28}{:>13}{:>13}'.format('policy', 'v(L1)', 'v(L2)'))
print('-' * 56)

results = []
for policy in all_policies:
    values = {state: value_of(env, policy, state, gamma) for state in states}
    results.append((policy, values))

    label = '{{L1:{:<5} L2:{:<5}}}'.format(
        env.action_meaning[policy['L1']], env.action_meaning[policy['L2']])
    print('{:<28}{:>13.4f}{:>13.4f}'.format(label, values['L1'], values['L2']))

# 모든 상태에서 동시에 가장 좋은 정책이 존재한다는 것이 MDP의 중요한 성질이다.
best_policy, best_values = max(results, key=lambda x: x[1]['L1'])
assert all(best_values[s] >= values[s] for _, values in results for s in states), \
    '모든 상태에서 동시에 최적인 정책이 있어야 한다'

print('-' * 56)
print('최적 정책: L1 -> {}, L2 -> {}'.format(
    env.action_meaning[best_policy['L1']], env.action_meaning[best_policy['L2']]))
print('최적 가치: v*(L1) = {:.4f}, v*(L2) = {:.4f}'.format(
    best_values['L1'], best_values['L2']))
print('\nL1에서 오른쪽으로 가 사과를 먹고, L2에서 왼쪽으로 돌아와')
print('다시 사과를 먹으러 가는 왕복 행동이 최선이다.')
print('\n다만 이 방법은 정책 수가 |A|^|S| 로 폭발한다.')
print('상태가 100개, 행동이 4개면 4^100 가지 — 전수 탐색은 불가능하다.')
print('그래서 3장의 벨만 방정식이 필요하다. (ch03/bellman_optimal.py 참고)')
