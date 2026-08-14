if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import itertools
import numpy as np
from common.two_square_grid import TwoSquareGrid


# =============================================================================
# 3.4~3.5절 - 벨만 최적 방정식
#
# 최적 정책의 가치 함수 v*는 다음을 만족한다.
#
#   v*(s) = max_a sum_s' p(s'|s,a) { r(s,a,s') + gamma * v*(s') }
#
# 벨만 방정식과 달리 'sum_a pi(a|s)' 자리에 'max_a'가 들어간다.
# max 때문에 더 이상 일차식이 아니어서 행렬로 한 번에 풀 수 없다.
# 대신 (1) 손으로 경우를 나눠 풀거나, (2) 반복 대입으로 풀 수 있다.
# =============================================================================
env = TwoSquareGrid()
gamma = 0.9


# -----------------------------------------------------------------------------
# 방법 1) 반복 대입 - 벨만 최적 방정식의 우변을 계속 대입한다 (5장 가치 반복법)
# -----------------------------------------------------------------------------
V = {state: 0.0 for state in env.states()}

for k in range(1000):
    new_V = {}
    for state in env.states():
        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            action_values.append(reward + gamma * V[next_state])
        new_V[state] = max(action_values)  # 벨만 방정식과 다른 부분: max
    delta = max(abs(new_V[s] - V[s]) for s in env.states())
    V = new_V
    if delta == 0.0:  # 더 이상 값이 변하지 않으면 수렴
        break

print('벨만 최적 방정식을 반복 대입으로 푼 결과 ({}회 반복)'.format(k + 1))
for state in env.states():
    print('  v*({}) = {:.4f}'.format(state, V[state]))

# v*로부터 최적 정책을 뽑는다 (탐욕화)
optimal_policy = {}
for state in env.states():
    best = max(env.actions(),
               key=lambda a: env.reward(state, a, env.next_state(state, a))
               + gamma * V[env.next_state(state, a)])
    optimal_policy[state] = best

print('\n최적 정책')
for state in env.states():
    print('  {} -> {}'.format(state, env.action_meaning[optimal_policy[state]]))


# -----------------------------------------------------------------------------
# 방법 2) 손으로 푼 결과 확인 (책 3.5절)
#
# L1에서 Right, L2에서 Left를 고른다고 가정하면 max가 사라지고 일차식이 된다.
#   v*(L1) = 1 + gamma * v*(L2)
#   v*(L2) = 0 + gamma * v*(L1)
# 이를 풀면 v*(L1) = 1/(1-gamma^2), v*(L2) = gamma/(1-gamma^2)
# -----------------------------------------------------------------------------
v1_analytic = 1 / (1 - gamma ** 2)
v2_analytic = gamma / (1 - gamma ** 2)
print('\n손으로 푼 해: v*(L1) = 1/(1-gamma^2) = {:.4f}, '
      'v*(L2) = gamma/(1-gamma^2) = {:.4f}'.format(v1_analytic, v2_analytic))
assert abs(V['L1'] - v1_analytic) < 1e-9
assert abs(V['L2'] - v2_analytic) < 1e-9
print('반복 대입의 결과와 일치한다.')

# 가정이 옳았는지 검증: 각 상태에서 정말 그 행동이 max인가?
for state in env.states():
    values = {a: env.reward(state, a, env.next_state(state, a))
              + gamma * V[env.next_state(state, a)] for a in env.actions()}
    assert optimal_policy[state] == max(values, key=values.get)
print('가정한 행동이 실제로 max를 달성하므로 이 해가 벨만 최적 방정식의 해다.')


# -----------------------------------------------------------------------------
# 방법 3) 2장의 전수 탐색 결과와 대조
# -----------------------------------------------------------------------------
print('\n' + '=' * 60)
print('ch02/policy_bruteforce.py 의 전수 탐색 결과와 대조')
print('=' * 60)


def value_of(env, policy, start_state, gamma, horizon=500):
    state, G = start_state, 0.0
    for k in range(horizon):
        next_state = env.next_state(state, policy[state])
        G += (gamma ** k) * env.reward(state, policy[state], next_state)
        state = next_state
    return G


states = list(env.states())
best_policy, best_values = None, None
for actions in itertools.product(env.actions(), repeat=len(states)):
    policy = dict(zip(states, actions))
    values = {s: value_of(env, policy, s, gamma) for s in states}
    if best_values is None or values['L1'] > best_values['L1']:
        best_policy, best_values = policy, values

print('전수 탐색 최적 정책: L1 -> {}, L2 -> {}'.format(
    env.action_meaning[best_policy['L1']], env.action_meaning[best_policy['L2']]))
print('전수 탐색 최적 가치: v*(L1) = {:.4f}, v*(L2) = {:.4f}'.format(
    best_values['L1'], best_values['L2']))

assert best_policy == optimal_policy
assert np.allclose([best_values[s] for s in states], [V[s] for s in states], atol=1e-6)
print('\n두 방법의 답이 같다.')
print('전수 탐색은 정책 수가 |A|^|S| 로 폭발하지만,')
print('벨만 최적 방정식은 상태 수에 비례하는 계산만으로 같은 답을 준다.')
