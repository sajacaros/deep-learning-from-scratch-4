if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.two_square_grid import TwoSquareGrid


# =============================================================================
# 2.2절 - 환경과 에이전트를 수식으로
#
# MDP는 다음 다섯 가지로 정의된다.
#   S     : 상태의 집합
#   A     : 행동의 집합
#   p     : 상태 전이 확률 p(s'|s, a)
#   r     : 보상 함수 r(s, a, s')
#   gamma : 할인율
# =============================================================================
env = TwoSquareGrid()
gamma = 0.9

print('S (상태 집합)  :', list(env.states()))
print('A (행동 집합)  :', [env.action_meaning[a] for a in env.actions()])
print('gamma (할인율) :', gamma)

print('\n상태 전이 확률 p(s\'|s, a) 와 보상 r(s, a, s\')')
print('-' * 52)
print('{:>5} {:>7} -> {:>5} {:>10} {:>10}'.format(
    's', 'a', "s'", "p(s'|s,a)", "r(s,a,s')"))
print('-' * 52)
for state in env.states():
    for action in env.actions():
        for next_state in env.states():
            prob = env.transition_prob(state, action, next_state)
            if prob == 0:  # 전이할 수 없는 조합은 생략
                continue
            reward = env.reward(state, action, next_state)
            print('{:>5} {:>7} -> {:>5} {:>10.1f} {:>10.1f}'.format(
                state, env.action_meaning[action], next_state, prob, reward))


# =============================================================================
# 2.3절 - MDP의 목표: 수익(return)
#
#   G_t = R_t + gamma*R_{t+1} + gamma^2*R_{t+2} + ...
#
# 정책을 하나 정한 뒤, 정의 그대로 수익을 계산해본다.
# (이 MDP는 결정적이므로 정책만 정해지면 궤적이 하나로 결정된다)
# =============================================================================
def play(env, policy, steps):
    """정책대로 steps번 행동하고 (상태, 행동, 보상) 기록을 반환"""
    state = env.reset()
    history = []
    for _ in range(steps):
        action = policy[state]
        next_state, reward, done = env.step(action)
        history.append((state, action, reward))
        state = next_state
    return history


def calc_return(rewards, gamma):
    """보상 나열로부터 수익 G를 계산 (뒤에서부터 거꾸로 접어 올린다)"""
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
    return G


# 정책 mu: L1에서는 Right, L2에서는 Left (사과를 계속 먹으러 오가는 정책)
mu = {'L1': 1, 'L2': 0}

print('\n\n정책 mu = {L1: Right, L2: Left} 로 10걸음 진행')
print('-' * 52)
history = play(env, mu, steps=10)
for t, (state, action, reward) in enumerate(history):
    print('t={:<2} S_t={:<3} A_t={:<6} R_t={:+.1f}'.format(
        t, state, env.action_meaning[action], reward))

rewards = [reward for _, _, reward in history]
print('\n10걸음까지의 수익 G_0 =', round(calc_return(rewards, gamma), 4))

# 걸음 수를 늘려가며 수익이 어떤 값으로 수렴하는지 확인한다.
print('\n걸음 수를 늘리면 수익 G_0 는 하나의 값으로 수렴한다.')
for steps in [1, 2, 5, 10, 50, 100, 500]:
    rewards = [r for _, _, r in play(env, mu, steps)]
    print('  steps={:<4} G_0 = {:.6f}'.format(steps, calc_return(rewards, gamma)))

# 상태 가치 함수 v_mu(s) = E[G_t | S_t=s] 는 결정적 MDP에서 기댓값 없이 그대로 구해진다.
print('\n정책 mu의 상태 가치 함수 (500걸음 근사)')
for start in env.states():
    env.start_state = start
    rewards = [r for _, _, r in play(env, mu, 500)]
    print('  v_mu({}) = {:.4f}'.format(start, calc_return(rewards, gamma)))
env.start_state = 'L1'

# 참고: 이 값은 3장의 벨만 방정식으로 계산이 아니라 '풀어서' 구할 수 있다.
#       ch03/bellman_linear.py 에서 같은 값을 확인해보자.
