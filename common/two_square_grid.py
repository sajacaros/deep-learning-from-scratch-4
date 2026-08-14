import numpy as np


class TwoSquareGrid:
    """책 2.4절의 '두 칸짜리 그리드 월드' MDP.

        +------+------+
        |  L1  |  L2  |   L2 칸에는 사과(보상 +1)가 놓여 있다.
        +------+------+

    - 상태는 L1, L2 두 개뿐이다.
    - 행동은 Left(0), Right(1) 두 개뿐이다.
    - 벽에 부딪히면 제자리에 머물고 보상 -1을 받는다.
    - L2로 이동하면 사과를 먹어 보상 +1을 받는다(사과는 곧바로 다시 생긴다).
    - 목표 상태가 없는 '연속적 과제'이므로 할인율 gamma로 수익을 수렴시킨다.

    common/gridworld.py의 GridWorld와 같은 이름의 메서드를 제공하므로
    같은 방식으로 다룰 수 있다.
    """

    def __init__(self):
        self.action_space = [0, 1]           # 행동 공간
        self.action_meaning = {              # 행동의 의미
            0: "Left",
            1: "Right",
        }
        self.state_space = ['L1', 'L2']      # 상태 공간
        self.start_state = 'L1'              # 시작 상태
        self.agent_state = self.start_state  # 에이전트의 현재 상태

    def actions(self):
        return self.action_space

    def states(self):
        for state in self.state_space:
            yield state

    def next_state(self, state, action):
        """상태 전이 함수 (결정적이므로 다음 상태를 하나만 반환)"""
        if state == 'L1':
            # Left는 벽에 막혀 제자리, Right는 L2로 이동
            return 'L1' if action == 0 else 'L2'
        else:  # state == 'L2'
            # Left는 L1으로 이동, Right는 벽에 막혀 제자리
            return 'L1' if action == 0 else 'L2'

    def transition_prob(self, state, action, next_state):
        """상태 전이 확률 p(s'|s, a). 이 MDP는 결정적이라 값이 0 또는 1이다."""
        return 1.0 if self.next_state(state, action) == next_state else 0.0

    def reward(self, state, action, next_state):
        """보상 함수 r(s, a, s')"""
        if state == next_state:  # 벽에 부딪혀 제자리에 머문 경우
            return -1.0
        if next_state == 'L2':   # 사과가 있는 칸으로 이동
            return 1.0
        return 0.0               # L2 -> L1

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        self.agent_state = next_state
        # 종료 상태가 없는 연속적 과제이므로 done은 항상 False
        return next_state, reward, False

    def state_index(self, state):
        """상태를 벡터/행렬의 인덱스로 변환 (3장에서 사용)"""
        return self.state_space.index(state)

    def policy_matrices(self, pi, gamma=0.9):
        """정책 pi에 대한 상태 전이 행렬 P_pi와 기대 보상 벡터 r_pi를 만든다.

        pi는 {상태: {행동: 확률}} 형태의 딕셔너리다.
        3장에서 벨만 방정식을 연립일차방정식으로 푸는 데 사용한다.
        """
        n = len(self.state_space)
        P = np.zeros((n, n))
        r = np.zeros(n)

        for state in self.states():
            i = self.state_index(state)
            for action, action_prob in pi[state].items():
                next_state = self.next_state(state, action)
                j = self.state_index(next_state)
                P[i, j] += action_prob
                r[i] += action_prob * self.reward(state, action, next_state)
        return P, r
