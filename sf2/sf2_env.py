# StreetFighterWrapper는 linyiLYi/street-fighter-ai의 street_fighter_custom_wrapper.py를
# 옮긴 것이다. 원본: Copyright 2023 LIN Yi, Apache License 2.0
# (https://github.com/linyiLYi/street-fighter-ai, http://www.apache.org/licenses/LICENSE-2.0)
#
# 바꾼 점: gym-retro/gym → stable-retro/gymnasium API(reset·step 반환값)로 옮겼다.
#          화면 전처리·걸음 길이·보상·종료 조건은 원본과 같다.
#          DQN용으로 행동을 번호로 고르게 하는 Discretizer를 덧붙였다(원본에는 없음).
import collections
import math
import os
import time

import gymnasium as gym
import numpy as np
import stable_retro as retro

GAME = 'StreetFighterIISpecialChampionEdition-Genesis-v0'
STATE = 'Champion.Level12.RyuVsBison'  # 최고 난이도, 류 vs 베가
INTEGRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'integration')


class StreetFighterWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super().__init__(env)
        self.num_frames = 9                # 최근 9프레임을 보관
        self.frame_stack = collections.deque(maxlen=self.num_frames)
        self.num_step_frames = 6           # 한 행동을 6프레임 동안 누르고 있음
        self.reward_coeff = 3.0            # 준 피해를 받은 피해보다 3배 크게 쳐줌
        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        self.reset_round = reset_round
        self.rendering = rendering

    def _stack_observation(self):
        # 2·5·8번째 프레임의 R·G·B 채널을 하나씩 뽑아 한 장으로 합친다(움직임이 색으로 담김)
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp
        self.total_timesteps = 0

        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])  # 가로세로 절반으로 축소
        return self._stack_observation(), info

    def step(self, action):
        for _ in range(self.num_step_frames):
            obs, _reward, _terminated, _truncated, info = self.env.step(action)
            self.frame_stack.append(obs[::2, ::2, :])
            if self.rendering:
                time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        self.total_timesteps += self.num_step_frames

        if curr_player_health < 0:    # 졌음: 상대 남은 체력이 많을수록 큰 벌점
            custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))
            custom_done = True
        elif curr_oppont_health < 0:  # 이겼음: 내 남은 체력이 많을수록 큰 보상
            custom_reward = math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff
            custom_done = True
        else:                         # 싸우는 중: 준 피해 ×3 − 받은 피해
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) \
                - (self.prev_player_health - curr_player_health)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        if not self.reset_round:
            custom_done = False

        # 보상 최대치가 약 1054라 0.001을 곱해 정규화
        return self._stack_observation(), 0.001 * custom_reward, custom_done, False, info


# 버튼 순서: ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
# 6버튼 패드 기준 X·Y·Z = 약·중·강 펀치, A·B·C = 약·중·강 킥
ACTIONS = [
    [],                  # 가만히
    ['LEFT'], ['RIGHT'], ['UP'], ['DOWN'],
    ['UP', 'LEFT'], ['UP', 'RIGHT'],
    ['DOWN', 'LEFT'], ['DOWN', 'RIGHT'],
    ['X'], ['Y'], ['Z'],  # 펀치
    ['A'], ['B'], ['C'],  # 킥
    ['DOWN', 'Z'],       # 앉아 강펀치
    ['DOWN', 'C'],       # 다리 걸기
]


class Discretizer(gym.ActionWrapper):
    """행동 번호 → 12개 버튼의 눌림 여부. DQN은 Q 값이 가장 큰 번호 하나를 고르므로 필요하다."""
    def __init__(self, env, actions=ACTIONS):
        super().__init__(env)
        buttons = env.unwrapped.buttons
        self._table = []
        for combo in actions:
            arr = np.zeros(len(buttons), dtype=np.int8)
            for button in combo:
                arr[buttons.index(button)] = 1
            self._table.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._table))

    def action(self, act):
        return self._table[act].copy()


def make_env(render_mode=None, discrete=True):
    """참고 프로젝트와 같은 환경. discrete=False면 원본처럼 MultiBinary(12)를 받는다(PPO용)."""
    # integration/의 data.json·scenario.json·.state가 stable-retro 기본 파일보다 먼저 쓰인다
    if INTEGRATION_DIR not in getattr(retro.data.Integrations, 'CUSTOM_PATHS', []):
        retro.data.Integrations.add_custom_path(INTEGRATION_DIR)
    env = retro.make(
        game=GAME,
        state=STATE,
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=render_mode,
    )
    env = StreetFighterWrapper(env, rendering=(render_mode == 'human'))
    if discrete:
        env = Discretizer(env)
    return env
