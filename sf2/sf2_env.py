# StreetFighterWrapper는 linyiLYi/street-fighter-ai의 street_fighter_custom_wrapper.py를
# 옮긴 것이다. 원본: Copyright 2023 LIN Yi, Apache License 2.0
# (https://github.com/linyiLYi/street-fighter-ai, http://www.apache.org/licenses/LICENSE-2.0)
#
# 바꾼 점: gym-retro/gym → stable-retro/gymnasium API(reset·step 반환값)로 옮겼다.
#          걸음 길이·보상·종료 조건은 원본과 같다.
#          화면은 컬러 대신 흑백으로 쌓고, 한 걸음 안에서 프레임마다 다른 버튼을 받을 수 있게 했다.
#          행동·쿨타임 래퍼 MoveSet을 덧붙였다(원본에는 없음).
import collections
import math
import os
import time

import gymnasium as gym
import numpy as np
import stable_retro as retro

GAME = 'StreetFighterIISpecialChampionEdition-Genesis-v0'
STATE = 'Champion.Level12.RyuVsBison'  # 최고 난이도, 류 vs 바이슨
INTEGRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'integration')
GRAY = np.array([0.299, 0.587, 0.114])  # RGB → 밝기


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

    def _shrink(self, frame):
        return (frame[::2, ::2, :] @ GRAY).astype(np.uint8)  # 가로세로 절반으로 줄이고 흑백으로

    def _stack_observation(self):
        # 3·6·9번째 프레임의 흑백 화면을 채널로 쌓는다(움직임이 담김)
        return np.stack([self.frame_stack[i * 3 + 2] for i in range(3)], axis=-1)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp
        self.total_timesteps = 0

        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(self._shrink(observation))
        return self._stack_observation(), info

    def step(self, action):
        for i in range(self.num_step_frames):
            # (6, 12) 배열이면 프레임마다 다른 버튼을 누른다(필살기 커맨드)
            buttons = action[i] if np.ndim(action) == 2 else action
            obs, _reward, _terminated, _truncated, info = self.env.step(buttons)
            self.frame_stack.append(self._shrink(obs))
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
BASIC_ACTIONS = [        # 한 걸음(6프레임) 내내 누르는 버튼 조합
    [],                  # 가만히
    ['LEFT'], ['RIGHT'], ['UP'], ['DOWN'],
    ['UP', 'LEFT'], ['UP', 'RIGHT'],
    ['DOWN', 'LEFT'], ['DOWN', 'RIGHT'],
    ['X'], ['Y'], ['Z'],  # 펀치
    ['A'], ['B'], ['C'],  # 킥
    ['DOWN', 'Z'],       # 앉아 강펀치
    ['DOWN', 'C'],       # 다리 걸기
]
SPECIALS = {             # 필살기 커맨드(오른쪽을 볼 때). 한 걸음 6프레임을 2프레임씩 나눠 누른다
    'HADOUKEN': [['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT', 'Z']],         # 파동권
    'SHORYUKEN': [['RIGHT'], ['DOWN'], ['DOWN', 'RIGHT', 'Z']],        # 승룡권
    'TATSUMAKI': [['DOWN'], ['DOWN', 'LEFT'], ['LEFT', 'C']],          # 용권선풍각
}
ACTION_NAMES = ['+'.join(a) if a else 'NOOP' for a in BASIC_ACTIONS] + list(SPECIALS)
N_BASIC = len(BASIC_ACTIONS)  # 이 번호부터가 필살기
COOLDOWN = 20                 # 필살기를 쓰고 나서 다시 못 쓰는 걸음 수(세 필살기가 함께 씀)


def special_ready(state):
    """상태의 네 번째 채널(남은 쿨타임)이 0이면 필살기를 쓸 수 있다. 배치(numpy·torch)도 받는다."""
    return state[..., 0, 0, -1] == 0


def random_action(state, rng):
    """쓸 수 있는 행동 중에서 무작위로 고른다(필살기가 맨 뒤 번호라 쿨타임 중이면 앞쪽만)."""
    return int(rng.integers(len(ACTION_NAMES) if special_ready(state) else N_BASIC))


def action_mask(state):
    """고를 수 있는 행동이면 True. 쿨타임 중이면 필살기가 False(MaskablePPO용)."""
    return np.arange(len(ACTION_NAMES)) < (len(ACTION_NAMES) if special_ready(state) else N_BASIC)


class MoveSet(gym.Wrapper):
    """행동 번호 → 버튼. 0~16은 버튼 조합, 17~19는 필살기 커맨드.
    필살기는 상대가 왼쪽에 있으면 좌우를 뒤집어 넣고, 쓰고 나면 COOLDOWN걸음 동안 막힌다.
    남은 쿨타임을 상태의 네 번째 채널로 붙여 에이전트가 볼 수 있게 한다."""
    def __init__(self, env):
        super().__init__(env)
        buttons = env.unwrapped.buttons

        def press(combo):
            arr = np.zeros(len(buttons), dtype=np.int8)
            for button in combo:
                arr[buttons.index(button)] = 1
            return arr

        def flip(combo):
            swap = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
            return [swap.get(b, b) for b in combo]

        self._basic = [press(c) for c in BASIC_ACTIONS]
        # facing_left → 필살기별 (6, 12) 버튼 배열
        self._special = {
            left: [np.repeat(np.stack([press(flip(c) if left else c) for c in cmd]), 2, axis=0)
                   for cmd in SPECIALS.values()]
            for left in (False, True)
        }
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, c + 1), dtype=np.uint8)
        self.cooldown = 0
        self.facing_left = False

    def _observation(self, obs):
        level = round(255 * self.cooldown / COOLDOWN)  # 0 = 쓸 수 있음, 255 = 방금 씀
        return np.concatenate([obs, np.full((*obs.shape[:2], 1), level, dtype=np.uint8)], axis=-1)

    def action_masks(self):  # MaskablePPO가 행동을 고르기 전에 부른다
        return np.arange(len(ACTION_NAMES)) < (len(ACTION_NAMES) if self.cooldown == 0 else N_BASIC)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cooldown = 0
        self.facing_left = False  # 라운드 시작에는 류가 왼쪽에 선다
        return self._observation(obs), info

    def step(self, action):
        if action < N_BASIC:
            buttons = self._basic[action]
            self.cooldown = max(0, self.cooldown - 1)
        elif self.cooldown == 0:
            buttons = self._special[self.facing_left][action - N_BASIC]
            self.cooldown = COOLDOWN
        else:  # 쿨타임 중에 고르면 가만히 있는다(에이전트는 마스크로 애초에 고르지 않는다)
            buttons = self._basic[0]
            self.cooldown -= 1
        obs, reward, terminated, truncated, info = self.env.step(buttons)
        self.facing_left = info['agent_x'] > info['enemy_x']
        return self._observation(obs), reward, terminated, truncated, info


class RandomStart(gym.Wrapper):
    """reset() 뒤에 0~max_steps걸음을 무작위 행동으로 진행해 출발 상황을 흔든다(evaluate.py의 random start와 같음).
    이 걸음들의 보상은 에이전트에게 넘기지 않는다."""
    def __init__(self, env, max_steps, seed):
        super().__init__(env)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        while True:
            obs, info = self.env.reset(**kwargs)
            for _ in range(self.rng.integers(0, self.max_steps + 1)):
                obs, _, terminated, truncated, info = self.env.step(random_action(obs, self.rng))
                if terminated or truncated:
                    break
            else:
                return obs, info


class PygameViewer(gym.Wrapper):
    """에뮬레이터가 내는 매 프레임을 pygame 창에 그린다.
    stable-retro의 human 렌더링은 pyglet+OpenGL(libGLU)이 필요해 WSL에서 막히기 쉬워 대신 쓴다."""
    def __init__(self, env, scale=3, fps=60):
        super().__init__(env)
        import pygame
        self.pygame = pygame
        self.scale, self.fps = scale, fps
        self.screen = None
        self.clock = pygame.time.Clock()

    def _show(self, frame):
        pg = self.pygame
        h, w = frame.shape[:2]
        if self.screen is None:
            pg.init()
            self.screen = pg.display.set_mode((w * self.scale, h * self.scale))
            pg.display.set_caption('Street Fighter II')
        pg.event.pump()  # 창이 '응답 없음'이 되지 않게 이벤트를 비운다
        surf = pg.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(pg.transform.scale(surf, self.screen.get_size()), (0, 0))
        pg.display.flip()
        self.clock.tick(self.fps)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._show(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._show(obs)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.screen is not None:
            self.pygame.quit()
        super().close()


def make_env(render_mode=None, random_start=0, seed=None):
    """DQN·PPO가 함께 쓰는 환경. 상태는 흑백 3시점 + 쿨타임 채널, 행동은 20가지.
    random_start를 주면 판마다 처음 0~random_start걸음을 무작위로 움직인 뒤 넘긴다(학습용)."""
    # integration/의 data.json·scenario.json·.state가 stable-retro 기본 파일보다 먼저 쓰인다
    if INTEGRATION_DIR not in getattr(retro.data.Integrations, 'CUSTOM_PATHS', []):
        retro.data.Integrations.add_custom_path(INTEGRATION_DIR)
    env = retro.make(
        game=GAME,
        state=STATE,
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
    )
    if render_mode == 'human':
        env = PygameViewer(env)  # 속도는 PygameViewer가 60fps로 맞추므로 rendering은 끈다
    env = MoveSet(StreetFighterWrapper(env))
    return RandomStart(env, random_start, seed) if random_start else env
