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
# 1인용 모드를 진행하며 만든 상대별 시작 상태(make_states.py). Level 숫자는 스테이지로, 뒤로 갈수록 CPU가 강하다
LADDER = ['Champion.Level1.RyuVsGuile', 'Champion.Level2.RyuVsKen', 'Champion.Level3.RyuVsChunLi',
          'Champion.Level4.RyuVsZangief', 'Champion.Level5.RyuVsDhalsim', 'Champion.Level6.RyuVsRyu',
          'Champion.Level7.RyuVsHonda', 'Champion.Level8.RyuVsBlanka', 'Champion.Level9.RyuVsBalrog',
          'Champion.Level10.RyuVsVega', 'Champion.Level11.RyuVsSagat', 'Champion.Level12.RyuVsBison.Ladder']
HOLDOUT = ['Champion.Level5.RyuVsDhalsim', STATE]  # 학습에 쓰지 않고 평가에만 쓰는 상태(처음 보는 상대, 최고 난이도 바이슨)
TRAIN_STATES = [s for s in LADDER if s not in HOLDOUT]
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
COOLDOWN = 20                 # 필살기를 쓰고 나서 다시 못 쓰는 걸음 수
HIT_WINDOW = [10, 4, 4]       # 필살기를 쓴 뒤 이 걸음 안에 피해가 나면 명중으로 본다(파동권은 날아가는 시간이 있어 길게)


def specials_ready(state):
    """필살기별로 쓸 수 있으면 True, 모양은 (..., 3). 배치(numpy·torch)도 받는다.
    화면 3채널 뒤의 쿨타임 채널이 1개면 세 필살기가 쿨타임을 함께 쓰고, 3개면 따로 쓴다."""
    ready = state[..., 0, 0, 3:] == 0
    if ready.shape[-1] == 1:
        shape = (*ready.shape[:-1], len(SPECIALS))
        ready = ready.expand(shape) if hasattr(ready, 'expand') else np.broadcast_to(ready, shape)
    return ready


def action_mask(state):
    """고를 수 있는 행동이면 True. 쿨타임 중인 필살기가 False(MaskablePPO용)."""
    return np.concatenate([np.ones(N_BASIC, dtype=bool), specials_ready(state)])


def random_action(state, rng):
    """쓸 수 있는 행동 중에서 무작위로 고른다."""
    return int(rng.choice(np.flatnonzero(action_mask(state))))


class MoveSet(gym.Wrapper):
    """행동 번호 → 버튼. 0~16은 버튼 조합, 17~19는 필살기 커맨드.
    필살기는 상대가 왼쪽에 있으면 좌우를 뒤집어 넣고, 쓰고 나면 COOLDOWN걸음 동안 막힌다.
    남은 쿨타임을 화면 뒤 채널로 붙여 에이전트가 볼 수 있게 한다.
      separate: True(기본)면 필살기마다 쿨타임을 따로 둔다(채널 3개). False면 하나를 쓰면 셋 다 막힌다(채널 1개, 이전 방식).
      bonus·penalty: 필살기를 쓴 뒤 HIT_WINDOW걸음 안에 상대가 피해를 입으면 +bonus, 아니면 -penalty를 보상에 더한다."""
    def __init__(self, env, separate=True, bonus=0.0, penalty=0.0):
        super().__init__(env)
        self.separate, self.bonus, self.penalty = separate, bonus, penalty
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
        n = len(SPECIALS) if separate else 1
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, c + n), dtype=np.uint8)
        self.cooldown = np.zeros(len(SPECIALS), dtype=int)  # 필살기별 남은 걸음(함께 쓰면 셋이 늘 같다)
        self.facing_left = False
        self.pending = []  # [남은 판정 걸음, 그동안 상대가 입은 피해]
        self.enemy_hp = 0

    def _observation(self, obs):
        cd = self.cooldown if self.separate else self.cooldown[:1]
        levels = np.round(255 * cd / COOLDOWN).astype(np.uint8)  # 0 = 쓸 수 있음, 255 = 방금 씀
        return np.concatenate([obs, np.broadcast_to(levels, (*obs.shape[:2], len(levels)))], axis=-1)

    def action_masks(self):  # MaskablePPO가 행동을 고르기 전에 부른다
        return np.concatenate([np.ones(N_BASIC, dtype=bool), self.cooldown == 0])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cooldown[:] = 0
        self.facing_left = False  # 라운드 시작에는 류가 왼쪽에 선다
        self.pending = []
        self.enemy_hp = self.env.full_hp
        return self._observation(obs), info

    def step(self, action):
        self.cooldown = np.maximum(0, self.cooldown - 1)
        if action < N_BASIC:
            buttons = self._basic[action]
        elif self.cooldown[action - N_BASIC] == 0:
            k = action - N_BASIC
            buttons = self._special[self.facing_left][k]
            if self.separate:
                self.cooldown[k] = COOLDOWN
            else:
                self.cooldown[:] = COOLDOWN
            if self.bonus or self.penalty:
                self.pending.append([HIT_WINDOW[k], 0])
        else:  # 쿨타임 중에 고르면 가만히 있는다(에이전트는 마스크로 애초에 고르지 않는다)
            buttons = self._basic[0]
        obs, reward, terminated, truncated, info = self.env.step(buttons)
        self.facing_left = info['agent_x'] > info['enemy_x']

        # 판정 창 안의 필살기마다 상대가 입은 피해를 더하고, 창이 끝나면(또는 판이 끝나면) 보너스·벌점을 준다
        dealt = max(0, self.enemy_hp - info['enemy_hp'])
        self.enemy_hp = info['enemy_hp']
        done = terminated or truncated
        for p in self.pending:
            p[0] -= 1
            p[1] += dealt
            if p[0] == 0 or done:
                reward += self.bonus if p[1] > 0 else -self.penalty
        self.pending = [p for p in self.pending if p[0] > 0 and not done]
        return self._observation(obs), reward, terminated, truncated, info


class StateMix(gym.Wrapper):
    """판마다 시작 상태(상대)를 하나 고른다. 최근 승률이 낮은 상대일수록 자주 고른다.
    고를 확률 ∝ (1 − 최근 승률) + floor. 최근 승률은 판이 끝날 때마다 지수 이동 평균으로 갱신한다.
    info['start_state']에 이번 판의 시작 상태를 넣는다."""
    def __init__(self, env, states, seed=None, floor=0.1, decay=0.9):
        super().__init__(env)
        self.states, self.floor, self.decay = states, floor, decay
        self.win_rate = np.zeros(len(states))
        self.rng = np.random.default_rng(seed)
        self.current = 0

    def reset(self, **kwargs):
        p = 1 - self.win_rate + self.floor
        self.current = int(self.rng.choice(len(self.states), p=p / p.sum()))
        self.unwrapped.load_state(self.states[self.current], retro.data.Integrations.ALL)
        obs, info = self.env.reset(**kwargs)
        info['start_state'] = self.states[self.current]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['start_state'] = self.states[self.current]
        if terminated or truncated:
            win = float(info['enemy_hp'] < 0)
            self.win_rate[self.current] = self.decay * self.win_rate[self.current] + (1 - self.decay) * win
        return obs, reward, terminated, truncated, info


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


def make_env(render_mode=None, random_start=0, seed=None, separate=True, bonus=0.0, penalty=0.0, state=STATE, states=None):
    """DQN·PPO가 함께 쓰는 환경. 상태는 흑백 3시점 + 필살기별 쿨타임 채널 3개 = (100, 128, 6), 행동은 20가지.
    random_start를 주면 판마다 처음 0~random_start걸음을 무작위로 움직인 뒤 넘긴다(학습용).
    separate·bonus·penalty는 MoveSet 참고. state는 시작할 저장 상태(integration/의 Champion.Level<스테이지>.RyuVs<상대>).
    states(목록)를 주면 판마다 그중 하나를 고른다(StateMix, 학습용)."""
    # integration/의 data.json·scenario.json·.state가 stable-retro 기본 파일보다 먼저 쓰인다
    if INTEGRATION_DIR not in getattr(retro.data.Integrations, 'CUSTOM_PATHS', []):
        retro.data.Integrations.add_custom_path(INTEGRATION_DIR)
    env = retro.make(
        game=GAME,
        state=states[0] if states else state,
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
    )
    if render_mode == 'human':
        env = PygameViewer(env)  # 속도는 PygameViewer가 60fps로 맞추므로 rendering은 끈다
    env = MoveSet(StreetFighterWrapper(env), separate, bonus, penalty)
    if states:
        env = StateMix(env, states, seed)
    return RandomStart(env, random_start, seed) if random_start else env
