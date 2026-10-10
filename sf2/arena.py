"""학습한 모델 여러 개(최대 4개)를 한 창에 2×2로 띄워 동시에 싸우게 한다.
Play 버튼(또는 스페이스)을 누르면 모든 모델이 한 라운드씩 같이 시작한다.
라운드마다 상대(저장 상태)를 무작위로 고르고, 처음 0~30걸음은 무작위로 움직여 매번 다른 판이 되게 한다.
네 모델에는 같은 상대·같은 무작위 시작을 줘서 같은 조건으로 비교한다.

  uv run python sf2/arena.py runs/dqn/qnet_200000.pt runs/dqn/qnet_700000.pt ...

stable-retro는 한 프로세스에 에뮬레이터를 하나만 띄울 수 있어 모델마다 프로세스를 따로 둔다.
각 프로세스는 매 프레임을 공유 메모리에 써 두고, 이 창은 그것을 읽어 그리기만 한다.
"""
import argparse
import multiprocessing as mp
import os
import time

import gymnasium as gym
import numpy as np

H, W = 200, 256  # 에뮬레이터 원본 화면 크기
BAR = 56         # 위쪽 버튼 줄 높이
LABEL = 28       # 칸마다 위에 붙는 이름 줄 높이
STATES = ['Champion.Level1.RyuVsGuile', 'Champion.Level12.RyuVsBison']  # 고를 수 있는 상대
MAX_RANDOM = 30  # 라운드 시작에 무작위로 움직일 최대 걸음 수


def new_round(rng):
    """이번 라운드의 상대와 무작위 시작용 시드. 모든 모델에 같은 값을 보낸다."""
    return 'play', STATES[rng.integers(len(STATES))], int(rng.integers(2**31))


def state_label(state):
    level, match = state.split('.')[1:3]  # 'Level12', 'RyuVsBison'
    return f'{level.replace("Level", "Level ")} · vs {match.split("Vs")[1]}'


class FrameSink(gym.Wrapper):
    """매 프레임을 공유 메모리에 쓰고, 60fps로 속도를 맞춘다."""
    def __init__(self, env, frame, fps=60):
        super().__init__(env)
        self.frame = frame
        self.dt = 1.0 / fps
        self.next_t = time.perf_counter()

    def _put(self, obs):
        self.frame[:] = obs
        self.next_t += self.dt
        wait = self.next_t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            self.next_t = time.perf_counter()  # 밀렸으면 따라잡으려 하지 않는다

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.next_t = time.perf_counter()
        self._put(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._put(obs)
        return obs, reward, terminated, truncated, info


def worker(model_path, shm, conn, epsilon):
    import torch
    import stable_retro as retro
    from dqn import QNet, mask_specials, random_action as dqn_random_action
    from sf2_env import INTEGRATION_DIR, STATE, GAME, StreetFighterWrapper, MoveSet, action_mask

    torch.set_num_threads(1)  # 네 프로세스가 CPU를 나눠 쓰므로
    frame = np.frombuffer(shm, dtype=np.uint8).reshape(H, W, 3)

    retro.data.Integrations.add_custom_path(INTEGRATION_DIR)
    env = retro.make(game=GAME, state=STATE, inttype=retro.data.Integrations.ALL,
                     use_restricted_actions=retro.Actions.FILTERED,
                     obs_type=retro.Observations.IMAGE, render_mode=None)
    env = FrameSink(env, frame)

    env = MoveSet(StreetFighterWrapper(env))
    random_action = lambda rng, state: dqn_random_action(state, rng)
    if model_path.endswith('.zip'):  # PPO(stable-baselines3) 체크포인트
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path, device='cpu', custom_objects={
            'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0})
        kind = 'PPO'
        act = lambda state: int(model.predict(state, deterministic=False, action_masks=action_mask(state))[0])
    else:
        params = torch.load(model_path, map_location='cpu')
        dueling = any(k.startswith('v.') for k in params)
        qnet = QNet(env.action_space.n, dueling=dueling)
        qnet.load_state_dict(params)
        qnet.eval()
        kind = 'Dueling' if dueling else ''

        def act(state):
            with torch.no_grad():
                x = torch.from_numpy(state).unsqueeze(0)
                return mask_specials(qnet(x), x).argmax().item()

    state, info = env.reset()  # 첫 화면을 띄워 둔다
    conn.send(('ready', kind))
    while True:
        msg = conn.recv()
        if msg == 'quit':
            break
        _, start_state, seed = msg
        env.unwrapped.load_state(start_state, retro.data.Integrations.ALL)
        state, info = env.reset()
        rng = np.random.default_rng(seed)
        done, total_reward = False, 0.0
        for _ in range(rng.integers(0, MAX_RANDOM + 1)):  # 무작위 시작(모든 모델이 같은 행동열)
            state, reward, terminated, truncated, info = env.step(random_action(rng, state))
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        while not done:
            action = env.action_space.sample() if np.random.rand() < epsilon else act(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        conn.send(('done', info['enemy_hp'] < 0, total_reward, info['agent_hp'], info['enemy_hp']))
    env.close()


def label_of(path):
    return f'{os.path.basename(os.path.dirname(path))}/{os.path.splitext(os.path.basename(path))[0]}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='+')
    p.add_argument('--scale', type=float, default=2.0)
    p.add_argument('--epsilon', type=float, default=0.0)
    args = p.parse_args()
    assert len(args.models) <= 4, '모델은 4개까지'

    ctx = mp.get_context('spawn')
    slots = []
    for path in args.models:
        shm = ctx.RawArray('B', H * W * 3)
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=worker, args=(path, shm, child, args.epsilon), daemon=True)
        proc.start()
        slots.append(dict(name=label_of(path), frame=np.frombuffer(shm, dtype=np.uint8).reshape(H, W, 3),
                          conn=parent, proc=proc, ready=False, playing=False,
                          wins=0, losses=0, last='', kind=''))

    import pygame as pg
    pg.init()
    pw, ph = int(W * args.scale), int(H * args.scale)
    cols = 1 if len(slots) == 1 else 2
    rows = (len(slots) + cols - 1) // cols
    screen = pg.display.set_mode((pw * cols, BAR + (LABEL + ph) * rows))
    pg.display.set_caption('Street Fighter II — Arena')
    font = pg.font.SysFont(None, 26)
    big = pg.font.SysFont(None, 34)
    button = pg.Rect(16, 10, 140, BAR - 20)
    clock = pg.time.Clock()
    rng = np.random.default_rng()
    rounds, opponent = 0, ''

    def can_play():
        return all(s['ready'] and not s['playing'] for s in slots)

    def play():
        nonlocal rounds, opponent
        rounds += 1
        msg = new_round(rng)
        opponent = state_label(msg[1])
        for s in slots:
            s['playing'], s['last'] = True, ''
            s['conn'].send(msg)

    running = True
    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_ESCAPE, pg.K_q)):
                running = False
            elif can_play() and ((e.type == pg.MOUSEBUTTONDOWN and e.button == 1 and button.collidepoint(e.pos))
                                 or (e.type == pg.KEYDOWN and e.key == pg.K_SPACE)):
                play()

        for s in slots:
            while s['conn'].poll():
                msg = s['conn'].recv()
                if msg[0] == 'ready':
                    s['ready'], s['kind'] = True, msg[1]
                else:
                    _, win, reward, my_hp, enemy_hp = msg
                    s['playing'] = False
                    s['wins' if win else 'losses'] += 1
                    s['last'] = f'{"WIN" if win else "LOSE"}  reward {reward:.2f}  hp {my_hp} vs {enemy_hp}'

        screen.fill((24, 24, 28))
        ok = can_play()
        pg.draw.rect(screen, (196, 64, 52) if ok else (80, 80, 86), button, border_radius=8)
        txt = big.render('Play' if ok else ('Loading…' if not all(s['ready'] for s in slots) else 'Playing…'),
                         True, (255, 255, 255))
        screen.blit(txt, txt.get_rect(center=button.center))
        screen.blit(font.render(f'round {rounds}  {opponent}   (Space: Play, Esc: quit)', True, (200, 200, 200)),
                    (button.right + 20, BAR // 2 - 9))

        for i, s in enumerate(slots):
            x, y = (i % cols) * pw, BAR + (i // cols) * (LABEL + ph)
            head = f'{s["name"]} {s["kind"]}   W {s["wins"]} / L {s["losses"]}'
            screen.blit(font.render(head, True, (235, 235, 235)), (x + 8, y + 5))
            if s['last']:
                color = (120, 220, 120) if s['last'].startswith('WIN') else (230, 110, 100)
                t = font.render(s['last'], True, color)
                screen.blit(t, (x + pw - t.get_width() - 8, y + 5))
            surf = pg.surfarray.make_surface(s['frame'].swapaxes(0, 1))
            screen.blit(pg.transform.scale(surf, (pw, ph)), (x, y + LABEL))

        pg.display.flip()
        clock.tick(60)

    for s in slots:
        try:
            s['conn'].send('quit')
        except (BrokenPipeError, OSError):
            pass
    pg.quit()
    for s in slots:
        s['proc'].join(timeout=2)
        if s['proc'].is_alive():
            s['proc'].terminate()


if __name__ == '__main__':
    main()
