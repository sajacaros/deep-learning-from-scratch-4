"""1인용 모드를 학습한 모델에게 진행시키며, 상대가 바뀔 때마다 라운드 시작 상태를 저장한다.
모델이 --assist-after번 다시 해도 못 이기는 상대는 RAM의 상대 체력을 1로 낮춰 넘긴다(저장되는 다음 상대에는 영향 없음).

  uv run python sf2/make_states.py runs/ppo_cd/ppo_final.zip

가일전(stable-retro 기본 상태, 1스테이지)에서 시작한다. 경기(3판 2선승)를 지면 그 경기의 시작 상태로 되돌려 다시 한다.
저장 위치는 integration/ 아래 Champion.Level<스테이지>.RyuVs<상대>.state(stable-retro와 같은 gzip 형식).
SF2는 뒤 스테이지일수록 CPU가 강해지므로, 상대뿐 아니라 난이도도 스테이지마다 다르다.
"""
import argparse
import gzip
import os

import numpy as np

from sf2_env import GAME, INTEGRATION_DIR, MoveSet, StreetFighterWrapper, action_mask

# RAM의 enemy_character 값 → 이름(가일=3, 바이슨=8은 기존 저장 상태로 확인)
CHARACTERS = ['Ryu', 'Honda', 'Blanka', 'Guile', 'Ken', 'ChunLi', 'Zangief', 'Dhalsim',
              'Bison', 'Sagat', 'Balrog', 'Vega']
FULL_HP = 176


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--max-steps', type=int, default=200_000)  # 전체 걸음 상한
    p.add_argument('--max-fight-steps', type=int, default=3_000)  # 경기 하나가 이보다 길면 되돌린다
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--start', default='Champion.Level1.RyuVsGuile')  # 이어서 만들려면 저장해 둔 상태 이름
    p.add_argument('--stage', type=int, default=1)                    # --start의 스테이지 번호
    p.add_argument('--assist-after', type=int, default=5)  # 이만큼 다시 해도 못 이기면 상대 체력을 1로 낮춰 넘긴다(상태 만들기 전용)
    args = p.parse_args()

    import stable_retro as retro
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    retro.data.Integrations.add_custom_path(INTEGRATION_DIR)
    raw = retro.make(game=GAME, state=args.start, inttype=retro.data.Integrations.ALL,
                     use_restricted_actions=retro.Actions.FILTERED, obs_type=retro.Observations.IMAGE, render_mode=None)
    env = MoveSet(StreetFighterWrapper(raw, reset_round=False))  # 라운드가 끝나도 판을 끝내지 않는다
    model = MaskablePPO.load(args.model, device='cpu', custom_objects={
        'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0})
    model.set_random_seed(args.seed)
    out_dir = os.path.join(INTEGRATION_DIR, GAME)

    env.reset()
    start = raw.em.get_state()
    state, _, _, _, info = env.step(0)  # reset 직후 info에는 RAM 값이 없어 한 걸음 진행해 상대를 읽는다
    stage, current = args.stage, info['enemy_character']
    wins = losses = fight_steps = retries = 0
    prev_agent, prev_enemy = FULL_HP, FULL_HP
    for step in range(args.max_steps):
        if retries >= args.assist_after and 0 < raw.data.lookup_value('enemy_hp') <= FULL_HP:
            raw.data.set_value('enemy_hp', 1)  # 다음 상대가 나오면(저장 직후) retries가 0이 되어 멈춘다
        action = int(model.predict(state, action_masks=action_mask(state))[0])
        state, _, _, _, info = env.step(action)
        fight_steps += 1
        agent, enemy, who = info['agent_hp'], info['enemy_hp'], info['enemy_character']

        # 라운드 결과: 체력이 0 아래로 내려가는 순간을 센다
        if enemy < 0 <= prev_enemy:
            wins += 1
        if agent < 0 <= prev_agent:
            losses += 1
        prev_agent, prev_enemy = agent, enemy

        if losses >= 2 or fight_steps > args.max_fight_steps:  # 경기를 졌으면 그 경기 시작으로 되돌린다
            raw.em.set_state(start)  # 쌓아 둔 화면은 다음 몇 걸음 안에 새 화면으로 바뀐다
            retries += 1
            why = 'too long' if losses < 2 else 'lost'
            print(f'stage {stage} {CHARACTERS[current]}: 다시 ({retries}, {why}, 라운드 {wins}승 {losses}패, {fight_steps}걸음)', flush=True)
            wins = losses = fight_steps = 0
            prev_agent, prev_enemy = FULL_HP, FULL_HP
            continue

        # 새 상대의 첫 라운드가 시작되면 저장한다
        # (두 저장 상태 모두 라운드 시작에 류 x=205, 상대 x=307이었다)
        if (who != current and 0 <= who < len(CHARACTERS) and agent == FULL_HP and enemy == FULL_HP
                and info['agent_x'] == 205 and info['enemy_x'] == 307):
            stage += 1
            current = who
            start = raw.em.get_state()
            name = f'Champion.Level{stage}.RyuVs{CHARACTERS[who]}.state'
            if os.path.exists(os.path.join(out_dir, name)):  # 이미 있는 상태(예: 학습에 쓰던 바이슨전)는 덮어쓰지 않는다
                name = name.replace('.state', '.Ladder.state')
            with gzip.open(os.path.join(out_dir, name), 'wb') as f:
                f.write(start)
            print(f'step {step}: saved {name} (재시도 {retries}번)', flush=True)
            wins = losses = fight_steps = retries = 0
            if who == 8:  # 바이슨이 마지막
                break
    env.close()


if __name__ == '__main__':
    main()
