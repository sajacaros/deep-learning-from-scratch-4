"""학습한 DQN이 싸우는 모습을 창으로 보기.

  uv run python sf2/play.py runs/dueling/qnet_best.pt --episodes 3
"""
import argparse

import numpy as np
import torch

from dqn import load_qnet, mask_specials, random_action
from sf2_env import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--epsilon', type=float, default=0.0)  # 같은 동작에 갇히면 0.02 정도로 올려본다
    args = p.parse_args()

    qnet = load_qnet(args.model)
    env = make_env(render_mode='human', separate=qnet.conv[0].in_channels > 4)  # 모델이 본 것과 같은 쿨타임 방식

    rng = np.random.default_rng()
    for episode in range(args.episodes):
        state, info = env.reset()
        done, total_reward = False, 0.0
        while not done:
            if np.random.rand() < args.epsilon:
                action = random_action(state, rng)
            else:
                with torch.no_grad():
                    x = torch.from_numpy(state).unsqueeze(0)
                    action = mask_specials(qnet(x), x).argmax().item()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        result = 'WIN' if info['enemy_hp'] < 0 else 'LOSE'
        print(f'episode {episode}: {result}, reward {total_reward:.3f}, '
              f'my hp {info["agent_hp"]}, enemy hp {info["enemy_hp"]}')
    env.close()


if __name__ == '__main__':
    main()
