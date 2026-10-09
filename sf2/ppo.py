"""PPO로 스트리트 파이터 2 학습하기(10장).

참고 프로젝트 linyiLYi/street-fighter-ai의 train.py와 같은 설정이다.
  - stable-baselines3의 PPO + CnnPolicy, 환경 16개를 프로세스마다 하나씩
  - 행동은 원본 그대로 MultiBinary(12)(버튼마다 누를지 따로 정함)
  - n_steps 512, batch 512, epoch 4, γ 0.94
  - 학습률 2.5e-4 → 2.5e-6, 클리핑 범위 0.15 → 0.025로 직선으로 줄임
--random-start를 주면 판마다 처음 0~N걸음을 무작위로 움직인 뒤 학습을 시작한다.
에뮬레이터가 결정적이라 이것이 없으면 매 판이 같은 상황에서 시작해, 이기는 행동 순서 하나만 외운다.

  uv run python sf2/ppo.py --steps 10000000 --device cuda --out runs/ppo
  uv run python sf2/ppo.py --random-start 30 --out runs/ppo_rs
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from sf2_env import make_env


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
                action = self.rng.integers(0, 2, self.action_space.n).astype(np.int8)
                obs, _, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    break
            else:
                return obs, info


def env_fn(seed, random_start):
    def _init():
        env = make_env(discrete=False)
        if random_start:
            env = RandomStart(env, random_start, seed)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init


def linear_schedule(start, end):
    # stable-baselines3는 남은 진행률(1 → 0)을 넘겨준다
    return lambda progress: end + (start - end) * progress


class EpisodeLogger(BaseCallback):
    """판이 끝날 때마다 보상·승패를 episodes.csv에 남긴다(dqn.py와 같은 형식)."""
    def __init__(self, out, num_envs):
        super().__init__()
        self.log = open(os.path.join(out, 'episodes.csv'), 'w')
        self.log.write('episode,step,reward,win,length\n')
        self.ep_reward = np.zeros(num_envs)
        self.ep_len = np.zeros(num_envs, dtype=int)
        self.episode = 0
        self.recent = []
        self.t0 = time.time()

    def _on_step(self):
        self.ep_reward += self.locals['rewards']
        self.ep_len += 1
        for i, done in enumerate(self.locals['dones']):
            if not done:
                continue
            win = int(self.locals['infos'][i]['enemy_hp'] < 0)
            self.recent = (self.recent + [win])[-100:]
            self.log.write(f'{self.episode},{self.num_timesteps},{self.ep_reward[i]:.4f},{win},{self.ep_len[i]}\n')
            if self.episode % 50 == 0:
                sps = self.num_timesteps / (time.time() - self.t0)
                print(f'episode {self.episode:6d} | step {self.num_timesteps:9d} | reward {self.ep_reward[i]:7.3f} | '
                      f'win rate(100) {np.mean(self.recent):.2f} | {sps:.0f} steps/s', flush=True)
            self.episode += 1
            self.ep_reward[i], self.ep_len[i] = 0.0, 0
        return True

    def _on_training_end(self):
        self.log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=10_000_000)        # 총 걸음 수(모든 환경 합계, 1걸음 = 6프레임)
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--save-interval', type=int, default=500_000)   # 모든 환경 합계 기준
    p.add_argument('--random-start', type=int, default=0)          # 판마다 처음 0~N걸음 무작위(0이면 안 함)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='runs/ppo')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = SubprocVecEnv([env_fn(i, args.random_start) for i in range(args.num_envs)])
    model = PPO(
        'CnnPolicy', env,
        device=args.device,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=linear_schedule(2.5e-4, 2.5e-6),
        clip_range=linear_schedule(0.15, 0.025),
        verbose=0,
    )
    print(f'{vars(args)}', flush=True)
    callbacks = [
        CheckpointCallback(save_freq=args.save_interval // args.num_envs, save_path=args.out, name_prefix='ppo'),
        EpisodeLogger(args.out, args.num_envs),
    ]
    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(os.path.join(args.out, 'ppo_final'))
    env.close()


if __name__ == '__main__':
    main()
