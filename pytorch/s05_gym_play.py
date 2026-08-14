"""ch08/s01_gym_play.py 의 Gymnasium 버전.

책이 쓰던 openai/gym 패키지는 유지보수가 끝났고, 후속 프로젝트인
Gymnasium(farama-foundation)이 그 자리를 이어받았다. API는 거의 같지만
reset()과 step()의 반환값이 달라졌다.

    (구) gym                          (신) gymnasium
    state = env.reset()               state, info = env.reset()
    s, r, done, info = env.step(a)    s, r, terminated, truncated, info = env.step(a)
"""
import gymnasium as gym
import numpy as np


if __name__ == '__main__':
    # render_mode='human' 은 실제 창을 띄운다.
    # WSL/서버 등 디스플레이가 없는 환경이라면 'rgb_array' 로 바꾸거나 이 인자를 뺀다.
    env = gym.make('CartPole-v1', render_mode='human')
    state, info = env.reset()
    done = False

    while not done:  # 에피소드가 끝날 때까지 반복
        env.render()                       # 진행 과정 시각화
        action = np.random.choice([0, 1])  # 행동 선택(무작위)
        next_state, reward, terminated, truncated, info = env.step(action)
        # terminated: 막대가 쓰러지는 등 '문제의 규칙상' 끝난 경우
        # truncated : 제한 시간(CartPole-v1은 500걸음)에 걸려 잘린 경우
        done = terminated or truncated
        state = next_state

    env.close()
