# 『밑바닥부터 시작하는 딥러닝 ❹』<br>: 이번엔 강화 학습이다!

<a href="http://www.yes24.com/Product/Goods/72173703"><img src="https://github.com/WegraLee/deep-learning-from-scratch-4/blob/master/cover.jpeg" width="150" align=right></a>

**강화 학습 핵심 이론부터 문제 풀이, 심층 강화 학습까지 한 권에!**

이 책의 특징은 제목 그대로 ‘밑바닥부터 만들어가는 것’입니다. 속을 알 수 없는 외부 라이브러리에 의존하지 않고 강화 학습 알고리즘을 처음부터 구현하면서 배웁니다. 그림으로 원리를 이해하고 수학으로 강화 학습 문제를 풀어본 다음, 코드로 구현해 배운 내용을 되짚어봅니다. 코드는 최대한 간결하면서도 강화 학습에서 중요한 아이디어가 명확하게 드러나도록 짰습니다. 단계적으로 수준을 높이면서 다양한 문제에 접할 수 있도록 구성하였으니 강화 학습의 어려움과 재미를 모두 느낄 수 있을 것입니다.


[미리보기](https://preview2.hanbit.co.kr/books/yyxd/#p=1) | [알려진 오류(정오표)](https://docs.google.com/document/d/1fsPVXyPF0gpmN57VV6k0uxMfWXUbiQCwno8vCTYpMc8/edit) | [본문 그림과 수식 이미지 모음](https://github.com/WegraLee/deep-learning-from-scratch-4/blob/master/equations_and_figures_4.zip?raw=true)

---

## 파일 구성

|폴더 이름 |설명                         |
|:--        |:--                          |
|ch01       |1장에서 사용하는 소스 코드 |
|ch02       |2장(마르코프 결정 과정) 보충 예제 ※원서에 없는 추가분 |
|ch03       |3장(벨만 방정식) 보충 예제 ※원서에 없는 추가분 |
|...        |...                          |
|ch09       |9장에서 사용하는 소스 코드    |
|common     |공통으로 사용하는 소스 코드   |
|notebooks  |주피터 노트북 형태의 소스 코드 |
|pytorch    |파이토치용으로 포팅된 소스 코드  |

### 2장·3장 보충 예제 (원서에 없는 추가분)

원서의 2장(MDP)과 3장(벨만 방정식)은 수식 중심이라 소스 코드가 없습니다.
개념을 코드로 직접 확인해볼 수 있도록 실행 가능한 예제를 추가했습니다.
두 장 모두 책 2.4절의 **두 칸짜리 그리드 월드**(`common/two_square_grid.py`)를 소재로 씁니다.

|파일 |대응 절 |내용 |
|:--|:--|:--|
|`ch02/s01_mdp.py` |2.1~2.3 |MDP의 5요소(S, A, p, r, γ)를 출력하고, 정책대로 진행하며 수익 G를 정의 그대로 계산 |
|`ch02/s02_discount.py` |2.3 |할인율 γ에 따라 어떤 정책이 좋아 보이는지가 달라지는 과정 + 그래프 |
|`ch02/s03_policy_bruteforce.py` |2.4 |결정적 정책 4가지를 전수 탐색해 최적 정책을 찾고, 이 방식이 왜 확장되지 않는지 확인 |
|`ch03/s01_bellman_linear.py` |3.1~3.2 |벨만 방정식을 연립일차방정식 `(I − γP)v = r`로 보고 정확해를 구함. 3×4 그리드 월드에서 4장의 반복 계산 결과와 일치함을 검증 |
|`ch03/s02_bellman_iterative.py` |3.1 |같은 문제를 반복 대입으로 풀어, 오차가 매 반복 정확히 γ배로 줄어드는 것(축소 사상)을 확인 |
|`ch03/s03_q_function.py` |3.3 |V↔Q 변환과 Q 버전 벨만 방정식을 수치로 검증하고, Q에서 탐욕 정책을 뽑아봄 |
|`ch03/s04_bellman_optimal.py` |3.4~3.5 |벨만 최적 방정식을 반복 대입·해석적 풀이 두 가지로 풀고, 2장의 전수 탐색 결과와 대조 |

읽는 순서는 `ch02/s01_mdp.py` → `ch02/s02_discount.py` → `ch02/s03_policy_bruteforce.py` →
`ch03/s01_bellman_linear.py` → `ch03/s02_bellman_iterative.py` → `ch03/s03_q_function.py` →
`ch03/s04_bellman_optimal.py`를 권합니다.
2장은 "정책을 전부 만들어보고 고르는" 방식의 한계를 보여주고,
3장은 그 한계를 벨만 방정식이 어떻게 넘어서는지를 보여주도록 이어집니다.

## 주피터 노트북
이 책의 코드는 주피터 노트북으로도 제공됩니다. 다음 표의 링크를 클릭하면 구글과 캐글 같은 클라우드 서비스에서 노트북을 실행할 수 있습니다.

| 장 | Colab | 캐글 | Studio Lab |
| :--- | :--- | :--- | :--- |
| 1장 밴디트 문제| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/01_bandit.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/01_bandit.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/01_bandit.ipynb) |
| 4장 동적 프로그래밍 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/04_dynamic_programming.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/04_dynamic_programming.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/04_dynamic_programming.ipynb) |
| 5장 몬테카를로법 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/05_montecarlo.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/05_montecarlo.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/05_montecarlo.ipynb) |
| 6장 TD법 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/06_temporal_difference.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/06_temporal_difference.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/06_temporal_difference.ipynb) |
| 7장 신경망과 Q 러닝 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/07_neural_networks.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/07_neural_networks.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/07_neural_networks.ipynb) |
| 8장 DQN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/08_dqn.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/08_dqn.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/08_dqn.ipynb) |
| 9장 정책 경사법  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/09_policy_gradient.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/09_policy_gradient.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-4/blob/master/notebooks/09_policy_gradient.ipynb) |


## 개발 환경 설정 (uv)

패키지 관리는 [uv](https://docs.astral.sh/uv/)로 합니다. 저장소 루트에서 다음 한 줄이면 끝입니다.

```bash
$ uv sync                # .venv 생성 + 의존성 설치 (파이썬 3.12는 uv가 알아서 받아옵니다)
$ uv sync --group dev    # 주피터 노트북까지 쓰려면
$ uv sync --group dezero # 7~9장의 DeZero 원본 코드를 돌려보려면
```

기본으로 설치되는 것:

|패키지 |용도 |
|:--|:--|
|numpy, matplotlib |1~6장 전부, 그리고 시각화 |
|torch |`pytorch/` 폴더의 파이토치 구현 |
|gymnasium[classic-control] |8~9장의 카트 폴 환경 |

몇 가지 참고사항:

* **GPU는 필요 없습니다.** 이 책의 신경망은 은닉층 128짜리 MLP 수준이라 CPU가 오히려 빠릅니다.
  그래서 `pyproject.toml`에서 CUDA 휠(2.5GB+) 대신 CPU 전용 휠을 받도록 지정해두었습니다.
  GPU를 쓰고 싶다면 `[[tool.uv.index]]`와 `[tool.uv.sources]`의 `torch` 항목만 지우고 다시 `uv sync` 하세요.
* **DeZero는 기본 설치에서 빠져 있습니다.** 파이토치 구현(`pytorch/` 폴더)이 7~9장을 모두 커버하므로,
  파이토치만 쓴다면 설치할 필요가 없습니다. 책 본문의 원본 코드(`ch07/*.py`, `ch08/s03_dqn.py`, `ch09/*.py`)를
  직접 돌려보고 싶을 때만 `uv sync --group dezero`를 실행하세요.
  참고로 PyPI 버전(0.0.13, 2020년)은 numpy 1.24에서 제거된 `np.int`를 사용해 `import dezero`부터 실패하므로,
  그 부분이 고쳐진 [원저자 저장소](https://github.com/oreilly-japan/deep-learning-from-scratch-3)에서 직접 받도록 설정해두었습니다.
  (기본 의존성에 두지 않은 이유이기도 합니다 — 매번 GitHub에 접속하게 되므로.)
* **gym → gymnasium**: 책이 쓰던 `gym` 패키지는 유지보수가 끝났습니다.
  후속 프로젝트인 Gymnasium을 쓰도록 8~9장의 `import` 한 줄과 환경 이름(`CartPole-v0` → `CartPole-v1`)만 바꿨습니다.

### VS Code

저장소 루트(`deep-learning-from-scratch-4/`)를 VS Code로 여세요. `.vscode/` 설정이 함께 들어 있어

* 인터프리터가 `.venv`로 자동 선택되고,
* 아무 챕터 파일이나 열어 **F5**를 누르면 저장소 루트를 작업 디렉터리로 실행되므로
  `from common.gridworld import GridWorld` 같은 import가 그대로 동작합니다.

## 실행 방법

예제 코드들은 장별로 나눠 저장되어 있습니다. 저장소 루트에서 다음과 같이 실행하세요.

```bash
$ uv run python ch01/s01_avg.py
$ uv run python ch03/s01_bellman_linear.py
$ uv run python pytorch/s07_dqn.py    # 파이토치 구현
```

7~9장의 DeZero 원본 코드는 `--group dezero`가 필요합니다.

```bash
$ uv sync --group dezero
$ uv run --group dezero python ch08/s03_dqn.py
```

`uv run`을 붙이면 가상환경을 따로 활성화하지 않아도 됩니다.
활성화해서 쓰고 싶다면 `source .venv/bin/activate` 후 `python ch01/s01_avg.py`로 실행하면 됩니다.

### 파이토치 구현

`pytorch/` 폴더에 7~9장의 파이토치 버전이 들어 있습니다.
DeZero 원본은 책 대조용으로 그대로 두었지만, 아래 대응표대로 파이토치 버전이 전부 준비되어 있으므로
DeZero를 설치하지 않고도 7~9장을 끝까지 실습할 수 있습니다.

|DeZero 원본 |파이토치 버전 |
|:--|:--|
|`ch07/s01_dezero1.py`, `ch07/s02_dezero2.py` |`pytorch/s01_torch_basics.py` |
|`ch07/s03_dezero3.py` |`pytorch/s02_linear_regression.py` |
|`ch07/s04_dezero4.py` |`pytorch/s03_neural_net.py` |
|`ch07/s05_q_learning_nn.py` |`pytorch/s04_q_learning_nn.py` |
|`ch08/s01_gym_play.py` |`pytorch/s05_gym_play.py` |
|`ch08/s02_replay_buffer.py` |`pytorch/s06_replay_buffer.py` |
|`ch08/s03_dqn.py` |`pytorch/s07_dqn.py` |
|`ch09/s01_simple_pg.py` |`pytorch/s08_simple_pg.py` |
|`ch09/s02_reinforce.py` |`pytorch/s09_reinforce.py` |
|`ch09/s03_actor_critic.py` |`pytorch/s10_actor_critic.py` |

### 그래프 창이 뜨지 않을 때

WSL이나 원격 서버처럼 디스플레이가 없는 환경에서는 `MPLBACKEND=Agg`를 붙여 창 없이 실행할 수 있습니다.

```bash
$ MPLBACKEND=Agg uv run python ch04/s05_policy_iter.py
```

`render_mode='human'`으로 카트 폴 창을 띄우는 코드(`pytorch/s05_gym_play.py` 등)는 디스플레이가 필요합니다.
WSL이라면 WSLg가 켜져 있으면 그대로 동작합니다.

---

## 팬픽 - 바닷속 딥러닝 어드벤처 (5부작)

<img src="https://github.com/WegraLee/deep-learning-from-scratch-5/blob/main/posters/%E1%84%87%E1%85%A1%E1%84%83%E1%85%A1%E1%86%BA%E1%84%89%E1%85%A9%E1%86%A8%20%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%8B%E1%85%A5%E1%84%83%E1%85%B3%E1%84%87%E1%85%A6%E1%86%AB%E1%84%8E%E1%85%A5.png?raw=true">

"<밑바닥부터 시작하는 딥러닝>의 주인공 생선들은 딥러닝 기술로 바닷속 생태계를 어떻게 혁신하고 있을까요? 어공지능의 첨단을 이끌어가는 밑시딥 생선들과 신나는 모험을 떠나보세요."
 
바닷속 세계를 배경으로, 해양 생물들이 자신의 특성과 필요에 맞는 딥러닝 기술을 개발하여 문제를 해결해 나가는 모험을 그린 연작 소설입니다. 시리즈를 읽으신 분은 더 많은 재미를 느끼실 수 있도록 딥러닝 요소들을 곳곳에 삽입하였습니다.

각 편의 주인공과 주제는 다음과 같습니다.

1. **시야를 찾아서**: 쏨뱅이(쏨)가 **이미지 처리 기술**을 개발하여 주변 환경을 선명하게 파악
1. **상어공주**: 괭이상어 공주(꽹)가 **자연어 처리** 기술로 돌고래 왕자와의 사랑을 쟁취
1. **DeZero의 창조자**: 나뭇잎해룡(잎룡)이 **딥러닝 프레임워크**를 만들어 기술 보급과 협업 촉진
1. **제발, 가즈아!**: 가자미(가즈아)가 **심층 강화 학습**으로 먹이가 풍부한 새로운 바다 개척
1. **피쉬카소와 천재의 초상**: 유령실고기(피쉬카소)가 **이미지 생성 모델**로 바닷속 예술계 혁신

<a href="https://www.hanbit.co.kr/channel/series/series_detail_list.html?hcs_idx=34" target="_blank" rel="noopener noreferrer">소설 보러 가기</a>

---

## 라이선스

이 저장소의 소스 코드는 [MIT 라이선스](http://www.opensource.org/licenses/MIT)를 따릅니다.
상업적 목적으로도 자유롭게 이용하실 수 있습니다.
