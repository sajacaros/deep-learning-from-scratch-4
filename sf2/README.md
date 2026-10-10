# 스트리트 파이터 2 학습

8장 DQN으로 시작해 9·10장(Actor-Critic, PPO)으로 넓혀갈 실습입니다.
환경은 [linyiLYi/street-fighter-ai](https://github.com/linyiLYi/street-fighter-ai)(PPO, Apache 2.0)를 바탕으로, 흑백 화면과 필살기 행동을 더했습니다. DQN과 PPO가 같은 환경을 씁니다.

|항목 |설정 |
|:--|:--|
|게임 |`StreetFighterIISpecialChampionEdition-Genesis-v0`(stable-retro) |
|시작 지점 |`Champion.Level12.RyuVsBison` — 최고 난이도, 류 vs 바이슨 |
|상태 |원본 200×256을 절반으로 줄인 100×128 **흑백** 3장(최근 9프레임 중 3·6·9번째) + **필살기 쿨타임 채널** = `(100, 128, 4)` |
|한 걸음 |6프레임. 버튼 조합은 내내 누르고, 필살기는 커맨드를 2프레임씩 나눠 누름 |
|보상 |싸우는 중 `3×(준 피해) − (받은 피해)`, 승패가 나면 남은 체력으로 큰 보상·벌점. 전부 ×0.001 |
|종료 |한 라운드가 끝나면 |
|행동 |버튼 조합 17가지 + 필살기 3가지(파동권·승룡권·용권선풍각) = 20가지(`MoveSet`). 필살기는 함께 20걸음 쿨타임 |

## 준비

롬은 저장소에 없습니다. 직접 마련한 **미국판** 롬(SHA1 `a5aad1d108046d9388e33247610dafb4c6516e0b`)이
들어 있는 폴더를 지정해 stable-retro에 등록합니다. `.smd` 형식은 등록할 때 자동으로 변환됩니다.

```bash
$ uv sync
$ uv run python -m stable_retro.import <롬이 있는 폴더>
```

`integration/`에는 참고 프로젝트의 `data.json`(체력 등의 RAM 주소)·`scenario.json`·저장 상태가 들어 있고,
`make_env()`가 stable-retro 기본 파일보다 이것을 먼저 읽습니다.

## 실행

```bash
$ uv run python sf2/dqn.py --lr-end 1e-5 --eval-interval 50000 --out runs/dqn   # 학습(--double, --dueling으로 8.4 확장, --random-start 30으로 무작위 시작)
$ uv run python sf2/evaluate.py runs/dqn --episodes 200                          # 체크포인트마다 승률(ε=0)
$ uv run python sf2/play.py runs/dqn/qnet_best.pt --episodes 3                   # 창으로 보기
$ uv run python sf2/ppo.py --random-start 30 --out runs/ppo                      # PPO(같은 환경, MaskablePPO)
$ uv run python sf2/evaluate.py runs/ppo --episodes 200                          # PPO 체크포인트도 같은 스크립트로
```

- 학습 중 `--save-interval`(기본 10만 걸음)마다 `qnet_<걸음>.pt`를 저장하고, 에피소드별 기록은 `episodes.csv`에 남습니다.
- `--eval-interval`을 주면 그 걸음마다 따로 평가해 가장 좋은 모델을 `qnet_best.pt`로 남깁니다.
- `evaluate.py`는 판마다 처음 0~30걸음을 무작위로 움직여 출발 상황을 흔듭니다(안 그러면 모든 판이 똑같이 흘러감).
- GPU 서버에서 학습하고 `.pt`만 가져와 로컬 CPU에서 `play.py`로 봐도 됩니다.
- 로컬 CPU에서 학습까지 돌리면 약 110걸음/초(1걸음 = 6프레임)였습니다.

## 결과

최고 난이도 바이슨 상대, 무작위 시작 200판 승률입니다.

|모델 |학습 걸음 |승률 |
|:--|--:|--:|
|무작위 행동(DQN 행동 20가지) | |6.5% |
|DQN 4종(dqn·double·dueling·double_dueling), 가장 좋은 체크포인트 |100만 |20~28% |
|DQN 4종, 마지막 모델 |100만 |14.5~17% |
|PPO + 학습 때도 무작위 시작 (이전 환경: 컬러, `MultiBinary(12)`) |1,000만 |86.5% |

- DQN 자세한 내용: [8장 부록](../ch08/README.md#부록-dqn으로-스트리트-파이터-2-학습하기), 그래프 [`eval_dqn_winrate.png`](eval_dqn_winrate.png)
- PPO: [9장 부록](../ch09/README.md#부록-ppo로-스트리트-파이터-2-학습하기), 실험 기록 전체: [`eval_report.md`](eval_report.md)
