# 스트리트 파이터 2 학습

8장 DQN으로 시작해 9·10장(Actor-Critic, PPO)으로 넓혀갈 실습입니다.
환경은 [linyiLYi/street-fighter-ai](https://github.com/linyiLYi/street-fighter-ai)(PPO, Apache 2.0)와 같게 맞췄습니다.

|항목 |설정 |
|:--|:--|
|게임 |`StreetFighterIISpecialChampionEdition-Genesis-v0`(stable-retro) |
|시작 지점 |`Champion.Level12.RyuVsBison` — 최고 난이도, 류 vs 베가 |
|상태 |원본 200×256을 절반으로 줄인 **100×128 컬러**. 최근 9프레임 중 2·5·8번째의 R·G·B 채널을 한 장으로 합침 |
|한 걸음 |같은 행동을 6프레임 동안 누름 |
|보상 |싸우는 중 `3×(준 피해) − (받은 피해)`, 승패가 나면 남은 체력으로 큰 보상·벌점. 전부 ×0.001 |
|종료 |한 라운드가 끝나면 |
|행동 |DQN: 버튼 조합 17가지(`Discretizer`) / PPO: 원본처럼 `MultiBinary(12)`(`make_env(discrete=False)`) |

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
$ uv run python sf2/dqn.py --steps 1000000 --out runs/dqn          # 학습
$ uv run python sf2/play.py runs/dqn/qnet_final.pt --episodes 3    # 창으로 보기
```

- 학습 중 `--save-interval`(기본 10만 걸음)마다 `qnet_<걸음>.pt`를 저장하고, 에피소드별 기록은 `episodes.csv`에 남습니다.
- GPU 서버에서 학습하고 `.pt`만 가져와 로컬 CPU에서 `play.py`로 봐도 됩니다.
- 로컬 CPU에서 학습까지 돌리면 약 110걸음/초(1걸음 = 6프레임)였습니다.
