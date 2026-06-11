# AI ARCADE — SAGE

**SAGE (Strategic Advisor for Guiding Everyone)**  
강화학습 기반 분석과 상용 LLM 코칭을 결합한 레트로 아케이드 게임 플랫폼.  
플레이 후 SAGE가 강화학습 에이전트의 Q-value 분석 결과를 바탕으로, 상용 LLM이 자연어 맞춤 전략 피드백을 제공합니다.

---

## 게임 목록

### Atari 게임 (D3QN 에이전트)

| 게임 | URL |
|---|---|
| Space Invaders | `/space-invaders` |
| Breakout | `/breakout` |
| Enduro | `/enduro` |
| Alien | `/alien` |
| Amidar | `/amidar` |
| Assault | `/assault` |
| Asterix | `/asterix` |
| Asteroids | `/asteroids` |
| Atlantis | `/atlantis` |
| MarioBros | `/mariobros` |

### 보드 게임 (PPO 에이전트)

| 게임 | URL |
|---|---|
| Gomoku 15×15 | `/gomoku` |

---

## 주요 기능

- **게임 플레이 + D3QN/PPO 에이전트 실시간 분석**: 모든 스텝에서 에이전트와 인간의 행동을 비교
- **TOP-5 핵심 순간 선택**: loss 기반 global top-5, 60프레임 간격 보장
- **Human vs Agent 비교 리플레이**: 30초 타이머, ×¼~×4 배속, Grad-CAM 시각화
- **연습 모드**: 핵심 순간부터 직접 플레이, 30초 타이머, AI 일치율 결과 카드
- **도전과제 시스템**: Atari·Gomoku 전 게임 Bronze/Silver/Gold/Platinum 도전과제, 실시간 토스트 알림
- **Rule Book**: 게임별 규칙/조작법/점수 조건
- **LLM 코칭 피드백**: OpenRouter 연동, 자동 폴백 (주 모델 → 예비 대형 → 자동 예비풀)
- **세션 저장/불러오기**: 분석 결과와 counterfactual 캐시 포함 저장
- **Grad-CAM 시각화**: NORMAL / HUMAN CAM / AGENT CAM / SPLIT CAM 전환 지원

---

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/SelianU/Strategic_Advisor_for_Guiding_Everyone.git
cd Strategic_Advisor_for_Guiding_Everyone
```

### 2. 의존성 설치

Python 3.11 권장

```bash
pip install -r requirements.txt
# 학습/분석 도구(training/, experiments/)까지 쓰려면:
pip install -r requirements.txt -r training/requirements.txt
```

### 3. 모델 파일 준비

```
data/checkpoints/
├── space_invaders/best_model_spaceinvaders.pth
├── breakout/best_model.pth
├── enduro/best_model_enduro.pth
├── alien/best_model_alien.pth
├── amidar/best_model.pth
├── assault/best_model_assault.pth
├── asterix/best_model.pth
├── asteroids/best_model.pth
├── atlantis/best_model_atlantis.pth
└── mariobros/best_model.pth

ai_agents/gomoku/gomoku_rl/pretrained_models/15_15/ppo/0.pt   # Gomoku PPO (서브모듈)
```

> 모델이 없어도 기본 플레이는 동작하며, 해당 게임의 Q-value 분석만 비활성화됩니다.

### 4. 서버 실행

```bash
python app.py
```

브라우저에서 `http://localhost:5001` 접속

### 5. OpenRouter 연동 (선택)

서버 실행 후 브라우저에서 메인 화면 우상단 **설정 버튼**을 눌러 API 키를 등록할 수 있습니다.  
입력한 키는 `config.json`에 저장되어 서버 재시작 후에도 유지됩니다.  
(공백이 포함된 플레이스홀더 값 등 유효하지 않은 형식의 키는 무시되며, LLM 상태가 OFF로 표시됩니다.)

```bash
export OPENROUTER_API_KEY='YOUR_KEY'
export OPENROUTER_MODEL='meta-llama/llama-3.3-70b-instruct:free'
```

모델은 **① 주 모델 → ② 예비 대형 → ③ 자동 예비풀** 순으로 폴백하며, LLM STATUS 패널에서 현재 선택된 모델을 확인할 수 있습니다.

> 설정하지 않아도 앱은 실행되며, 이 경우 로컬 데이터 기반 코칭 피드백을 사용합니다.

---

## AI 모델 소개

### D3QN (Dueling Double Deep Q-Network) — Atari 게임
- **환경**: ALE/SpaceInvaders-v5, ALE/Breakout-v5, ALE/Enduro-v5 등
- **입력**: 4프레임 스택 (84×84 grayscale)
- **아키텍처**: Dueling DQN (Value Stream + Advantage Stream)
- **Frame Skip**: 인간 액션 주기와 유사한 수준으로 설정해 코칭 비교의 타당성 확보
  - Space Invaders: `frameskip=15` 적용 완료
  - 나머지 게임: `frameskip=20` 으로 순차적으로 학습 중
  - 논문 기준(frameskip=4) 대비 높은 수치임에도 에이전트 성능이 인간 평균을 상회해 코칭 정당성 유지

### PPO (Proximal Policy Optimization) — Gomoku
- **환경**: 15×15 오목, 5목 승리 조건
- **알고리즘**: PPO (Proximal Policy Optimization)

---

## 분석 기능

| 항목 | Atari 게임 | Gomoku |
|---|---|---|
| 행동 분포 통계 | ✅ | — |
| Q-value 분석 | ✅ (D3QN) | ✅ (PPO) |
| 최악의 선택 TOP-5 | ✅ | ✅ |
| AI 권장 행동 비교 | ✅ | ✅ |
| Human vs Agent 비교 리플레이 | ✅ | ✅ |
| 비교 리플레이 배속 컨트롤 | ✅ (×¼~×4) | — |
| Grad-CAM 시각화 | ✅ | — |
| 연습 모드 | ✅ | — |
| 도전과제 시스템 | ✅ | ✅ |
| 상용 LLM / 로컬 코칭 피드백 | ✅ | ✅ |
| 세션 저장 / 불러오기 | ✅ | ✅ |

---

## 프로젝트 구조

```
SAGE/
├── app.py                          # 서버 진입점 (eventlet patch + 등록 트리거 + run)
├── extensions.py                   # Flask/SocketIO 공유 인스턴스 + DATA_DIR 등 경로
├── atari_registry.py               # Atari 게임 임포트 + ATARI_GAMES 리스트
├── routes.py                       # Flask 라우트 + 설정 API + LLM 테스트 소켓
├── config_store.py                 # config.json 영속화 + API 키 검증
├── llm_feedback.py                 # LLM 코칭 피드백 생성기
├── config.json                     # OpenRouter API 키 설정 (자동 생성)
│
├── games/                          # ── 게임 플러그인 ──────────────────
│   ├── atari/                      #   AtariGame 공통 로직
│   │   ├── base.py                 #     베이스 클래스 (게임 루프 + 라우트/핸들러 등록)
│   │   ├── achievements.py         #     선언적 도전과제 스펙 엔진
│   │   ├── ach_helper.py           #     도전과제 판정 공통 함수
│   │   ├── analysis.py / counterfactual.py / practice.py / sessions.py
│   │   └── preprocessing.py / gradcam.py
│   ├── space_invaders.py           #   SpaceInvadersGame
│   ├── breakout.py                 #   BreakoutGame
│   ├── enduro.py                   #   EnduroGame
│   ├── alien.py / amidar.py / ...  #   나머지 Atari 게임들
│   └── gomoku/                     #   Gomoku 패키지
│       ├── handlers.py             #     SocketIO 핸들러 (게임/세션/분석/counterfactual)
│       ├── state.py                #     게임 상태 싱글턴 + PPO 모델
│       ├── engine.py               #     순수 로직 (Q-value / 승률 / 승리선)
│       ├── render.py               #     보드 시각화
│       ├── sessions.py             #     세션 직렬화 / 디스크 영속화
│       └── game_info.py            #     Rule Book 정적 메타데이터
│
├── ai_agents/                      # ── 모델 로딩 헬퍼 ────────────────
│   ├── d3qn_helper.py              #   D3QN 공통 로더 + GAME_CONFIGS
│   ├── _compat.py                  #   게임별 하위 호환 shim 공통 팩토리
│   ├── <game>/__init__.py          #   하위 호환 shim (d3qn_helper 위임)
│   ├── <game>/coach_config.py      #   게임별 LLM 코칭 설정
│   └── gomoku/gomoku_rl/           #   Gomoku PPO (서브모듈, pretrained 포함)
│
├── data/                           # ── 런타임 산출물 ─────────────────
│   ├── checkpoints/<game>/         #   학습된 모델 (git 추적)
│   ├── saved_sessions/             #   세션 자동 저장 (git 미추적)
│   └── training_data/              #   플레이 기록 pkl (git 미추적)
│
├── training/                       # ── 학습·분석 오프라인 도구 ───────
│   ├── requirements.txt            #   학습 전용 추가 의존성
│   └── space_invaders/             #   D3QN/SAC 학습, Grad-CAM, 플레이 기록 등
│
├── experiments/                    # ── 학습 실험 ─────────────────────
│   ├── d3qn/                       #   D3QN 모듈화 구현 (agent/buffer/env/runner)
│   └── top5_sac_feedback/          #   TOP-5 피드백 증류 실험
│
├── teacher_agent/                  # ── SAC 티처 에이전트 ─────────────
│
├── templates/
│   ├── index.html
│   ├── atari_game.html             # Atari 게임 통합 템플릿
│   └── gomoku.html
│
├── static/css/
│   ├── common.css                  # 게임 페이지 공통 스타일
│   ├── atari_game.css / gomoku.css # 페이지 전용 스타일
│   └── index.css                   # 로비 전용 스타일
│
└── static/js/
    ├── common_utils.js             # 공통 유틸 (소켓/API 래퍼/도전과제 모달 등)
    ├── atari_replay.js             # 비교 리플레이, Grad-CAM, 타이머
    ├── atari_practice.js           # 연습 모드 전체
    ├── atari_achievements.js       # 도전과제 시스템
    └── atari_game.js               # core (캔버스, 키 입력, 게임 루프, 세션, UI 탭)
```

---

## 새 게임 추가하기

Atari ALE 게임은 파일 하나와 두 줄이면 추가됩니다.

### 1단계: `games/mygame.py` 생성

```python
# games/pong.py
from games.atari import AtariGame

class PongGame(AtariGame):
    game_id      = 'pong'
    game_title   = 'PONG'
    game_icon    = '🏓'
    env_name     = 'ALE/Pong-v5'
    prefix       = 'po_'          # 소켓 이벤트 접두사 (유일해야 함)
    theme_color  = '#00f5ff'
    model_path_parts = ('data', 'checkpoints', 'pong', 'best_model.pth')

    action_names = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}

    keyboard_keys = [
        {'id': 'left',  'label': '←',    'actions': [3, 5]},
        {'id': 'right', 'label': '→',    'actions': [2, 4]},
        {'id': 'fire',  'label': 'FIRE', 'actions': [1, 4, 5]},
    ]

    key_combos = {
        'fire+left': 5, 'fire+right': 4,
        'left': 3, 'right': 2, 'fire': 1, '': 0,
    }

    # 도전과제: 선언적 스펙만 정의하면 목록·실시간·사후 판정 자동 처리
    achievement_specs = [
        {'id': 'score_5', 'title': '첫 득점', 'tier': 'bronze', 'desc': '5점 달성',
         'metric': 'score', 'value': 5},
    ]
```

D3QN 모델이라면 `_load_model` / `_get_q_values` 구현이 필요 없습니다 —
베이스가 `ai_agents/d3qn_helper`로 자동 위임합니다.
(단, `ai_agents/d3qn_helper.py`의 `GAME_CONFIGS`에 게임 설정을 등록해야 합니다.)

### 2단계: `atari_registry.py`에 두 줄 추가

```python
from games.pong import PongGame

ATARI_GAMES = [
    ...,
    PongGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),  # ← 추가
]
```

베이스 클래스가 자동 처리하는 것: Flask 라우트, 소켓 핸들러 전체, 모델 로드/Q-value 위임, 도전과제 판정(achievement_specs), Q-value 분석 백그라운드 태스크, counterfactual 리플레이, 연습 모드, 세션 저장/불러오기, HTML 테마 및 가상 키보드 렌더링.

### 도전과제 스펙 메트릭

`achievement_specs`에서 사용할 수 있는 metric 종류 (`games/atari/achievements.py` 참조):

| metric | 의미 | 추가 파라미터 |
|---|---|---|
| `score` | 누적 점수 도달 (`cmp: '>'`로 초과 판정) | — |
| `combo` | window 스텝 내 보상 이벤트 n회 | `window` |
| `streak` | 간격 gap 이내 연속 보상 n회 (목숨 손실 시 초기화) | `gap` |
| `survive` | n스텝 생존 | — |
| `no_death` | 목숨 잃지 않고 n스텝 생존 | — |
| `reward_count` | 특정 보상 이벤트 n회 | `reward_exact` 또는 `reward_min` |

라운드 클리어 감지 등 RAM 기반 커스텀 판정이 필요하면 스펙 대신
`_check_realtime_achievements` / `_compute_achievements`를 직접 구현합니다 (예: `games/space_invaders.py`).

---

## 학습 도구 (training/)

서버 런타임과 분리된 오프라인 학습·분석 스크립트입니다.

```bash
pip install -r requirements.txt -r training/requirements.txt

# D3QN 학습 (Space Invaders)
python training/space_invaders/D3QN_v3.py

# Discrete SAC 학습
python training/space_invaders/sac_atari_v2.py

# 사람 플레이 기록 → 피드백 분석 리포트
python training/space_invaders/human_play_recorder.py
python training/space_invaders/feedback_analyzer.py --data ./human_data/play_001.pkl
```

학습된 모델을 `data/checkpoints/<game>/`에 배치하면 서버가 자동으로 로드합니다.

---

## 기술 스택

- **Backend**: Flask, Flask-SocketIO, eventlet
- **RL 환경**: Gymnasium (ALE/Atari), gomoku_rl 서브모듈 (Gomoku)
- **AI 모델**: PyTorch (D3QN, PPO)
- **Frontend**: Vanilla JS, Socket.IO, Canvas API
- **디자인**: Press Start 2P + Share Tech Mono 폰트, 레트로 아케이드 테마

---

## 라이선스
- Team A1
