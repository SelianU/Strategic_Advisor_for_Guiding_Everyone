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
```

### 3. 모델 파일 준비

```
ai_agents/
├── space_invaders/checkpoints/best_model_spaceinvaders.pth
├── breakout/checkpoints/best_model.pth
├── enduro/checkpoints/best_model_enduro.pth
├── alien/checkpoints/best_model_alien.pth
├── amidar/checkpoints/best_model.pth
├── assault/checkpoints/best_model_assault.pth
├── asterix/checkpoints/best_model.pth
├── asteroids/checkpoints/best_model.pth
├── atlantis/checkpoints/best_model_atlantis.pth
├── mariobros/checkpoints/best_model.pth
└── gomoku/gomoku_rl/pretrained_models/15_15/ppo/0.pt
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
- **Frame Skip**: 게임마다 `frameskip=20` 적용 (현재 학습 완료 또는 진행 중)
  - 인간 액션 주기와 유사한 수준으로 설정해 코칭 비교의 타당성 확보
  - 논문 기준(frameskip=4) 대비 높은 수치임에도 에이전트 성능이 인간 평균을 상회해 코칭 정당성 유지
  - Space Invaders는 frameskip=20 학습 모델 적용 완료, 나머지 게임은 순차적으로 전환 중

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
├── app.py                          # 서버 진입점 (라우트 + Atari 게임 등록)
├── extensions.py                   # Flask/SocketIO 공유 인스턴스
├── llm_feedback.py                 # LLM 코칭 피드백 생성기
├── config.json                     # OpenRouter API 키 설정 (자동 생성)
│
├── games/                          # ── 게임 플러그인 ──────────────────
│   ├── atari/                      #   AtariGame 공통 로직 (분석/counterfactual/연습 등)
│   ├── space_invaders.py           #   SpaceInvadersGame
│   ├── breakout.py                 #   BreakoutGame
│   ├── enduro.py                   #   EnduroGame
│   ├── alien.py / amidar.py / ...  #   나머지 Atari 게임들
│   └── gomoku/                     #   Gomoku 패키지
│       ├── handlers.py             #     SocketIO 핸들러 (게임/세션/분석/counterfactual)
│       ├── state.py                #     게임 상태 싱글턴 + PPO 모델
│       ├── engine.py               #     순수 로직 (Q-value / 승률 / 승리선)
│       ├── render.py               #     보드 시각화
│       └── sessions.py             #     세션 직렬화 / 디스크 영속화
│
├── ai_agents/                      # ── 학습된 모델 ──────────────────
│   ├── space_invaders/checkpoints/best_model_spaceinvaders.pth
│   ├── breakout/checkpoints/best_model.pth
│   ├── enduro/checkpoints/best_model_enduro.pth
│   ├── alien/checkpoints/best_model_alien.pth
│   ├── amidar/checkpoints/best_model.pth
│   ├── assault/checkpoints/best_model_assault.pth
│   ├── asterix/checkpoints/best_model.pth
│   ├── asteroids/checkpoints/best_model.pth
│   ├── atlantis/checkpoints/best_model_atlantis.pth
│   ├── mariobros/checkpoints/best_model.pth
│   └── gomoku/gomoku_rl/pretrained_models/15_15/ppo/0.pt
│
├── templates/
│   ├── index.html
│   ├── atari_game.html             # Atari 게임 통합 템플릿
│   └── gomoku.html
│
├── static/js/
│   ├── atari_replay.js             # 소켓 초기화, 비교 리플레이, Grad-CAM, 타이머
│   ├── atari_practice.js           # 연습 모드 전체
│   ├── atari_achievements.js       # 도전과제 시스템
│   └── atari_game.js               # core (캔버스, 키 입력, 게임 루프, 세션, UI 탭)
│
└── saved_sessions/                 # 세션 자동 저장
    ├── space_invaders/
    ├── breakout/
    └── gomoku/
```

---

## 새 게임 추가하기

Atari ALE 게임은 파일 하나와 두 줄이면 추가됩니다.

### 1단계: `games/mygame.py` 생성

```python
# games/pong.py
import os
from games.atari.base import AtariGame

class PongGame(AtariGame):
    game_id      = 'pong'
    game_title   = 'PONG'
    game_icon    = '🏓'
    env_name     = 'ALE/Pong-v5'
    prefix       = 'po_'          # 소켓 이벤트 접두사 (유일해야 함)
    theme_color  = '#00f5ff'
    model_path_parts = ('ai_agents', 'pong', 'checkpoints', 'best_model.pth')

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

    def _load_model(self, path):
        if not os.path.exists(path): return None
        from ai_agents.pong import load_d3qn
        net, _ = load_d3qn(path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.pong import get_q_values
        return get_q_values(self.net, stacked_state, self.device)
```

### 2단계: `app.py`에 두 줄 추가

```python
from games.pong import PongGame

ATARI_GAMES = [
    ...,
    PongGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),  # ← 추가
]
```

베이스 클래스가 자동 처리하는 것: Flask 라우트, 소켓 핸들러 전체, Q-value 분석 백그라운드 태스크, counterfactual 리플레이, 연습 모드, 세션 저장/불러오기, HTML 테마 및 가상 키보드 렌더링.

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
