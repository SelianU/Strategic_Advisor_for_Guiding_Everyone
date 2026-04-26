# 🕹️ AI ARCADE — SAGE

**SAGE (Strategic Advisor for Guiding Everyone)**  
강화학습 기반 분석과 상용 LLM 코칭을 결합한 레트로 아케이드 게임 플랫폼.  
플레이 후 SAGE가 강화학습 에이전트의 Q-value 분석 결과를 바탕으로, 상용 LLM이 자연어 맞춤 전략 피드백을 제공합니다.

---

## 🎮 게임 목록

| 게임 | 에이전트 | URL |
|---|---|---|
| 👾 Space Invaders | D3QN | `/space-invaders` |
| 🎯 Breakout | D3QN | `/breakout` |
| ⚫ Gomoku 15×15 | PPO | `/gomoku` |

### 👾 Space Invaders
- Atari Space Invaders를 직접 플레이
- **D3QN (Dueling Double DQN)** 모델이 플레이어의 모든 행동을 분석
- 게임 종료 후 Q-value 손실, 행동 일치율, 최악의 선택 TOP-5 등 상세 피드백 제공
- TOP-5는 시간적 다양성 보장 (에피소드 5구간 분할 + 최소 3스텝 간격)
- TOP-5 후보를 선택해 **Human vs Agent 비교 리플레이**와 코칭 피드백 확인 가능
- 비교 리플레이 중 **×¼~×4 배속 전환** 및 **±1초 시크** 컨트롤 제공
- 리플레이 하단에 **가상 키보드(← → SPACE)** 표시: 프레임별 실제 액션에 따라 인간/에이전트 각각 실시간 점등

### 🎯 Breakout
- Atari Breakout을 직접 플레이
- **D3QN** 모델 기반 Q-value 분석 (Space Invaders와 동일한 분석 파이프라인)
- 비교 리플레이, 가상 키보드 시각화, 세션 저장/불러오기 모두 지원

### ⚫ Gomoku vs AI
- **PPO** 기반 AI와 15×15 오목 대결
- 게임 종료 후 각 착수의 Q-value와 최선 착수를 비교 분석
- TOP-5 후보를 선택해 **Human vs Agent 비교 리플레이**와 코칭 피드백 확인 가능
- 비교 리플레이에서 **Q-value 히트맵** 시각화 제공: 백돌 착수 프레임에 흑돌 후보 위치를 색상(파랑→초록→빨강)과 크기로 표시
- 비교 리플레이는 인간/에이전트 화면을 **분리 출력**, 1회 자동재생 후 **◀/▶ 버튼 및 키보드 ←→**로 수동 프레임 탐색 가능

---

## ⚙️ 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/SelianU/Strategic_Advisor_for_Guiding_Everyone.git
cd Strategic_Advisor_for_Guiding_Everyone
```

### 2. 의존성 설치

Python 3.11 권장

```bash
pip install flask flask-socketio eventlet
pip install gymnasium[atari] ale-py
pip install torch torchvision
pip install opencv-python numpy requests
```

### 3. 모델 파일 준비

```
ai_agents/
├── space_invaders/checkpoints/best_model.pth
├── breakout/checkpoints/best_model.pth
└── gomoku/gomoku_rl/pretrained_models/15_15/ppo/0.pt
```

> 모델이 없어도 기본 플레이는 동작하며, 해당 게임의 Q-value 분석만 비활성화됩니다.

### 4. 서버 실행

```bash
python app.py
```

브라우저에서 `http://localhost:5001` 접속

### 5. OpenRouter 연동 (선택)

서버 실행 후 브라우저에서 메인 화면 우상단 **⚙ 설정 버튼**을 눌러 API 키를 등록할 수 있습니다.  
입력한 키는 `config.json`에 저장되어 서버 재시작 후에도 유지됩니다.

```bash
export OPENROUTER_API_KEY='YOUR_KEY'
export OPENROUTER_MODEL='meta-llama/llama-3.3-70b-instruct:free'
```

> 설정하지 않아도 앱은 실행되며, 이 경우 로컬 데이터 기반 코칭 피드백을 사용합니다.

---

## 🧠 AI 모델 소개

### D3QN (Dueling Double Deep Q-Network) — Atari 게임
- **환경**: ALE/SpaceInvaders-v5, ALE/Breakout-v5
- **입력**: 4프레임 스택 (84×84 grayscale), `frameskip=1`
- **아키텍처**: Dueling DQN (Value Stream + Advantage Stream)
- **행동 공간**: Space Invaders 6가지 / Breakout 4가지

### PPO (Proximal Policy Optimization) — Gomoku
- **환경**: 15×15 오목, 5목 승리 조건
- **알고리즘**: PPO (Proximal Policy Optimization)

---

## 🔍 분석 기능

| 항목 | Space Invaders | Breakout | Gomoku |
|---|---|---|---|
| 행동 분포 통계 | ✅ | ✅ | — |
| Q-value 분석 | ✅ (D3QN) | ✅ (D3QN) | ✅ (PPO) |
| 최악의 선택 TOP-5 | ✅ | ✅ | ✅ |
| AI 권장 행동 비교 | ✅ | ✅ | ✅ |
| Human vs Agent 비교 리플레이 | ✅ | ✅ | ✅ |
| 비교 리플레이 배속 컨트롤 | ✅ (×¼~×4) | ✅ (×¼~×4) | — |
| 비교 리플레이 수동 프레임 탐색 | — | — | ✅ (◀/▶) |
| 비교 리플레이 Q-value 히트맵 | — | — | ✅ |
| 리플레이 가상 키보드 시각화 | ✅ | ✅ | — |
| 상용 LLM / 로컬 코칭 피드백 | ✅ | ✅ | ✅ |
| 세션 저장 / 불러오기 | ✅ | ✅ | ✅ |

---

## 🏗️ 프로젝트 구조

```
ai_arcade/
├── app.py                      # 서버 진입점 (Gomoku 핸들러 + 게임 등록)
├── llm_feedback.py             # LLM 코칭 피드백 생성기
├── config.json                 # OpenRouter API 키 설정 (자동 생성)
│
├── games/                      # ── Atari 게임 플러그인 ──────────
│   ├── atari_base.py           #   AtariGame 베이스 클래스 (공통 로직)
│   ├── space_invaders.py       #   SpaceInvadersGame (설정값 + 모델 로더)
│   └── breakout.py             #   BreakoutGame      (설정값 + 모델 로더)
│
├── ai_agents/                  # ── 학습된 모델 ──────────────────
│   ├── space_invaders/checkpoints/best_model.pth
│   ├── breakout/checkpoints/best_model.pth
│   └── gomoku/gomoku_rl/pretrained_models/15_15/ppo/0.pt
│
├── templates/
│   ├── index.html
│   ├── atari_game.html         # Atari 게임 통합 템플릿 (파라미터화)
│   └── gomoku.html
│
└── saved_sessions/             # 세션 자동 저장
    ├── space_invaders/
    ├── breakout/
    └── gomoku/
```

---

## ➕ 새 게임 추가하기

Atari ALE 게임은 파일 하나와 두 줄이면 추가됩니다.

### 1단계: `games/mygame.py` 생성

```python
# games/pong.py
import os
from games.atari_base import AtariGame

class PongGame(AtariGame):
    game_id      = 'pong'
    game_title   = 'PONG'
    game_icon    = '🏓'
    env_name     = 'ALE/Pong-v5'
    prefix       = 'po_'          # 소켓 이벤트 접두사 (유일해야 함)
    theme_color  = '#00f5ff'
    model_path_parts = ('ai_agents', 'pong', 'checkpoints', 'best_model.pth')

    action_names = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}

    keyboard_keys = [             # 가상 키보드 키 목록
        {'id': 'left',  'label': '←',    'actions': [3, 5]},
        {'id': 'right', 'label': '→',    'actions': [2, 4]},
        {'id': 'fire',  'label': 'FIRE', 'actions': [1, 4, 5]},
    ]

    key_combos = {                # 키 조합 → 액션 (키 이름 알파벳순 + 연결)
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
    SpaceInvadersGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    BreakoutGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    PongGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),  # ← 추가
]
```

### 3단계 (선택): `llm_feedback.py`에 피드백 분기 추가

```python
elif game_type == 'pong':
    # build_messages, build_fallback_feedback 구현
    ...
```

베이스 클래스가 자동 처리하는 것: Flask 라우트, 소켓 핸들러 전체, Q-value 분석 백그라운드 태스크, 카운터팩추얼 리플레이, 세션 저장/불러오기, HTML 테마 및 가상 키보드 렌더링.

---

## 🛠️ 기술 스택

- **Backend**: Flask, Flask-SocketIO, eventlet
- **RL 환경**: Gymnasium (ALE/Atari), 자체 구현 (Gomoku)
- **AI 모델**: PyTorch (D3QN, PPO)
- **Frontend**: Vanilla JS, Socket.IO, Canvas API
- **디자인**: Press Start 2P + Share Tech Mono 폰트, 레트로 아케이드 테마

---

## 📝 라이선스
- Team A1