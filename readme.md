# 🕹️ AI ARCADE

강화학습 AI와 함께하는 레트로 아케이드 게임 플랫폼.
플레이 후 AI가 당신의 행동을 Q-value 기반으로 분석해줍니다.

---

## 🎮 게임 목록

### 👾 Space Invaders
- Atari Space Invaders를 직접 플레이
- **D3QN (Dueling Double DQN)** 모델이 플레이어의 모든 행동을 분석
- 게임 종료 후 Q-value 손실, 행동 일치율, 최악의 선택 TOP-5 등 상세 피드백 제공
- TOP-5는 시간적 다양성 보장 (에피소드 5구간 분할 + 최소 3스텝 간격)
- TOP-5 후보를 선택해 **Human vs Agent 비교 리플레이**와 코칭 피드백 확인 가능
- 비교 리플레이 중 **×¼~×4 배속 전환** 및 **±1초 시크** 컨트롤 제공
- 리플레이 하단에 **가상 키보드(← → SPACE)** 표시: 프레임별 실제 액션에 따라 인간/에이전트 각각 실시간 점등

### ⚫ Gomoku vs AI
- **AlphaZero** 기반 AI와 8×8 오목 대결
- 게임 종료 후 각 착수의 Q-value와 최선 착수를 비교 분석
- TOP-5 후보를 선택해 **Human vs Agent 비교 리플레이**와 코칭 피드백 확인 가능
- 비교 리플레이에서 **Q-value 히트맵** 시각화 제공: 백돌 착수 프레임에 흑돌 후보 위치를 색상(파랑→초록→빨강)과 크기로 표시
- 비교 리플레이는 인간/에이전트 화면을 **분리 출력**, 1회 자동재생 후 **◀/▶ 버튼 및 키보드 ←→**로 수동 프레임 탐색 가능

---

## ⚙️ 설치 및 실행

### 1. 저장소 클론 (서브모듈 포함)

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/ai-arcade.git
cd ai-arcade
```

이미 클론했다면:

```bash
git submodule update --init --recursive
```

### 2. 의존성 설치

Python 3.11 권장

```bash
pip install flask flask-socketio eventlet
pip install gymnasium[atari] ale-py
pip install torch torchvision
pip install opencv-python numpy requests
```

### 3. D3QN 모델 준비

학습된 모델 파일(`best_model.pth`)을 `checkpoints_v3_logs/` 폴더에 넣어주세요.

```bash
mkdir checkpoints_v3_logs
# best_model.pth 파일을 해당 폴더에 복사
```

> 모델이 없어도 기본 플레이 분석(행동 통계)은 동작하며, D3QN Q-value 분석만 비활성화됩니다.

### 4. Gomoku 모델 준비

`AlphaZero_Gomoku/best_policy_8_8_5.model` 파일이 있어야 합니다.
원본 저장소에서 다운로드하거나 직접 학습할 수 있습니다.

### 5. 서버 실행

```bash
python app.py
```

브라우저에서 `http://localhost:5001` 접속

### 6. OpenRouter 연동 (선택)

서버 실행 후 브라우저에서 메인 화면 우상단 **⚙ 설정 버튼**을 눌러 API 키를 등록할 수 있습니다.
입력한 키는 `config.json`에 저장되어 서버 재시작 후에도 유지됩니다.

터미널에서 직접 설정하는 방법도 동일하게 지원합니다.

```bash
export OPENROUTER_API_KEY='YOUR_KEY'
export OPENROUTER_MODEL='meta-llama/llama-3.3-70b-instruct:free'
```

> 설정하지 않아도 앱은 실행되며, 이 경우 로컬 데이터 기반 코칭 피드백을 사용합니다.

---

## 🧠 AI 모델 소개

### D3QN (Dueling Double Deep Q-Network)
- **환경**: ALE/SpaceInvaders-v5
- **입력**: 4프레임 스택 (84×84 grayscale)
- **아키텍처**: Dueling DQN (Value Stream + Advantage Stream)
- **행동 공간**: 6가지 (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)

### AlphaZero (Gomoku)
- **환경**: 8×8 오목, 5목 승리 조건
- **알고리즘**: MCTS + Policy-Value Network
- **출처**: [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)

---

## 🔍 분석 기능

| 항목 | Space Invaders | Gomoku |
|------|---------------|--------|
| 행동 분포 통계 | ✅ | — |
| Q-value 분석 | ✅ (D3QN) | ✅ (AlphaZero) |
| 최악의 선택 TOP-5 (다양성 보장) | ✅ | ✅ |
| AI 권장 행동 비교 | ✅ | ✅ |
| Human vs Agent 비교 리플레이 | ✅ | ✅ |
| 비교 리플레이 배속 컨트롤 | ✅ (×¼~×4) | — |
| 비교 리플레이 수동 프레임 탐색 | — | ✅ (◀/▶) |
| 비교 리플레이 Q-value 히트맵 | — | ✅ |
| 리플레이 가상 키보드 시각화 | ✅ | — |
| LLM / 로컬 코칭 피드백 | ✅ | ✅ |
| 세션 저장 / 불러오기 | ✅ | ✅ |

---

## 🛠️ 기술 스택

- **Backend**: Flask, Flask-SocketIO, eventlet
- **RL 환경**: Gymnasium (ALE/Atari)
- **AI 모델**: PyTorch
- **Frontend**: Vanilla JS, Socket.IO, Canvas API
- **디자인**: Press Start 2P 폰트, 레트로 아케이드 테마

---

## 📝 라이선스
- Team A1
- AlphaZero_Gomoku 서브모듈: [원본 라이선스](https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/LICENSE) 참조
