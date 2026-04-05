# 🕹️ AI ARCADE

강화학습 AI와 함께하는 레트로 아케이드 게임 플랫폼.
플레이 후 AI가 당신의 행동을 Q-value 기반으로 분석해줍니다.

---

## 🎮 게임 목록

### 👾 Space Invaders
- Atari Space Invaders를 직접 플레이
- **D3QN (Dueling Double DQN)** 모델이 플레이어의 모든 행동을 분석
- 게임 종료 후 Q-value 손실, 행동 일치율, 최악의 선택 TOP 5 등 상세 피드백 제공
- Top-5 후보를 선택해 **Human vs Agent 비교 리플레이**와 코칭 피드백 확인 가능

### ⚫ Gomoku vs AI
- **AlphaZero** 기반 AI와 8×8 오목 대결
- 게임 종료 후 각 착수의 Q-value와 최선 착수를 비교 분석
- Top-5 후보를 선택해 **Human vs Agent 비교 리플레이**와 코칭 피드백 확인 가능

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

외부 LLM 코칭 피드백을 사용하려면 환경 변수를 설정하세요.

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
| 플레이 스타일 분류 | ✅ | — |
| 행동 분포 통계 | ✅ | — |
| Q-value 분석 | ✅ (D3QN) | ✅ (AlphaZero) |
| 최악의 선택 TOP 5 | ✅ | ✅ |
| AI 권장 행동 비교 | ✅ | ✅ |
| Human vs Agent 비교 리플레이 | ✅ | ✅ |
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
