"""
app.py — AI ARCADE 서버 진입점

새 게임 추가 방법:
  1. games/mygame.py 생성 (AtariGame 상속)
  2. atari_registry.py 의 ATARI_GAMES 리스트에 추가
  그게 전부입니다.

모듈 분리 구조:
  app.py                  — 진입점 (eventlet patch + 등록 트리거 + run)
  extensions.py           — Flask / SocketIO 인스턴스 + DEVICE 등 공유 객체
  config_store.py         — config.json 영속화
  atari_registry.py       — Atari 게임 임포트 + ATARI_GAMES 리스트
  routes.py               — Flask 라우트 + LLM 테스트 소켓
  gomoku/                 — Gomoku 게임 백엔드 패키지
    __init__.py
    state.py              — 상태 싱글턴 + PPO 모델 + 보드 상수
    engine.py             — 순수 로직 (Q-value / 승률 / 승리선 / 결과)
    render.py             — cv2 보드 렌더링 + 프레임 인코딩
    sessions.py           — 세션 디스크 영속화 + 직렬화 헬퍼
    handlers.py           — SocketIO 핸들러 (게임 / 세션 / 분석 / counterfactual)
"""

# ── 1. eventlet monkey patch는 가장 먼저 ─────────────────────────────────────
import eventlet
eventlet.monkey_patch()

# ── 2. 공유 인스턴스 생성 (Flask / SocketIO) ────────────────────────────────
from extensions import app, socketio

# ── 3. 설정 로드 ────────────────────────────────────────────────────────────
from config_store import load_config
load_config()

# ── 4. Atari 게임 등록 (각 게임 생성자가 자체 라우트/핸들러를 등록) ───────
import atari_registry  # noqa: F401  (side effect: ATARI_GAMES 생성)

# ── 5. Gomoku 모델 로드 (gomoku.state 임포트 시 자동 로드) ──────────────────
import games.gomoku.state     # noqa: F401  (side effect: PPO 모델 로드)

# ── 6. SocketIO 핸들러 등록 ─────────────────────────────────────────────────
import games.gomoku.handlers  # noqa: F401  (side effect: @socketio.on 등록)
import routes           # noqa: F401  (side effect: @app.route + @socketio.on 등록)


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('🚀 AI Arcade server running on http://127.0.0.1:5001')
    socketio.run(app, host='0.0.0.0', debug=False, port=5001)