"""
atari_registry.py — Atari 게임 임포트 + ATARI_GAMES 리스트.

새 게임 추가 시:
  1) games/mygame.py 작성 (AtariGame 상속)
  2) 아래 import 추가
  3) ATARI_GAMES 리스트에 인스턴스 추가
"""
import ale_py  # noqa: F401  (ALE 환경 등록 side effect)

from games.space_invaders import SpaceInvadersGame
from games.breakout       import BreakoutGame
from games.enduro         import EnduroGame
from games.alien          import AlienGame
# from games.pong         import PongGame  # ← 예시: 새 게임 추가 시 이 줄 활성화

from extensions import app, socketio, DEVICE, SAVED_SESSIONS_DIR

ATARI_GAMES = [
    SpaceInvadersGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    BreakoutGame     (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    EnduroGame       (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AlienGame        (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
]