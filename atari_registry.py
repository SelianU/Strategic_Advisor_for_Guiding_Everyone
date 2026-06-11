"""
atari_registry.py — Atari 게임 임포트 + ATARI_GAMES 리스트.

새 게임 추가 시:
  1) games/mygame.py 작성 (AtariGame 상속)
  2) 아래 import 추가
  3) ATARI_GAMES 리스트에 인스턴스 추가
"""
import ale_py  # noqa: F401  (ALE 환경 등록 side effect)
import gymnasium as gym

gym.register_envs(ale_py)  # ALE 환경 명시적 등록

from games.space_invaders import SpaceInvadersGame
from games.breakout       import BreakoutGame
from games.enduro         import EnduroGame
from games.alien          import AlienGame
from games.amidar         import AmidarGame
from games.assault        import AssaultGame
from games.asterix        import AsterixGame
from games.asteroids      import AsteroidsGame
from games.atlantis       import AtlantisGame
from games.mariobros      import MarioBrosGame

from extensions import app, socketio, DEVICE, SAVED_SESSIONS_DIR

ATARI_GAMES = [
    SpaceInvadersGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    BreakoutGame     (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    EnduroGame       (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AlienGame        (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AmidarGame       (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AssaultGame      (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AsterixGame      (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AsteroidsGame    (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AtlantisGame     (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    MarioBrosGame    (DEVICE, socketio, app, SAVED_SESSIONS_DIR),
]
