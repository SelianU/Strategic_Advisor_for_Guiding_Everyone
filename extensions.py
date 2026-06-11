"""
extensions.py — 모든 모듈이 공유하는 Flask / SocketIO 인스턴스와 경로 상수.

여기를 별도 모듈로 분리하는 이유:
  - 다른 모듈에서 `from extensions import app, socketio` 로 가져올 수 있어야
    @app.route 와 @socketio.on 데코레이터를 분산 등록할 수 있음.
  - app.py 가 이 모듈을 임포트하기 전에 eventlet.monkey_patch() 가 호출되어 있으므로,
    flask_socketio 를 임포트하는 본 모듈 단계에서는 patch 가 이미 끝나 있음.
"""
import os
import torch
from flask import Flask
from flask_socketio import SocketIO

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

# 런타임 산출물 루트 — 체크포인트(data/checkpoints)·세션(data/saved_sessions) 등
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SAVED_SESSIONS_DIR = os.path.join(DATA_DIR, 'saved_sessions')