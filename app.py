"""
app.py — AI ARCADE 서버

새 게임 추가 방법:
  1. games/mygame.py 생성 (AtariGame 상속)
  2. 아래 ATARI_GAMES 리스트에 추가
  그게 전부입니다.
"""

import eventlet
eventlet.monkey_patch()

import os, json
from flask import render_template, request, jsonify
from flask_socketio import emit

from extensions import app, socketio, DEVICE, SAVED_SESSIONS_DIR
from llm_feedback import (generate_feedback, test_openrouter_connection,
                          DEFAULT_MODEL, FALLBACK_MODELS,
                          FALLBACK_PRIORITY, FALLBACK_POOL, short_model_name)

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if cfg.get('openrouter_api_key'):
                os.environ['OPENROUTER_API_KEY'] = cfg['openrouter_api_key']
            if cfg.get('openrouter_model'):
                os.environ['OPENROUTER_MODEL'] = cfg['openrouter_model']
        except Exception as e:
            print(f'⚠️  config.json 로드 실패: {e}')

def save_config(api_key: str, model: str):
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            pass
    cfg['openrouter_api_key'] = api_key
    cfg['openrouter_model']   = model
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

load_config()

# ── Atari 게임 등록 ──────────────────────────────────────────────────────────

import ale_py  # noqa  (ALE 환경 등록)
import gymnasium as gym
gym.register_envs(ale_py)  # ALE 환경 명시적 등록

from games.space_invaders import SpaceInvadersGame
from games.breakout import BreakoutGame
from games.enduro import EnduroGame
from games.alien import AlienGame
from games.amidar import AmidarGame
from games.assault import AssaultGame
from games.asterix import AsterixGame
from games.asteroids import AsteroidsGame
from games.atlantis import AtlantisGame
from games.mariobros import MarioBrosGame

ATARI_GAMES = [
    SpaceInvadersGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    BreakoutGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    EnduroGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AlienGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AmidarGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AssaultGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AsterixGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AsteroidsGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    AtlantisGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
    MarioBrosGame(DEVICE, socketio, app, SAVED_SESSIONS_DIR),
]

# ── Gomoku 핸들러 등록 ───────────────────────────────────────────────────────
import games.gomoku.handlers  # noqa  (side effect: @socketio.on 등록)
from games.gomoku.sessions import list_saved_sessions as _list_gomoku_sessions

# ── Flask Routes ──────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip('#')
    return ','.join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))

@app.route('/')
def index():
    # ATARI_GAMES 메타를 Jinja에 전달 → 카드 자동 생성
    games_meta = [
        {
            'url':         g.url_path,
            'game_id':     g.game_id,
            'game_title':  g.game_title,
            'game_icon':   g.game_icon,
            'theme_color': g.theme_color,
            'theme_rgb':   _hex_to_rgb(g.theme_color),
            'has_model':   g.net is not None,
            'summary':     (g.game_info or {}).get('summary', ''),
        }
        for g in ATARI_GAMES
    ]
    return render_template('index.html', atari_games=games_meta, dance_games=[])

_GOMOKU_GAME_INFO = {
    'summary': '15×15 바둑판에서 가로·세로·대각선으로 5개를 먼저 이으면 승리하는 전략 보드 게임입니다.',
    'objective': 'AlphaZero 기반 AI를 상대로 먼저 5목을 달성하세요.',
    'how_to_play': [
        '원하는 교차점을 클릭해 돌을 놓습니다.',
        '흑(●)이 먼저 두며, 흑·백 교대로 진행됩니다.',
        '가로·세로·대각선 어느 방향으로든 5개 연속이면 즉시 승리입니다.',
        '시작 전 COLOR 패널에서 흑·백을 선택할 수 있습니다.',
    ],
    'scoring': [
        '승률 바: 화면 좌측에 흑·백 실시간 승률 표시 (AI 분석 기준)',
        '평균 착수 손실(Avg Loss): 게임 후 내 수의 품질 평가',
        '착수 손실 < 0.05 → 탁월한 플레이',
        '착수 손실 < 0.12 → 좋은 플레이',
        '착수 손실 < 0.20 → 평균적 플레이',
        '착수 손실 ≥ 0.20 → 개선 필요',
    ],
    'end_condition': '5목 달성 즉시 종료됩니다. 보드가 꽉 차면 무승부입니다.',
    'tips': [
        '승률 바가 급격히 내려가는 수가 패인 — 게임 후 TOP-5 분석에서 확인하세요.',
        '공격과 수비를 동시에 고려하세요. 4목을 막지 못하면 즉시 역전됩니다.',
    ],
}

@app.route('/gomoku')
def gomoku_page():
    primary = os.getenv('OPENROUTER_MODEL', DEFAULT_MODEL)
    return render_template('gomoku.html',
        has_openrouter=bool(os.getenv('OPENROUTER_API_KEY')),
        openrouter_model=primary,
        primary_short=short_model_name(primary),
        priority_models=[(m, short_model_name(m)) for m in FALLBACK_PRIORITY],
        pool_models=[(m, short_model_name(m)) for m in FALLBACK_POOL],
        saved_sessions=_list_gomoku_sessions('gomoku'),
        game_info=_GOMOKU_GAME_INFO,
    )

@app.route('/api/settings', methods=['GET'])
def get_settings():
    key = os.getenv('OPENROUTER_API_KEY', '')
    masked = ('*' * (len(key)-4) + key[-4:]) if len(key) > 4 else '*' * len(key)
    return jsonify({'has_key': bool(key), 'masked_key': masked,
                    'model': os.getenv('OPENROUTER_MODEL', DEFAULT_MODEL)})

@app.route('/api/settings', methods=['POST'])
def post_settings():
    data    = request.get_json(force=True) or {}
    api_key = data.get('api_key', '').strip()
    model   = data.get('model', '').strip() or DEFAULT_MODEL
    if data.get('clear_key'): os.environ.pop('OPENROUTER_API_KEY', None)
    elif api_key:             os.environ['OPENROUTER_API_KEY'] = api_key
    os.environ['OPENROUTER_MODEL'] = model
    save_config(os.getenv('OPENROUTER_API_KEY', ''), model)
    return jsonify({'ok': True})

@app.route('/api/settings/test', methods=['POST'])
def test_settings():
    ok, payload = test_openrouter_connection()
    summary = payload.get('summary', '') if isinstance(payload, dict) else str(payload)
    return jsonify({'ok': ok, 'message': summary})

# ── Socket: LLM test ─────────────────────────────────────────────────────────

@socketio.on('llm_test')
def handle_llm_test():
    ok, payload = test_openrouter_connection()
    if isinstance(payload, str): payload = {'summary': payload, 'results': []}
    emit('llm_test_result', {'ok': ok, **payload})

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('🚀 AI Arcade server running on http://127.0.0.1:5001')
    socketio.run(app, host='0.0.0.0', debug=False, port=5001)
