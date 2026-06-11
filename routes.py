"""
routes.py — Flask 라우트 및 설정 관련 엔드포인트.

  GET  /                     → 게임 카드 메인
  GET  /gomoku               → Gomoku 게임 페이지
  GET  /api/settings         → 현재 LLM 설정 조회
  POST /api/settings         → LLM 설정 저장 (또는 키 해제)
  POST /api/settings/test    → OpenRouter 연결 테스트

  SocketIO event 'llm_test'  → 동일 연결 테스트 (소켓 경로)

LLM 테스트 SocketIO 핸들러도 /api/settings/test 와 짝을 이루므로
설정 도메인끼리 같은 모듈에 모아 둔다.
"""
import os

from flask         import render_template, request, jsonify
from flask_socketio import emit

from extensions     import app, socketio
from config_store   import save_config, is_valid_api_key
from atari_registry import ATARI_GAMES
from llm_feedback   import (test_openrouter_connection, DEFAULT_MODEL,
                            FALLBACK_PRIORITY, FALLBACK_POOL, short_model_name)
from games.gomoku.game_info import GOMOKU_GAME_INFO
from games.gomoku.sessions  import list_saved_sessions


# ── 헬퍼 ────────────────────────────────────────────────────────────────────
def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip('#')
    return ','.join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ── 페이지 라우트 ────────────────────────────────────────────────────────────
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


@app.route('/gomoku')
def gomoku_page():
    primary = os.getenv('OPENROUTER_MODEL', DEFAULT_MODEL)
    return render_template('gomoku.html',
        has_openrouter=bool(os.getenv('OPENROUTER_API_KEY')),
        openrouter_model=primary,
        primary_short=short_model_name(primary),
        priority_models=[(m, short_model_name(m)) for m in FALLBACK_PRIORITY],
        pool_models=[(m, short_model_name(m)) for m in FALLBACK_POOL],
        saved_sessions=list_saved_sessions('gomoku'),
        game_info=GOMOKU_GAME_INFO,
    )


# ── 설정 API ────────────────────────────────────────────────────────────────
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
    if data.get('clear_key'):          os.environ.pop('OPENROUTER_API_KEY', None)
    elif is_valid_api_key(api_key):    os.environ['OPENROUTER_API_KEY'] = api_key
    elif api_key:
        return jsonify({'ok': False, 'message': '유효한 API 키 형식이 아닙니다.'})
    os.environ['OPENROUTER_MODEL'] = model
    save_config(os.getenv('OPENROUTER_API_KEY', ''), model)
    return jsonify({'ok': True})


@app.route('/api/settings/test', methods=['POST'])
def test_settings():
    ok, payload = test_openrouter_connection()
    summary = payload.get('summary', '') if isinstance(payload, dict) else str(payload)
    return jsonify({'ok': ok, 'message': summary})


# ── 설정 SocketIO ───────────────────────────────────────────────────────────
@socketio.on('llm_test')
def handle_llm_test():
    ok, payload = test_openrouter_connection()
    if isinstance(payload, str): payload = {'summary': payload, 'results': []}
    emit('llm_test_result', {'ok': ok, **payload})
