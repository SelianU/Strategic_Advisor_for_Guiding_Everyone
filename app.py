"""
app.py — AI ARCADE 서버

새 게임 추가 방법:
  1. games/mygame.py 생성 (AtariGame 상속)
  2. 아래 ATARI_GAMES 리스트에 추가
  그게 전부입니다.
"""

import eventlet
eventlet.monkey_patch()

import os, copy, pickle, base64, json, uuid
from datetime import datetime
from collections import deque

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import torch

from ai_agents.gomoku import GomokuRLBoard, GomokuPPOEngine, load_gomoku_ppo
from llm_feedback import (generate_feedback, test_openrouter_connection,
                          DEFAULT_MODEL, FALLBACK_MODELS,
                          FALLBACK_PRIORITY, FALLBACK_POOL, short_model_name)

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVED_SESSIONS_DIR = os.path.join(os.path.dirname(__file__), 'saved_sessions')

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

DANCE_GAMES = []

# ── Gomoku 모델 ──────────────────────────────────────────────────────────────

BOARD_W, BOARD_H, N_IN_ROW = 15, 15, 5
_GOMOKU_PPO_PATH = os.path.join(
    os.path.dirname(__file__), 'ai_agents', 'gomoku', 'gomoku_rl',
    'pretrained_models', '15_15', 'ppo', '0.pt'
)
gomoku_net = load_gomoku_ppo(_GOMOKU_PPO_PATH, board_size=BOARD_W, device=DEVICE)
print('✅ Gomoku PPO 모델 로드 완료 (15×15)')

# ── Gomoku 상태 ──────────────────────────────────────────────────────────────

gomoku_board             = None
gomoku_history           = []
gomoku_active            = False
ai_player                = None
gomoku_analysis_results  = []
gomoku_counterfactual_cache = {}
gomoku_session_id        = 0
gomoku_analysis_timer    = None
gomoku_ai_turn_timer     = None
gomoku_state_seq         = 0
gomoku_human_color       = 1

# ── Gomoku 유틸 ──────────────────────────────────────────────────────────────

def encode_frame(frame):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')

def encode_frames(frames):
    return [encode_frame(f) for f in frames]

def compute_q_values(board, policy_value_net, n_playout=80):
    legal = list(board.availables)
    if not legal:
        return {'q_values': np.full(board.width * board.height, np.nan, dtype=np.float32),
                'best_action': -1, 'v_s': 0.0}
    act_probs_iter, v_s = policy_value_net.policy_value_fn(board)
    act_probs = dict(act_probs_iter)
    q_values  = np.full(board.width * board.height, np.nan, dtype=np.float32)
    for a in legal:
        q_values[a] = float(act_probs.get(a, 0.0))
    return {'q_values': q_values,
            'best_action': int(max(legal, key=lambda a: q_values[a])),
            'v_s': float(v_s)}

def compute_human_winrate(board) -> float:
    """
    현재 보드 상태에서 '인간 플레이어' 관점의 승률(0~100)을 계산합니다.
    
    PPO critic의 v_s는 "다음에 둘 차례인 플레이어" 관점:
      - 그 플레이어가 이길 것 같으면 +1에 가까움
      - 그 플레이어가 질 것 같으면 -1에 가까움
    
    인간이 둘 차례면 v_s를 그대로 쓰고,
    AI가 둘 차례면 부호를 뒤집어 인간 관점으로 변환합니다.
    """
    legal = list(board.availables)
    if not legal:
        return 50.0  # 둘 곳 없으면 중립
    _, v_s = gomoku_net.policy_value_fn(board)
    # board.current_player: 1=흑, 2=백 (다음에 둘 사람)
    v_human = v_s if board.current_player == gomoku_human_color else -v_s
    # v_human ∈ [-1, 1]을 [0, 100]으로 변환
    win_rate = (v_human + 1.0) / 2.0 * 100.0
    return round(float(win_rate), 1)

def choose_gomoku_best_move(board, n_playout=80):
    res = compute_q_values(board, gomoku_net, n_playout=n_playout)
    return int(res['best_action']), res

def emit_gomoku_terminal_state(payload, sid):
    socketio.emit('gomoku_state', payload, room=sid, namespace='/')

def board_to_dict(board):
    cells = [int(board.states.get(r * BOARD_W + c, 0))
             for r in range(BOARD_H) for c in range(BOARD_W)]
    return {'cells': cells, 'current_player': int(board.current_player),
            'last_move': int(board.last_move) if board.last_move != -1 else -1,
            'width': BOARD_W, 'height': BOARD_H}

def find_gomoku_winning_line(board):
    width, height, states, n = board.width, board.height, board.states, board.n_in_row
    moved = list(set(range(width * height)) - set(board.availables))
    if len(moved) < n * 2 - 1:
        return None, -1
    for move in moved:
        row, col = move // width, move % width
        player = states.get(move, -1)
        if player == -1:
            continue
        for dx, dy, cond in [
            (1, 0, col <= width - n),
            (0, 1, row <= height - n),
            (1, 1, col <= width - n and row <= height - n),
            (1,-1, col <= width - n and row >= n - 1),
        ]:
            line = [move + (dx + dy * width) * i for i in range(n)]
            if cond and len(set(states.get(p, -1) for p in line)) == 1:
                return line, player
    return None, -1

def human_perspective_outcome_text(winner):
    if winner == -1:          return 'DRAW'
    if winner == gomoku_human_color: return 'WIN'
    return 'LOSE'

def _q_to_bgr(t):
    if t < 0.5:
        s = t * 2
        return (int(210-160*s), int(80+130*s), int(30+20*s))
    s = (t-0.5)*2
    return (int(50-50*s), int(210-180*s), int(50+205*s))

def render_gomoku_board(board, highlight_move=None, title='', highlight_black_only=True,
                         winning_line=None, outcome_text=None, q_values=None):
    margin = 36
    cell   = max(24, 510 // BOARD_W)
    size   = margin * 2 + cell * (BOARD_W - 1)
    stone_r = max(9, cell // 3)
    img = np.full((size + 48, size, 3), 226, dtype=np.uint8)
    img[:, :] = (226, 194, 140)
    cv2.rectangle(img, (0, 0), (size-1, size+47), (70, 45, 20), 2)
    if title:
        cv2.putText(img, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    for i in range(BOARD_W):
        x = margin + i * cell
        cv2.line(img, (x, margin+12), (x, margin+12+cell*(BOARD_H-1)), (80, 60, 35), 1)
    for i in range(BOARD_H):
        y = margin + 12 + i * cell
        cv2.line(img, (margin, y), (margin+cell*(BOARD_W-1), y), (80, 60, 35), 1)
    if q_values is not None:
        valid = [(pos, float(q_values[pos])) for pos in board.availables
                 if pos < len(q_values) and not np.isnan(q_values[pos]) and q_values[pos] > 1e-6]
        if valid:
            valid.sort(key=lambda x: x[1], reverse=True)
            top_k = valid[:8]
            q_min, q_max = top_k[-1][1], top_k[0][1]
            q_range = max(q_max - q_min, 1e-8)
            heat = img.copy()
            hr_min, hr_max = max(4, cell//7), max(7, cell//4)
            for pos, qv in top_k:
                t   = (qv - q_min) / q_range
                row, col = divmod(pos, BOARD_W)
                x   = margin + col * cell
                y   = margin + 12 + ((BOARD_H-1) - row) * cell
                cv2.circle(heat, (x, y), int(hr_min + (hr_max-hr_min)*t), _q_to_bgr(t), -1)
                cv2.circle(heat, (x, y), int(hr_min + (hr_max-hr_min)*t), (210, 210, 210), 1)
            img = cv2.addWeighted(heat, 0.75, img, 0.25, 0)
    for move, player in board.states.items():
        row, col = divmod(move, BOARD_W)
        x = margin + col * cell
        y = margin + 12 + ((BOARD_H-1) - row) * cell
        color = (30, 30, 30) if player == 1 else (240, 240, 240)
        cv2.circle(img, (x, y), stone_r, color, -1)
        cv2.circle(img, (x, y), stone_r, (60, 60, 60), 1)
    if winning_line:
        pts = []
        for move in winning_line:
            row, col = divmod(move, BOARD_W)
            pts.append((margin+col*cell, margin+12+((BOARD_H-1)-row)*cell))
        lc = (70,220,120) if outcome_text=='WIN' else ((80,80,255) if outcome_text=='LOSE' else (0,215,255))
        cv2.line(img, pts[0], pts[-1], lc, 4, cv2.LINE_AA)
        for p in pts: cv2.circle(img, p, stone_r+4, lc, 2, cv2.LINE_AA)
    if highlight_move is not None and highlight_move != -1:
        hp = board.states.get(highlight_move)
        if highlight_black_only and hp != 1 and not winning_line and not outcome_text:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        row, col = divmod(highlight_move, BOARD_W)
        x = margin + col * cell
        y = margin + 12 + ((BOARD_H-1) - row) * cell
        cv2.circle(img, (x, y), stone_r+5, (0, 255, 255), 2)
    if outcome_text:
        ov = img.copy()
        cv2.rectangle(ov, (82, 18), (size-82, 66), (20, 20, 20), -1)
        img = cv2.addWeighted(ov, 0.42, img, 0.58, 0)
        tc  = (70,220,120) if outcome_text=='WIN' else ((80,80,255) if outcome_text=='LOSE' else (0,215,255))
        tw  = cv2.getTextSize(outcome_text, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)[0]
        cv2.putText(img, outcome_text, ((size-tw[0])//2, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.95, tc, 3, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def serialize_gomoku_history():
    return [{'move': e['move'], 'row': e['row'], 'col': e['col'],
             **({'ai_move': e['ai_move']} if e.get('ai_move') is not None else {})}
            for e in gomoku_history]

def rebuild_gomoku_history(saved_moves):
    board = GomokuRLBoard(board_size=BOARD_W, n_in_row=N_IN_ROW)
    board.init_board(start_player=0)
    rebuilt = []
    for item in saved_moves:
        bb = copy.deepcopy(board)
        board.do_move(item['move'])
        ri = {'move': item['move'], 'row': item['row'], 'col': item['col'], 'board_before': bb}
        ai = item.get('ai_move')
        if ai is not None and ai in board.availables:
            board.do_move(ai)
            ri['ai_move'] = ai
        rebuilt.append(ri)
    return board, rebuilt

def gomoku_analysis_snapshot():
    if not gomoku_analysis_results: return None
    losses = [a['loss'] for a in gomoku_analysis_results if a['loss'] is not None]
    return {
        'analyses':    gomoku_analysis_results,
        'worst':       max((a for a in gomoku_analysis_results if a['loss'] is not None), key=lambda a: a['loss'], default=None),
        'worst_10':    sorted((a for a in gomoku_analysis_results if a['loss'] is not None), key=lambda a: a['loss'], reverse=True)[:5],
        'avg_loss':    round(sum(losses)/len(losses), 3) if losses else 0,
        'total_moves': len(gomoku_history),
    }

def serialize_gomoku_cf_cache():
    saved = []
    for (sid, move_num), payload in gomoku_counterfactual_cache.items():
        if sid != gomoku_session_id: continue
        item = copy.deepcopy(payload)
        item['move_num'] = move_num
        saved.append(item)
    return saved

def list_saved_sessions(game_name):
    path = os.path.join(SAVED_SESSIONS_DIR, game_name)
    os.makedirs(path, exist_ok=True)
    sessions = []
    for sid in os.listdir(path):
        meta_path = os.path.join(path, sid, 'meta.json')
        if not os.path.isfile(meta_path): continue
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            sessions.append({'id': sid, 'title': meta.get('title') or '이름 없음',
                             'saved_at': meta.get('saved_at') or '', 'summary': meta.get('summary') or ''})
        except Exception: continue
    sessions.sort(key=lambda x: x['saved_at'], reverse=True)
    return sessions

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
    dance_meta = [
        {
            'url':         g.url_path,
            'game_id':     g.game_id,
            'game_title':  g.game_title,
            'game_icon':   g.game_icon,
            'theme_color': g.theme_color,
            'theme_rgb':   _hex_to_rgb(g.theme_color),
            'has_model':   True,
        }
        for g in DANCE_GAMES
    ]
    return render_template('index.html', atari_games=games_meta, dance_games=dance_meta)

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
        saved_sessions=list_saved_sessions('gomoku'),
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

# ── Socket: Gomoku Sessions ──────────────────────────────────────────────────

@socketio.on('gomoku_list_sessions')
def handle_gomoku_list_sessions():
    emit('gomoku_sessions_list', {'sessions': list_saved_sessions('gomoku')})

@socketio.on('gomoku_save_session')
def handle_gomoku_save_session(data):
    if not gomoku_history or not gomoku_analysis_results:
        emit('gomoku_session_saved', {'ok': False, 'message': '저장할 분석 결과가 없습니다.'}); return
    title    = (data.get('title') or '').strip() or 'Gomoku 기록'
    sid      = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
    sess_dir = os.path.join(SAVED_SESSIONS_DIR, 'gomoku', sid)
    os.makedirs(sess_dir, exist_ok=True)
    snapshot = gomoku_analysis_snapshot()
    meta = {'game': 'gomoku', 'title': title,
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'summary': f'총 {len(gomoku_history)}수 · 평균 손실 {snapshot["avg_loss"]}'}
    with open(os.path.join(sess_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(sess_dir, 'session.pkl'), 'wb') as f:
        pickle.dump({'history': serialize_gomoku_history(), 'analysis_results': gomoku_analysis_results,
                     'analysis_snapshot': snapshot, 'cached_counterfactuals': serialize_gomoku_cf_cache()}, f)
    emit('gomoku_session_saved', {'ok': True, 'message': '기록을 저장했습니다.', 'sessions': list_saved_sessions('gomoku')})

@socketio.on('gomoku_load_session')
def handle_gomoku_load_session(data):
    global gomoku_board, gomoku_history, gomoku_analysis_results, gomoku_counterfactual_cache, gomoku_session_id, gomoku_active
    sess_dir = os.path.join(SAVED_SESSIONS_DIR, 'gomoku', data.get('session_id', ''))
    meta_p, data_p = os.path.join(sess_dir, 'meta.json'), os.path.join(sess_dir, 'session.pkl')
    if not (os.path.isfile(meta_p) and os.path.isfile(data_p)):
        emit('gomoku_session_loaded', {'ok': False, 'message': '기록을 찾지 못했습니다.'}); return
    with open(meta_p, 'r', encoding='utf-8') as f: meta = json.load(f)
    with open(data_p, 'rb') as f: payload = pickle.load(f)
    gomoku_session_id += 1
    gomoku_board, gomoku_history = rebuild_gomoku_history(payload['history'])
    gomoku_analysis_results = payload['analysis_results']
    gomoku_counterfactual_cache = {}
    cached = payload.get('cached_counterfactuals') or []
    for item in cached:
        mn = int(item.get('move_num', -1))
        restored = copy.deepcopy(item); restored['session_id'] = gomoku_session_id
        gomoku_counterfactual_cache[(gomoku_session_id, mn)] = restored
    gomoku_active = False
    emit('gomoku_session_loaded', {'ok': True, 'meta': meta,
        'analysis': payload.get('analysis_snapshot') or gomoku_analysis_snapshot(),
        'board': board_to_dict(gomoku_board), 'cached_counterfactuals': cached,
        'session_id': gomoku_session_id})

@socketio.on('gomoku_delete_session')
def handle_gomoku_delete_session(data):
    import shutil
    sess_dir = os.path.join(SAVED_SESSIONS_DIR, 'gomoku', data.get('session_id', ''))
    if os.path.isdir(sess_dir): shutil.rmtree(sess_dir)
    emit('gomoku_session_deleted', {'ok': True, 'message': '기록을 삭제했습니다.', 'sessions': list_saved_sessions('gomoku')})

# ── Socket: Gomoku Game ───────────────────────────────────────────────────────

def schedule_gomoku_analysis(session_id):
    global gomoku_analysis_timer
    if gomoku_analysis_timer is not None:
        try: gomoku_analysis_timer.cancel()
        except Exception: pass
    gomoku_analysis_timer = eventlet.spawn_after(3.2, run_gomoku_analysis, session_id)

@socketio.on('gomoku_start')
def handle_gomoku_start(data=None):
    global gomoku_board, gomoku_history, gomoku_active, ai_player, gomoku_analysis_results
    global gomoku_counterfactual_cache, gomoku_session_id, gomoku_analysis_timer
    global gomoku_ai_turn_timer, gomoku_state_seq, gomoku_human_color
    for timer in [gomoku_analysis_timer, gomoku_ai_turn_timer]:
        if timer is not None:
            try: timer.cancel()
            except Exception: pass
    gomoku_analysis_timer = gomoku_ai_turn_timer = None
    data = data or {}
    gomoku_human_color = int(data.get('human_color', 1))
    gomoku_session_id += 1
    gomoku_board = GomokuRLBoard(board_size=BOARD_W, n_in_row=N_IN_ROW)
    gomoku_board.init_board(start_player=0)
    gomoku_history = []; gomoku_analysis_results = []; gomoku_counterfactual_cache = {}
    gomoku_state_seq = 1; gomoku_active = True; ai_player = gomoku_net
    color_text = '흑(●)' if gomoku_human_color == 1 else '백(○)'
    emit('gomoku_state', {**board_to_dict(gomoku_board), 'message': f'당신은 {color_text}입니다!',
        'human_color': gomoku_human_color, 'session_id': gomoku_session_id, 'state_seq': gomoku_state_seq,
        'win_rate': compute_human_winrate(gomoku_board)})
    if gomoku_human_color == 2:
        sid = request.sid; eventlet.sleep(0); eventlet.sleep(0.4)
        _process_gomoku_ai_opening(gomoku_session_id, sid)

def _process_gomoku_ai_opening(session_id, sid):
    global gomoku_board, gomoku_active, gomoku_state_seq
    if session_id != gomoku_session_id or not gomoku_active or gomoku_board is None: return
    try:
        ai_move = ai_player.get_action(gomoku_board, temp=1e-3)
        ai_r, ai_c = int(ai_move // BOARD_W), int(ai_move % BOARD_W)
        gomoku_board.do_move(ai_move); gomoku_state_seq += 1
        socketio.emit('gomoku_state', {**board_to_dict(gomoku_board),
            'message': f'AI 선착: ({BOARD_H - ai_r},{ai_c + 1}) — 당신의 차례입니다.',
            'human_color': gomoku_human_color, 'session_id': session_id, 'state_seq': gomoku_state_seq,
            'win_rate': compute_human_winrate(gomoku_board)},
            room=sid, namespace='/')
    except Exception as exc: print(f'AI 선착 오류: {exc}')

@socketio.on('gomoku_move')
def handle_gomoku_move(data):
    global gomoku_board, gomoku_history, gomoku_active, gomoku_state_seq
    if not gomoku_active: return
    sid = request.sid
    row, col = data.get('row'), data.get('col')
    move = row * BOARD_W + col
    if move not in gomoku_board.availables:
        emit('gomoku_error', {'message': '이미 돌이 놓인 위치입니다.'}, to=sid); return
    if gomoku_board.current_player != gomoku_human_color: return
    board_before = copy.deepcopy(gomoku_board)
    gomoku_board.do_move(move)
    gomoku_history.append({'move': move, 'row': row, 'col': col, 'board_before': board_before})
    end, winner = gomoku_board.game_end()
    if end:
        gomoku_active = False
        r = '흑(●) 승리!' if winner==1 else ('백(○) 승리!' if winner==2 else '무승부')
        wl, _ = find_gomoku_winning_line(gomoku_board) if winner in (1,2) else (None,-1)
        emit_gomoku_terminal_state({**board_to_dict(gomoku_board), 'message': f'게임 종료 — {r}',
            'game_over': True, 'winner': int(winner), 'winning_line': wl,
            'outcome_text': human_perspective_outcome_text(winner),
            'human_color': gomoku_human_color, 'session_id': gomoku_session_id,
            'state_seq': gomoku_state_seq+1,
            'win_rate': 100.0 if winner == gomoku_human_color else (50.0 if winner == -1 else 0.0)}, sid)
        gomoku_state_seq += 1; schedule_gomoku_analysis(gomoku_session_id); return
    gomoku_state_seq += 1
    emit('gomoku_state', {**board_to_dict(gomoku_board),
        'message': f'당신의 착수: ({BOARD_H - row},{col + 1}) — AI 생각 중...',
        'human_color': gomoku_human_color, 'session_id': gomoku_session_id, 'state_seq': gomoku_state_seq,
        'win_rate': compute_human_winrate(gomoku_board)}, to=sid)
    eventlet.sleep(0); eventlet.sleep(0.55)
    process_gomoku_ai_turn(gomoku_session_id, len(gomoku_history)-1, sid)

def process_gomoku_ai_turn(session_id, history_index, sid):
    global gomoku_board, gomoku_history, gomoku_active, gomoku_state_seq
    try:
        if session_id != gomoku_session_id or not gomoku_active or history_index >= len(gomoku_history): return
        ai_move = ai_player.get_action(gomoku_board, temp=1e-3)
        ai_r, ai_c = int(ai_move // BOARD_W), int(ai_move % BOARD_W)
        gomoku_board.do_move(ai_move); gomoku_history[history_index]['ai_move'] = ai_move
        end2, winner2 = gomoku_board.game_end()
        if end2:
            gomoku_active = False
            r = '흑(●) 승리!' if winner2==1 else ('백(○) 승리!' if winner2==2 else '무승부')
            wl, _ = find_gomoku_winning_line(gomoku_board) if winner2 in (1,2) else (None,-1)
            emit_gomoku_terminal_state({**board_to_dict(gomoku_board), 'message': f'AI 착수: ({BOARD_H - ai_r},{ai_c + 1}) — {r}',
                'game_over': True, 'winner': int(winner2), 'winning_line': wl,
                'outcome_text': human_perspective_outcome_text(winner2),
                'human_color': gomoku_human_color, 'session_id': session_id,
                'state_seq': gomoku_state_seq+1,
                'win_rate': 100.0 if winner2 == gomoku_human_color else (50.0 if winner2 == -1 else 0.0)}, sid)
            gomoku_state_seq += 1; schedule_gomoku_analysis(session_id); return
        gomoku_state_seq += 1
        socketio.emit('gomoku_state', {**board_to_dict(gomoku_board), 'message': f'AI 착수: ({BOARD_H - ai_r},{ai_c + 1})',
            'human_color': gomoku_human_color, 'session_id': session_id, 'state_seq': gomoku_state_seq,
            'win_rate': compute_human_winrate(gomoku_board)},
            room=sid, namespace='/')
    except Exception as exc: print(f'오목 AI 턴 처리 오류: {exc}')

def run_gomoku_analysis(expected_session_id=None):
    global gomoku_analysis_results, gomoku_session_id
    if (expected_session_id is not None and expected_session_id != gomoku_session_id) or not gomoku_history: return
    sid   = gomoku_session_id
    total = len(gomoku_history)
    socketio.emit('gomoku_analysis_start', {'total': total, 'session_id': sid})
    analyses = []
    for i, entry in enumerate(gomoku_history):
        if sid != gomoku_session_id: return
        socketio.emit('gomoku_analysis_progress', {'current': i+1, 'total': total,
            'move': f"({BOARD_H - entry['row']},{entry['col'] + 1})", 'session_id': sid})
        try:
            res = compute_q_values(entry['board_before'], gomoku_net, n_playout=300)
            ba  = res['best_action']; br, bc = ba // BOARD_W, ba % BOARD_W
            aq  = float(res['q_values'][entry['move']]) if not np.isnan(res['q_values'][entry['move']]) else None
            bq  = float(res['q_values'][ba])
            analyses.append({'move_num': i+1, 'row': entry['row'], 'col': entry['col'],
                'actual_q': round(aq, 3) if aq is not None else None,
                'best_q': round(bq, 3), 'best_row': br, 'best_col': bc,
                'loss': round(bq - aq, 3) if aq is not None else None,
                'v_s': round(float(res['v_s']), 3), 'is_best': entry['move'] == ba})
        except Exception as e:
            print(f'오목 분석 오류 {i+1}수: {e}')
            analyses.append({'move_num': i+1, 'row': entry['row'], 'col': entry['col'],
                'actual_q': None, 'best_q': None, 'loss': None, 'v_s': None, 'is_best': False})
    if sid != gomoku_session_id: return
    gomoku_analysis_results = analyses
    losses = [a['loss'] for a in analyses if a['loss'] is not None]
    socketio.emit('gomoku_analysis_done', {
        'analyses': analyses,
        'worst':    max((a for a in analyses if a['loss'] is not None), key=lambda a: a['loss'], default=None),
        'worst_10': sorted((a for a in analyses if a['loss'] is not None), key=lambda a: a['loss'], reverse=True)[:5],
        'avg_loss': round(sum(losses)/len(losses), 3) if losses else 0,
        'total_moves': total, 'session_id': sid,
    })
    print('✅ Gomoku 분석 완료')

@socketio.on('gomoku_request_counterfactual')
def handle_gomoku_counterfactual(data):
    global gomoku_counterfactual_cache
    if not gomoku_history or not gomoku_analysis_results:
        emit('gomoku_counterfactual_error', {'message': '오목 counterfactual 데이터를 찾지 못했습니다.'}); return
    move_num = int(data.get('move_num', -1))
    req_sid  = int(data.get('session_id', gomoku_session_id))
    if req_sid != gomoku_session_id:
        emit('gomoku_counterfactual_error', {'message': '이전 판의 요청입니다.', 'session_id': req_sid, 'move_num': move_num}); return
    cache_key = (req_sid, move_num)
    if cache_key in gomoku_counterfactual_cache:
        emit('gomoku_counterfactual_ready', gomoku_counterfactual_cache[cache_key]); return
    candidate     = next((a for a in gomoku_analysis_results if a['move_num'] == move_num), None)
    history_entry = next((h for idx, h in enumerate(gomoku_history, start=1) if idx == move_num), None)
    if candidate is None or history_entry is None:
        emit('gomoku_counterfactual_error', {'message': '선택한 착수를 찾지 못했습니다.', 'move_num': move_num}); return

    human_board = copy.deepcopy(history_entry['board_before'])
    agent_board = copy.deepcopy(history_entry['board_before'])
    hf, af      = [], []
    h_labels, a_labels = [], []
    h_outcome = a_outcome = None
    LIMIT = 10

    remaining = gomoku_history[move_num-1:]
    h_turns = 0
    for te in remaining:
        end_h, win_h = human_board.game_end()
        if end_h or h_turns >= LIMIT: break
        if te['move'] in human_board.availables:
            human_board.do_move(te['move'])
            h_labels.append(f"({BOARD_H - te['row']},{te['col'] + 1})")
            hf.append(render_gomoku_board(human_board, highlight_move=te['move'])); h_turns += 1
        end_h, win_h = human_board.game_end()
        if end_h or h_turns >= LIMIT:
            if end_h: h_outcome = human_perspective_outcome_text(win_h); break
        ai_m = te.get('ai_move')
        if ai_m is not None and ai_m in human_board.availables:
            ar, ac = divmod(ai_m, BOARD_W)
            human_board.do_move(ai_m); h_labels.append(f"({BOARD_H - ar},{ac + 1})")
            hf.append(render_gomoku_board(human_board, highlight_move=ai_m)); h_turns += 1
            end_h, win_h = human_board.game_end()
            if end_h: h_outcome = human_perspective_outcome_text(win_h); break

    best_move = candidate['best_row'] * BOARD_W + candidate['best_col']
    if best_move in agent_board.availables:
        agent_board.do_move(best_move)
        a_labels.append(f"({BOARD_H - candidate['best_row']},{candidate['best_col'] + 1})")
        af.append(render_gomoku_board(agent_board, highlight_move=best_move))
        end_a, win_a = agent_board.game_end()
        if end_a: a_outcome = human_perspective_outcome_text(win_a)
    a_turns = 1 if af else 0
    while a_turns < LIMIT:
        end_a, win_a = agent_board.game_end()
        if end_a: break
        cp   = agent_board.current_player
        na, _ = choose_gomoku_best_move(agent_board, n_playout=80)
        if na not in agent_board.availables: break
        nr, nc = divmod(na, BOARD_W)
        agent_board.do_move(na); a_labels.append(f"({BOARD_H - nr},{nc + 1})"); a_turns += 1
        end_a, win_a = agent_board.game_end()
        if end_a:
            a_outcome = human_perspective_outcome_text(win_a)
            af.append(render_gomoku_board(agent_board, highlight_move=na)); break
        bq = compute_q_values(agent_board, gomoku_net, n_playout=80)['q_values'] if cp==2 else None
        af.append(render_gomoku_board(agent_board, highlight_move=na, q_values=bq))

    if not hf: hf.append(render_gomoku_board(human_board))
    if not af: af.append(render_gomoku_board(agent_board))
    length = max(len(hf), len(af))
    while len(hf) < length: hf.append(hf[-1].copy())
    while len(af) < length: af.append(af[-1].copy())

    summary = {'move_num': move_num, 'loss': candidate['loss'],
               'actual_row': BOARD_H - history_entry['row'], 'actual_col': history_entry['col'] + 1,
               'actual_q': candidate.get('actual_q'), 'best_q': candidate.get('best_q'),
               'best_row': BOARD_H - candidate['best_row'], 'best_col': candidate['best_col'] + 1,
               'human_sequence_labels': h_labels, 'agent_sequence_labels': a_labels,
               'human_outcome': h_outcome, 'agent_outcome': a_outcome}
    feedback, fb_structured, fb_src, fb_model, fb_route = generate_feedback('gomoku', summary)
    payload = {'human_frames': encode_frames(hf), 'agent_frames': encode_frames(af),
               'summary': summary, 'feedback': feedback,
               'feedback_structured': fb_structured,
               'feedback_source': fb_src,
               'feedback_model': fb_model, 'feedback_route': fb_route,
               'session_id': req_sid, 'move_num': move_num}
    gomoku_counterfactual_cache[cache_key] = payload
    emit('gomoku_counterfactual_ready', payload)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('🚀 AI Arcade server running on http://127.0.0.1:5001')
    socketio.run(app, host='0.0.0.0', debug=False, port=5001)