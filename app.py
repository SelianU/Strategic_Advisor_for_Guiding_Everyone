import eventlet
eventlet.monkey_patch()

import sys, os, copy, pickle, base64, json, uuid
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import gymnasium as gym
import ale_py  # noqa
import cv2
import numpy as np
import torch

# ── AlphaZero (Gomoku) ───────────────────────────────────────
ALPHAZERO_DIR = os.path.join(os.path.dirname(__file__), 'AlphaZero_Gomoku')
sys.path.insert(0, ALPHAZERO_DIR)
from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy

# ── D3QN (Space Invaders) ────────────────────────────────────
from d3qn_helper import load_d3qn, get_q_values, ACTION_NAMES
from llm_feedback import generate_feedback, test_openrouter_connection, DEFAULT_MODEL, FALLBACK_MODELS

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Config (API 키 영속 저장) ─────────────────────────────────
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
            print(f"⚠️  config.json 로드 실패: {e}")

def save_config(api_key: str, model: str):
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            pass
    cfg['openrouter_api_key'] = api_key
    cfg['openrouter_model'] = model
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

load_config()

# ── Atari 환경 ───────────────────────────────────────────────
# _si_cfg     = D3QNConfig()
from collections import deque
_si_env    = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array',
                      frameskip=1, repeat_action_probability=0.0)
_si_frames = deque(maxlen=4)   # D3QN 분석용 프레임 스택

def _preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

def _get_stacked():
    return np.array(_si_frames, dtype=np.uint8)


def load_model(model_path, board_width, board_height):
    try:
        with open(model_path, 'rb') as f:
            net_params = pickle.load(f)
    except Exception:
        with open(model_path, 'rb') as f:
            net_params = pickle.load(f, encoding='bytes')
    return PolicyValueNetNumpy(board_width, board_height, net_params)


def make_policy_value_fn(policy_value_net):
    return policy_value_net.policy_value_fn


def compute_q_values(board, policy_value_net, n_playout=80):
    """Estimate action values for each legal move from policy logits and state value."""
    legal_positions = list(board.availables)
    if not legal_positions:
        return {
            'q_values': np.full(board.width * board.height, np.nan, dtype=np.float32),
            'best_action': -1,
            'v_s': 0.0,
        }

    act_probs_iter, v_s = policy_value_net.policy_value_fn(board)
    act_probs = dict(act_probs_iter)
    q_values = np.full(board.width * board.height, np.nan, dtype=np.float32)
    for action in legal_positions:
        q_values[action] = float(act_probs.get(action, 0.0))

    best_action = max(legal_positions, key=lambda a: q_values[a])
    return {
        'q_values': q_values,
        'best_action': int(best_action),
        'v_s': float(v_s),
    }

# ── D3QN 모델 ────────────────────────────────────────────────
D3QN_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 'checkpoints_v3_logs', 'best_model.pth'
)
d3qn_net = None
if os.path.exists(D3QN_MODEL_PATH):
    d3qn_net, _ = load_d3qn(D3QN_MODEL_PATH, DEVICE)
else:
    print(f"⚠️  D3QN 모델 없음: {D3QN_MODEL_PATH}")

# ── Gomoku 모델 ──────────────────────────────────────────────
BOARD_W, BOARD_H, N_IN_ROW = 8, 8, 5
gomoku_net = load_model(
    os.path.join(ALPHAZERO_DIR, 'best_policy_8_8_5.model'),
    BOARD_W, BOARD_H
)
print("✅ Gomoku 모델 로드 완료")

# ── 상태 ─────────────────────────────────────────────────────
si_episode_data  = []
si_env_ready     = False
si_stacked_state = None
si_last_rgb      = None
si_valid_entries = []
si_analysis_results = []
si_counterfactual_cache = {}
si_session_id = 0

gomoku_board  = None
gomoku_history = []
gomoku_active  = False
ai_player     = None
gomoku_analysis_results = []
gomoku_counterfactual_cache = {}
gomoku_session_id = 0
gomoku_analysis_timer = None
gomoku_ai_turn_timer = None
gomoku_state_seq = 0


def emit_gomoku_terminal_state(payload, sid):
    # Reuse the standard live-state event so the final AI move/result is rendered
    # through the exact same path as ordinary Gomoku moves.
    socketio.emit('gomoku_state', payload, room=sid, namespace='/')

SAVED_SESSIONS_DIR = os.path.join(os.path.dirname(__file__), 'saved_sessions')


def ensure_session_dir(game_name):
    path = os.path.join(SAVED_SESSIONS_DIR, game_name)
    os.makedirs(path, exist_ok=True)
    return path


def build_session_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]


def list_saved_sessions(game_name):
    game_dir = ensure_session_dir(game_name)
    sessions = []
    for session_id in os.listdir(game_dir):
        meta_path = os.path.join(game_dir, session_id, 'meta.json')
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            sessions.append({
                'id': session_id,
                'title': meta.get('title') or '이름 없음',
                'note': meta.get('note') or '',
                'saved_at': meta.get('saved_at') or '',
                'summary': meta.get('summary') or '',
            })
        except Exception:
            continue
    sessions.sort(key=lambda item: item['saved_at'], reverse=True)
    return sessions


def select_si_top5(analyses, k=5, min_gap=3):
    """전체를 k구간으로 나눠 구간별 최대 loss를 뽑되, step 간격 min_gap 이상 보장."""
    if not analyses:
        return []
    total = len(analyses)
    seg_size = max(1, total // k)
    selected = []
    for seg_idx in range(k):
        start = seg_idx * seg_size
        end = total if seg_idx == k - 1 else (seg_idx + 1) * seg_size
        candidates = sorted(analyses[start:end], key=lambda a: a['loss'], reverse=True)
        for candidate in candidates:
            if all(abs(candidate['step'] - s['step']) >= min_gap for s in selected):
                selected.append(candidate)
                break
    return sorted(selected, key=lambda a: a['loss'], reverse=True)


def si_analysis_snapshot():
    if not si_analysis_results:
        return None
    losses = [a['loss'] for a in si_analysis_results]
    avg_loss = round(float(np.mean(losses)), 3) if losses else 0.0
    worst = max(si_analysis_results, key=lambda a: a['loss']) if si_analysis_results else None
    worst_10 = select_si_top5(si_analysis_results)
    agree = round(sum(1 for a in si_analysis_results if a['is_best']) / len(si_analysis_results) * 100, 1) if si_analysis_results else 0.0
    from collections import Counter
    p_cnt = Counter(a['action_name'] for a in si_analysis_results)
    ai_cnt = Counter(a['best_action_name'] for a in si_analysis_results)
    return {
        'avg_loss': avg_loss,
        'worst': worst,
        'worst_10': worst_10,
        'total_steps': len(si_analysis_results),
        'agree_rate': agree,
        'player_actions': dict(p_cnt),
        'ai_actions': dict(ai_cnt),
    }


def serialize_si_counterfactual_cache():
    saved = []
    for (session_id, entry_index, replay_horizon), payload in si_counterfactual_cache.items():
        if session_id != si_session_id:
            continue
        item = copy.deepcopy(payload)
        item['entry_index'] = entry_index
        item['replay_horizon'] = replay_horizon
        saved.append(item)
    return saved


def gomoku_analysis_snapshot():
    if not gomoku_analysis_results:
        return None
    losses = [a['loss'] for a in gomoku_analysis_results if a['loss'] is not None]
    avg_loss = round(sum(losses) / len(losses), 3) if losses else 0
    worst = max((a for a in gomoku_analysis_results if a['loss'] is not None), key=lambda a: a['loss'], default=None)
    worst_10 = sorted((a for a in gomoku_analysis_results if a['loss'] is not None), key=lambda a: a['loss'], reverse=True)[:5]
    return {
        'analyses': gomoku_analysis_results,
        'worst': worst,
        'worst_10': worst_10,
        'avg_loss': avg_loss,
        'total_moves': len(gomoku_history),
    }


def serialize_gomoku_counterfactual_cache():
    saved = []
    for (session_id, move_num), payload in gomoku_counterfactual_cache.items():
        if session_id != gomoku_session_id:
            continue
        item = copy.deepcopy(payload)
        item['move_num'] = move_num
        saved.append(item)
    return saved


def serialize_gomoku_history():
    saved = []
    for entry in gomoku_history:
        item = {
            'move': entry['move'],
            'row': entry['row'],
            'col': entry['col'],
        }
        if 'ai_move' in entry and entry['ai_move'] is not None:
            item['ai_move'] = entry['ai_move']
        saved.append(item)
    return saved


def rebuild_gomoku_history(saved_moves):
    board = Board(width=BOARD_W, height=BOARD_H, n_in_row=N_IN_ROW)
    board.init_board(start_player=0)
    rebuilt = []
    for item in saved_moves:
        board_before = copy.deepcopy(board)
        board.do_move(item['move'])
        rebuilt_item = {
            'move': item['move'],
            'row': item['row'],
            'col': item['col'],
            'board_before': board_before,
        }
        ai_move = item.get('ai_move')
        if ai_move is not None and ai_move in board.availables:
            board.do_move(ai_move)
            rebuilt_item['ai_move'] = ai_move
        rebuilt.append(rebuilt_item)
    return board, rebuilt


def encode_frame(frame):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')


def encode_frames(frames):
    return [encode_frame(frame) for frame in frames]


def greedy_action_from_state(state):
    q_vals = get_q_values(d3qn_net, state, DEVICE)
    return int(np.argmax(q_vals)), q_vals


def make_space_env():
    return gym.make(
        'ALE/SpaceInvaders-v5',
        render_mode='rgb_array',
        frameskip=1,
        repeat_action_probability=0.0,
    )


def restore_space_env(env, snapshot):
    env.reset()
    env.unwrapped.ale.restoreSystemState(snapshot)


def compose_space_compare_frame(human_frame, agent_frame, overlay):
    h, w = human_frame.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = human_frame
    canvas[:, w:w * 2] = agent_frame
    cv2.line(canvas, (w, 0), (w, h), (80, 80, 80), 2)
    return canvas


def build_space_fallback_feedback(summary):
    human_steps = summary.get('human_reward_steps', [])
    agent_steps = summary.get('agent_reward_steps', [])
    human_steps_text = ", ".join(f"{s}프레임" for s in human_steps[:5]) if human_steps else "득점이 없었습니다"
    agent_steps_text = ", ".join(f"{s}프레임" for s in agent_steps[:5]) if agent_steps else "득점이 없었습니다"

    lines = [
        f"이 상황에서는 인간 플레이어가 {summary['human_action_name']}을 선택했지만, 에이전트의 {summary['agent_action_name']}가 더 유리했습니다. 비교 영상을 보면 에이전트 쪽은 이동보다 공격 타이밍을 먼저 살리면서 득점 흐름을 앞당겼고, 인간 쪽은 위험을 피하는 데는 성공했지만 득점으로 이어지는 기회를 더 늦게 잡았습니다.",
        f"인간 쪽 득점은 {human_steps_text}에 나왔고, 에이전트 쪽 득점은 {agent_steps_text}에 이어졌습니다. 이번 장면에서 중요한 점은 최종 점수 차이보다 에이전트가 더 이른 프레임부터 반복적으로 득점 기회를 만들었다는 점입니다.",
    ]
    return "\n\n".join(lines)


def find_gomoku_winning_line(board):
    width = board.width
    height = board.height
    states = board.states
    n = board.n_in_row
    moved = list(set(range(width * height)) - set(board.availables))
    if len(moved) < n * 2 - 1:
        return None, -1

    for move in moved:
        row = move // width
        col = move % width
        player = states.get(move, -1)
        if player == -1:
            continue

        horizontal = list(range(move, move + n))
        if col in range(width - n + 1) and len(set(states.get(i, -1) for i in horizontal)) == 1:
            return horizontal, player

        vertical = list(range(move, move + n * width, width))
        if row in range(height - n + 1) and len(set(states.get(i, -1) for i in vertical)) == 1:
            return vertical, player

        diag_down = list(range(move, move + n * (width + 1), width + 1))
        if col in range(width - n + 1) and row in range(height - n + 1) and len(set(states.get(i, -1) for i in diag_down)) == 1:
            return diag_down, player

        diag_up = list(range(move, move + n * (width - 1), width - 1))
        if col in range(n - 1, width) and row in range(height - n + 1) and len(set(states.get(i, -1) for i in diag_up)) == 1:
            return diag_up, player

    return None, -1


def human_perspective_outcome_text(winner):
    if winner == 1:
        return "WIN"
    if winner == 2:
        return "LOSE"
    if winner == -1:
        return "DRAW"
    return None


def _q_to_bgr(t):
    """t in [0,1]. Returns BGR: 0=cool blue, 0.5=green, 1=hot red."""
    if t < 0.5:
        s = t * 2
        return (int(210 - 160 * s), int(80 + 130 * s), int(30 + 20 * s))
    else:
        s = (t - 0.5) * 2
        return (int(50 - 50 * s), int(210 - 180 * s), int(50 + 205 * s))


def render_gomoku_board(board, highlight_move=None, title="", highlight_black_only=True, winning_line=None, outcome_text=None, q_values=None):
    size = 420
    margin = 36
    cell = (size - margin * 2) // (BOARD_W - 1)
    img = np.full((size + 48, size, 3), 226, dtype=np.uint8)
    img[:, :] = (226, 194, 140)
    cv2.rectangle(img, (0, 0), (size - 1, size + 47), (70, 45, 20), 2)
    if title:
        cv2.putText(img, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    for i in range(BOARD_W):
        x = margin + i * cell
        cv2.line(img, (x, margin + 12), (x, margin + 12 + cell * (BOARD_H - 1)), (80, 60, 35), 1)
    for i in range(BOARD_H):
        y = margin + 12 + i * cell
        cv2.line(img, (margin, y), (margin + cell * (BOARD_W - 1), y), (80, 60, 35), 1)
    # ── Q-value 히트맵 (빈 교차점, 돌 그리기 전) ─────────────
    if q_values is not None:
        valid = [
            (pos, float(q_values[pos])) for pos in board.availables
            if pos < len(q_values) and not np.isnan(q_values[pos]) and q_values[pos] > 1e-6
        ]
        if valid:
            valid.sort(key=lambda x: x[1], reverse=True)
            top_k = valid[:8]
            q_max = top_k[0][1]
            q_min = top_k[-1][1]
            q_range = max(q_max - q_min, 1e-8)
            heat_layer = img.copy()
            for pos, qv in top_k:
                t = (qv - q_min) / q_range
                bgr = _q_to_bgr(t)
                row, col = divmod(pos, BOARD_W)
                display_row = (BOARD_H - 1) - row
                x = margin + col * cell
                y = margin + 12 + display_row * cell
                radius = int(5 + 9 * t)
                cv2.circle(heat_layer, (x, y), radius, bgr, -1)
                cv2.circle(heat_layer, (x, y), radius, (210, 210, 210), 1)
            img = cv2.addWeighted(heat_layer, 0.75, img, 0.25, 0)
    for move, player in board.states.items():
        row, col = divmod(move, BOARD_W)
        display_row = (BOARD_H - 1) - row
        x = margin + col * cell
        y = margin + 12 + display_row * cell
        color = (30, 30, 30) if player == 1 else (240, 240, 240)
        cv2.circle(img, (x, y), 16, color, -1)
        cv2.circle(img, (x, y), 16, (60, 60, 60), 1)
    if winning_line:
        line_points = []
        for move in winning_line:
            row, col = divmod(move, BOARD_W)
            display_row = (BOARD_H - 1) - row
            x = margin + col * cell
            y = margin + 12 + display_row * cell
            line_points.append((x, y))
        line_color = (70, 220, 120) if outcome_text == "WIN" else ((80, 80, 255) if outcome_text == "LOSE" else (0, 215, 255))
        cv2.line(img, line_points[0], line_points[-1], line_color, 4, cv2.LINE_AA)
        for point in line_points:
            cv2.circle(img, point, 20, line_color, 2, cv2.LINE_AA)
    if highlight_move is not None and highlight_move != -1:
        highlight_player = board.states.get(highlight_move)
        if highlight_black_only and highlight_player != 1 and not winning_line and not outcome_text:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        row, col = divmod(highlight_move, BOARD_W)
        display_row = (BOARD_H - 1) - row
        x = margin + col * cell
        y = margin + 12 + display_row * cell
        cv2.circle(img, (x, y), 21, (0, 255, 255), 2)
    if outcome_text:
        overlay = img.copy()
        cv2.rectangle(overlay, (82, 18), (size - 82, 66), (20, 20, 20), -1)
        alpha = 0.42
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        text_color = (70, 220, 120) if outcome_text == "WIN" else ((80, 80, 255) if outcome_text == "LOSE" else (0, 215, 255))
        text_size = cv2.getTextSize(outcome_text, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)[0]
        text_x = (size - text_size[0]) // 2
        cv2.putText(img, outcome_text, (text_x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.95, text_color, 3, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def append_gomoku_terminal_hold(frames, board, winner, hold_frames=10):
    winning_line, _ = find_gomoku_winning_line(board)
    outcome_text = human_perspective_outcome_text(winner)
    terminal_frame = render_gomoku_board(
        board,
        highlight_black_only=False,
        winning_line=winning_line,
        outcome_text=outcome_text,
    )
    for _ in range(hold_frames):
        frames.append(terminal_frame.copy())


def pad_gomoku_hold(frames, hold_frames=10):
    if not frames:
        return
    frozen = frames[-1].copy()
    for _ in range(hold_frames):
        frames.append(frozen.copy())


def choose_gomoku_best_move(board, n_playout=80):
    res = compute_q_values(board, gomoku_net, n_playout=n_playout)
    return int(res['best_action']), res


def build_gomoku_fallback_feedback(summary):
    actual = f"({summary['actual_row']}, {summary['actual_col']})"
    best = f"({summary['best_row']}, {summary['best_col']})"
    human_seq = ", ".join(summary.get('human_sequence_labels', [])[:6]) or "기록 없음"
    agent_seq = ", ".join(summary.get('agent_sequence_labels', [])[:6]) or "기록 없음"
    lines = [
        f"이 장면에서는 인간 플레이어가 {actual}에 착수했지만, AI가 권장한 {best}가 더 유리했습니다. 실제 착수는 돌을 길게 이어 가거나 상대 위협을 먼저 끊어내는 힘이 다소 약했던 반면, 권장 착수는 더 높은 가치로 평가되어 이후 전개에서 주도권을 잡기 쉬운 선택이었습니다.",
        f"비교 전개를 보면 인간 쪽은 이후 수순이 {human_seq} 순으로 이어졌고, 에이전트 쪽은 {agent_seq}처럼 더 빠르게 핵심 자리를 선점하면서 열린 3목이나 4목으로 이어질 가능성을 넓혔습니다. 특히 권장 수는 한 방향만 잇는 데 그치지 않고 양쪽으로 확장될 여지를 만들거나, 상대가 먼저 위협을 만들기 전에 흐름을 끊는 데 더 적합한 수였습니다. 이번 장면에서 중요한 점은 한 수의 위치 차이가 돌의 연결, 상대 위협 차단, 다음 수를 강제하는 흐름, 이후 수순 전체의 효율을 함께 바꾼다는 점입니다.",
    ]
    return "\n\n".join(lines)




# ══ 라우트 ════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/space-invaders')
def space_invaders_page():
    return render_template(
        'space_invaders.html',
        has_d3qn=(d3qn_net is not None),
        has_openrouter=bool(os.getenv("OPENROUTER_API_KEY")),
        openrouter_model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        openrouter_fallback=", ".join(FALLBACK_MODELS),
        saved_sessions=list_saved_sessions('space_invaders'),
    )

@app.route('/gomoku')
def gomoku_page():
    return render_template(
        'gomoku.html',
        has_openrouter=bool(os.getenv("OPENROUTER_API_KEY")),
        openrouter_model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        openrouter_fallback=", ".join(FALLBACK_MODELS),
        saved_sessions=list_saved_sessions('gomoku'),
    )


@app.route('/api/settings', methods=['GET'])
def get_settings():
    api_key = os.getenv('OPENROUTER_API_KEY', '')
    masked = ('*' * (len(api_key) - 4) + api_key[-4:]) if len(api_key) > 4 else ('*' * len(api_key))
    return jsonify({
        'has_key': bool(api_key),
        'masked_key': masked,
        'model': os.getenv('OPENROUTER_MODEL', DEFAULT_MODEL),
    })

@app.route('/api/settings', methods=['POST'])
def post_settings():
    data = request.get_json(force=True) or {}
    api_key = data.get('api_key', '').strip()
    model = data.get('model', '').strip() or DEFAULT_MODEL
    if data.get('clear_key'):
        os.environ.pop('OPENROUTER_API_KEY', None)
    elif api_key:
        os.environ['OPENROUTER_API_KEY'] = api_key
    # 빈 키가 전송된 경우 기존 키를 유지 (모달 재오픈 시 입력란이 비워져도 덮어쓰지 않음)
    os.environ['OPENROUTER_MODEL'] = model
    save_config(os.getenv('OPENROUTER_API_KEY', ''), model)
    return jsonify({'ok': True})

@app.route('/api/settings/test', methods=['POST'])
def test_settings():
    ok, payload = test_openrouter_connection()
    if isinstance(payload, str):
        return jsonify({'ok': ok, 'message': payload})
    summary = payload.get('summary', '') if isinstance(payload, dict) else str(payload)
    return jsonify({'ok': ok, 'message': summary})


@socketio.on('llm_test')
def handle_llm_test():
    ok, payload = test_openrouter_connection()
    if isinstance(payload, str):
        payload = {'summary': payload, 'results': []}
    emit('llm_test_result', {'ok': ok, **payload})


@socketio.on('si_list_sessions')
def handle_si_list_sessions():
    emit('si_sessions_list', {'sessions': list_saved_sessions('space_invaders')})


@socketio.on('gomoku_list_sessions')
def handle_gomoku_list_sessions():
    emit('gomoku_sessions_list', {'sessions': list_saved_sessions('gomoku')})


@socketio.on('si_save_session')
def handle_si_save_session(data):
    if not si_episode_data or not si_analysis_results:
        emit('si_session_saved', {'ok': False, 'message': '저장할 Space Invaders 분석 결과가 없습니다.'})
        return
    title = (data.get('title') or '').strip() or 'Space Invaders 기록'
    note = (data.get('note') or '').strip()
    session_id = build_session_id()
    session_dir = os.path.join(ensure_session_dir('space_invaders'), session_id)
    os.makedirs(session_dir, exist_ok=True)
    basic_report = basic_analyze(si_episode_data)
    analysis_snapshot = si_analysis_snapshot()
    metadata = {
        'game': 'space_invaders',
        'title': title,
        'note': note,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'summary': f"점수 {basic_report['total_reward']} · 스텝 {basic_report['total_steps']}",
        'basic_report': basic_report,
    }
    with open(os.path.join(session_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(session_dir, 'session.pkl'), 'wb') as f:
        pickle.dump({
            'episode_data': si_episode_data,
            'analysis_results': si_analysis_results,
            'analysis_snapshot': analysis_snapshot,
            'cached_counterfactuals': serialize_si_counterfactual_cache(),
        }, f)
    emit('si_session_saved', {'ok': True, 'message': '기록을 저장했습니다.', 'sessions': list_saved_sessions('space_invaders')})


@socketio.on('gomoku_save_session')
def handle_gomoku_save_session(data):
    if not gomoku_history or not gomoku_analysis_results:
        emit('gomoku_session_saved', {'ok': False, 'message': '저장할 Gomoku 분석 결과가 없습니다.'})
        return
    title = (data.get('title') or '').strip() or 'Gomoku 기록'
    note = (data.get('note') or '').strip()
    session_id = build_session_id()
    session_dir = os.path.join(ensure_session_dir('gomoku'), session_id)
    os.makedirs(session_dir, exist_ok=True)
    analysis_snapshot = gomoku_analysis_snapshot()
    metadata = {
        'game': 'gomoku',
        'title': title,
        'note': note,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'summary': f"총 {len(gomoku_history)}수 · 평균 손실 {analysis_snapshot['avg_loss']}",
    }
    with open(os.path.join(session_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(session_dir, 'session.pkl'), 'wb') as f:
        pickle.dump({
            'history': serialize_gomoku_history(),
            'analysis_results': gomoku_analysis_results,
            'analysis_snapshot': analysis_snapshot,
            'cached_counterfactuals': serialize_gomoku_counterfactual_cache(),
        }, f)
    emit('gomoku_session_saved', {'ok': True, 'message': '기록을 저장했습니다.', 'sessions': list_saved_sessions('gomoku')})


@socketio.on('si_load_session')
def handle_si_load_session(data):
    global si_episode_data, si_analysis_results, si_valid_entries, si_counterfactual_cache, si_session_id
    session_id = data.get('session_id')
    session_dir = os.path.join(ensure_session_dir('space_invaders'), session_id)
    meta_path = os.path.join(session_dir, 'meta.json')
    data_path = os.path.join(session_dir, 'session.pkl')
    if not os.path.isfile(meta_path) or not os.path.isfile(data_path):
        emit('si_session_loaded', {'ok': False, 'message': '불러올 기록을 찾지 못했습니다.'})
        return
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(data_path, 'rb') as f:
        payload = pickle.load(f)
    si_session_id += 1
    si_episode_data = payload['episode_data']
    si_analysis_results = payload['analysis_results']
    si_valid_entries = [entry for entry in si_episode_data if entry.get('action') is not None]
    si_counterfactual_cache = {}
    cached_counterfactuals = payload.get('cached_counterfactuals') or []
    for item in cached_counterfactuals:
        entry_index = int(item.get('entry_index', -1))
        replay_horizon = int(item.get('replay_horizon', 60))
        restored = copy.deepcopy(item)
        restored['session_id'] = si_session_id
        si_counterfactual_cache[(si_session_id, entry_index, replay_horizon)] = restored
    emit('si_session_loaded', {
        'ok': True,
        'meta': meta,
        'basic': meta.get('basic_report', {}),
        'analysis': payload.get('analysis_snapshot') or si_analysis_snapshot(),
        'cached_counterfactuals': cached_counterfactuals,
        'session_id': si_session_id,
    })


@socketio.on('gomoku_load_session')
def handle_gomoku_load_session(data):
    global gomoku_board, gomoku_history, gomoku_analysis_results, gomoku_counterfactual_cache, gomoku_session_id, gomoku_active
    session_id = data.get('session_id')
    session_dir = os.path.join(ensure_session_dir('gomoku'), session_id)
    meta_path = os.path.join(session_dir, 'meta.json')
    data_path = os.path.join(session_dir, 'session.pkl')
    if not os.path.isfile(meta_path) or not os.path.isfile(data_path):
        emit('gomoku_session_loaded', {'ok': False, 'message': '불러올 기록을 찾지 못했습니다.'})
        return
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(data_path, 'rb') as f:
        payload = pickle.load(f)
    gomoku_session_id += 1
    gomoku_board, gomoku_history = rebuild_gomoku_history(payload['history'])
    gomoku_analysis_results = payload['analysis_results']
    gomoku_counterfactual_cache = {}
    cached_counterfactuals = payload.get('cached_counterfactuals') or []
    for item in cached_counterfactuals:
        move_num = int(item.get('move_num', -1))
        restored = copy.deepcopy(item)
        restored['session_id'] = gomoku_session_id
        gomoku_counterfactual_cache[(gomoku_session_id, move_num)] = restored
    gomoku_active = False
    emit('gomoku_session_loaded', {
        'ok': True,
        'meta': meta,
        'analysis': payload.get('analysis_snapshot') or gomoku_analysis_snapshot(),
        'board': board_to_dict(gomoku_board),
        'cached_counterfactuals': cached_counterfactuals,
        'session_id': gomoku_session_id,
    })


@socketio.on('si_delete_session')
def handle_si_delete_session(data):
    session_id = data.get('session_id')
    session_dir = os.path.join(ensure_session_dir('space_invaders'), session_id)
    if os.path.isdir(session_dir):
        import shutil
        shutil.rmtree(session_dir)
    emit('si_session_deleted', {'ok': True, 'message': '기록을 삭제했습니다.', 'sessions': list_saved_sessions('space_invaders')})


@socketio.on('gomoku_delete_session')
def handle_gomoku_delete_session(data):
    session_id = data.get('session_id')
    session_dir = os.path.join(ensure_session_dir('gomoku'), session_id)
    if os.path.isdir(session_dir):
        import shutil
        shutil.rmtree(session_dir)
    emit('gomoku_session_deleted', {'ok': True, 'message': '기록을 삭제했습니다.', 'sessions': list_saved_sessions('gomoku')})


# ══ Space Invaders ════════════════════════════════════════════

def basic_analyze(data):
    acts   = [d['action'] for d in data if d['action'] is not None]
    total  = len(acts) or 1
    reward = sum(d['reward'] for d in data)
    left, right, fire = acts.count(3), acts.count(2), acts.count(1)
    agg = fire / total * 100
    if agg > 40:   style, advice = '공격형', '발사 빈도가 높습니다. 이동으로 외계인을 유인해도 좋습니다.'
    elif agg < 15: style, advice = '수비형', '발사 빈도가 낮습니다. 더 공격적으로 처치해도 좋습니다.'
    else:          style, advice = '균형형', '발사와 이동의 균형이 좋습니다.'
    bias = '좌편향' if left > right * 1.3 else ('우편향' if right > left * 1.3 else '균형')
    return {
        'play_style': style, 'total_reward': int(reward), 'total_steps': total,
        'left_ratio': f'{left/total*100:.1f}%', 'right_ratio': f'{right/total*100:.1f}%',
        'fire_ratio': f'{agg:.1f}%', 'move_bias': bias, 'ai_feedback': advice,
    }

@socketio.on('si_start')
def handle_si_start():
    global si_episode_data, si_env_ready, si_stacked_state, si_last_rgb, si_valid_entries, si_analysis_results, si_counterfactual_cache, si_session_id
    si_session_id += 1
    si_env_ready = False
    si_episode_data = []
    si_valid_entries = []
    si_analysis_results = []
    si_counterfactual_cache = {}

    obs_raw, _ = _si_env.reset()
    proc = _preprocess(obs_raw)
    for _ in range(4):
        _si_frames.append(proc)
    si_last_rgb = obs_raw.copy()

    si_episode_data.append({
        'step': 0, 'rgb': obs_raw,
        'stacked_state': _get_stacked().copy(),
        'action': None, 'reward': 0.0, 'done': False
    })
    si_env_ready = True
    emit('si_frame', {'image': encode_frame(obs_raw), 'done': False, 'score': 0.0, 'session_id': si_session_id})

@socketio.on('si_action')
def handle_si_action(data):
    global si_episode_data, si_env_ready, si_last_rgb
    if not si_env_ready:
        return

    action = int(data.get('action', 0))
    pre_snapshot = _si_env.unwrapped.ale.cloneSystemState()
    pre_stacked_state = _get_stacked().copy()
    pre_rgb = si_last_rgb.copy() if si_last_rgb is not None else _si_env.render()
    obs_raw, reward, terminated, truncated, _ = _si_env.step(action)
    done = terminated or truncated
    si_last_rgb = obs_raw.copy()
    current_score = float(sum(d['reward'] for d in si_episode_data if d.get('action') is not None) + float(reward))

    _si_frames.append(_preprocess(obs_raw))

    si_episode_data.append({
        'step': len(si_episode_data),
        'pre_snapshot': pre_snapshot,
        'pre_stacked_state': pre_stacked_state,
        'pre_rgb': pre_rgb,
        'rgb': obs_raw,
        'stacked_state': _get_stacked().copy(),
        'action': action,
        'reward': float(reward),
        'done': done
    })

    if done:
        si_env_ready = False
        basic = basic_analyze(si_episode_data)
        emit('si_over', {**basic, 'has_d3qn': (d3qn_net is not None)})
        if d3qn_net is not None:
            socketio.start_background_task(run_si_analysis)
    else:
        emit('si_frame', {'image': encode_frame(obs_raw), 'done': False, 'score': current_score})

def run_si_analysis():
    global si_valid_entries, si_analysis_results, si_session_id
    valid = [d for d in si_episode_data if d['action'] is not None]
    if not valid:
        return
    analysis_session_id = si_session_id
    si_valid_entries = valid
    total = len(valid)
    socketio.emit('si_analysis_start', {'total': total, 'session_id': analysis_session_id})

    analyses = []
    for i, entry in enumerate(valid):
        if analysis_session_id != si_session_id:
            return
        if i % 50 == 0:
                socketio.emit('si_analysis_progress',
                          {'current': i, 'total': total, 'pct': round(i/total*100), 'session_id': analysis_session_id})
        q_vals   = get_q_values(d3qn_net, entry['pre_stacked_state'], DEVICE)
        action   = entry['action']
        best_act = int(np.argmax(q_vals))
        player_q = float(q_vals[action])
        best_q   = float(q_vals[best_act])
        analyses.append({
            'step': entry['step'], 'action': action, 'entry_index': i,
            'action_name':      ACTION_NAMES.get(action, str(action)),
            'best_action':      best_act,
            'best_action_name': ACTION_NAMES.get(best_act, str(best_act)),
            'q_values':         [round(float(v), 3) for v in q_vals],
            'player_q':         round(player_q, 3),
            'best_q':           round(best_q, 3),
            'loss':             round(best_q - player_q, 3),
            'reward':           entry['reward'],
            'is_best':          (action == best_act),
        })

    losses    = [a['loss'] for a in analyses]
    avg_loss  = round(float(np.mean(losses)), 3)
    worst     = max(analyses, key=lambda a: a['loss'])
    worst_10  = select_si_top5(analyses)
    agree     = round(sum(1 for a in analyses if a['is_best']) / len(analyses) * 100, 1)
    si_analysis_results = analyses

    from collections import Counter
    p_cnt  = Counter(a['action_name']      for a in analyses)
    ai_cnt = Counter(a['best_action_name'] for a in analyses)

    socketio.emit('si_analysis_done', {
        'avg_loss': avg_loss, 'worst': worst, 'worst_10': worst_10,
        'total_steps': total, 'agree_rate': agree,
        'player_actions': dict(p_cnt), 'ai_actions': dict(ai_cnt),
        'session_id': analysis_session_id,
    })
    print(f"✅ SI 분석 완료 avg_loss={avg_loss:.3f} agree={agree}%")


@socketio.on('si_request_counterfactual')
def handle_si_counterfactual(data):
    global si_counterfactual_cache
    if d3qn_net is None or not si_valid_entries:
        emit('si_counterfactual_error', {'message': 'Space Invaders counterfactual 데이터를 찾지 못했습니다.'})
        return

    entry_index = int(data.get('entry_index', -1))
    replay_horizon = int(data.get('horizon', 60))
    request_session_id = int(data.get('session_id', si_session_id))
    if request_session_id != si_session_id:
        emit('si_counterfactual_error', {'message': '이전 판의 비교 요청이라 취소되었습니다.', 'session_id': request_session_id, 'entry_index': entry_index})
        return
    cache_key = (request_session_id, entry_index, replay_horizon)
    if cache_key in si_counterfactual_cache:
        emit('si_counterfactual_ready', si_counterfactual_cache[cache_key])
        return
    if entry_index < 0 or entry_index >= len(si_valid_entries):
        emit('si_counterfactual_error', {'message': '선택한 후보를 찾지 못했습니다.', 'entry_index': entry_index})
        return

    entry = si_valid_entries[entry_index]
    q_vals = get_q_values(d3qn_net, entry['pre_stacked_state'], DEVICE)
    best_action = int(np.argmax(q_vals))

    human_env = make_space_env()
    agent_env = make_space_env()
    try:
        restore_space_env(human_env, entry['pre_snapshot'])
        restore_space_env(agent_env, entry['pre_snapshot'])

        human_frames = deque(entry['pre_stacked_state'], maxlen=4)
        agent_frames = deque(entry['pre_stacked_state'], maxlen=4)
        human_reward_steps = []
        agent_reward_steps = []
        human_first_reward_step = None
        agent_first_reward_step = None
        human_score = 0.0
        agent_score = 0.0
        rendered_frames = []
        human_actions_log = []
        agent_actions_log = []
        human_done = False
        agent_done = False
        human_frame = entry['pre_rgb'].copy()
        agent_frame = entry['pre_rgb'].copy()

        for offset in range(replay_horizon):
            human_action = si_valid_entries[entry_index + offset]['action'] if entry_index + offset < len(si_valid_entries) else 0
            agent_action = best_action if offset == 0 else greedy_action_from_state(np.array(agent_frames, dtype=np.uint8))[0]
            human_actions_log.append(int(human_action))
            agent_actions_log.append(int(agent_action))

            if not human_done:
                obs_h, reward_h, term_h, trunc_h, _ = human_env.step(human_action)
                human_frame = obs_h
                human_frames.append(_preprocess(obs_h))
                human_score += float(reward_h)
                if reward_h > 0:
                    human_reward_steps.append(offset + 1)
                    if human_first_reward_step is None:
                        human_first_reward_step = offset + 1
                human_done = term_h or trunc_h

            if not agent_done:
                obs_a, reward_a, term_a, trunc_a, _ = agent_env.step(agent_action)
                agent_frame = obs_a
                agent_frames.append(_preprocess(obs_a))
                agent_score += float(reward_a)
                if reward_a > 0:
                    agent_reward_steps.append(offset + 1)
                    if agent_first_reward_step is None:
                        agent_first_reward_step = offset + 1
                agent_done = term_a or trunc_a

            rendered_frames.append(compose_space_compare_frame(human_frame, agent_frame, {}))

        summary = {
            'step': entry['step'],
            'loss': round(float(q_vals[best_action] - q_vals[entry['action']]), 3),
            'human_action_name': ACTION_NAMES.get(entry['action'], str(entry['action'])),
            'agent_action_name': ACTION_NAMES.get(best_action, str(best_action)),
            'human_q': round(float(q_vals[entry['action']]), 4),
            'agent_q': round(float(q_vals[best_action]), 4),
            'gap': round(float(q_vals[best_action] - q_vals[entry['action']]), 4),
            'human_reward_steps': human_reward_steps,
            'agent_reward_steps': agent_reward_steps,
            'human_first_reward_step': human_first_reward_step,
            'agent_first_reward_step': agent_first_reward_step,
            'human_score_delta': round(human_score, 1),
            'agent_score_delta': round(agent_score, 1),
            'human_done': human_done,
            'agent_done': agent_done,
            'replay_horizon': replay_horizon,
        }
        feedback, feedback_source, feedback_model, feedback_route = generate_feedback("space_invaders", summary)
        payload = {
            'frames': encode_frames(rendered_frames),
            'human_actions': human_actions_log,
            'agent_actions': agent_actions_log,
            'summary': summary,
            'feedback': feedback,
            'feedback_source': feedback_source,
            'feedback_model': feedback_model,
            'feedback_route': feedback_route,
            'session_id': request_session_id,
            'entry_index': entry_index,
        }
    finally:
        human_env.close()
        agent_env.close()
    si_counterfactual_cache[cache_key] = payload
    emit('si_counterfactual_ready', payload)


# ══ Gomoku ════════════════════════════════════════════════════

def board_to_dict(board):
    cells = [int(board.states.get(r * BOARD_W + c, 0))
             for r in range(BOARD_H) for c in range(BOARD_W)]
    return {'cells': cells, 'current_player': int(board.current_player),
            'last_move': int(board.last_move) if board.last_move != -1 else -1,
            'width': BOARD_W, 'height': BOARD_H}


def schedule_gomoku_analysis(session_id):
    global gomoku_analysis_timer
    if gomoku_analysis_timer is not None:
        try:
            gomoku_analysis_timer.cancel()
        except Exception:
            pass
    gomoku_analysis_timer = eventlet.spawn_after(3.2, run_gomoku_analysis, session_id)

@socketio.on('gomoku_start')
def handle_gomoku_start():
    global gomoku_board, gomoku_history, gomoku_active, ai_player, gomoku_analysis_results, gomoku_counterfactual_cache, gomoku_session_id, gomoku_analysis_timer, gomoku_ai_turn_timer, gomoku_state_seq
    if gomoku_analysis_timer is not None:
        try:
            gomoku_analysis_timer.cancel()
        except Exception:
            pass
        gomoku_analysis_timer = None
    if gomoku_ai_turn_timer is not None:
        try:
            gomoku_ai_turn_timer.cancel()
        except Exception:
            pass
        gomoku_ai_turn_timer = None
    gomoku_session_id += 1
    gomoku_board = Board(width=BOARD_W, height=BOARD_H, n_in_row=N_IN_ROW)
    gomoku_board.init_board(start_player=0)
    gomoku_history = []
    gomoku_analysis_results = []
    gomoku_counterfactual_cache = {}
    gomoku_state_seq = 0
    gomoku_active  = True
    ai_player = MCTSPlayer(make_policy_value_fn(gomoku_net), c_puct=5,
                           n_playout=400, is_selfplay=0)
    gomoku_state_seq += 1
    emit('gomoku_state', {
        **board_to_dict(gomoku_board),
        'message': '당신은 흑(●)입니다!',
        'session_id': gomoku_session_id,
        'state_seq': gomoku_state_seq,
    })

@socketio.on('gomoku_move')
def handle_gomoku_move(data):
    global gomoku_board, gomoku_history, gomoku_active, ai_player, gomoku_ai_turn_timer, gomoku_state_seq
    if not gomoku_active:
        return
    sid = request.sid
    row, col = data.get('row'), data.get('col')
    move = row * BOARD_W + col
    if move not in gomoku_board.availables:
        emit('gomoku_error', {'message': '이미 돌이 놓인 위치입니다.'}, to=sid)
        return

    board_before = copy.deepcopy(gomoku_board)
    gomoku_board.do_move(move)
    gomoku_history.append({'move': move, 'row': row, 'col': col,
                           'board_before': board_before})

    end, winner = gomoku_board.game_end()
    if end:
        gomoku_active = False
        r = '흑(●) 승리!' if winner==1 else ('백(○) 승리!' if winner==2 else '무승부')
        winning_line, _ = find_gomoku_winning_line(gomoku_board) if winner in (1, 2) else (None, -1)
        outcome_text = human_perspective_outcome_text(winner)
        emit_gomoku_terminal_state({
            **board_to_dict(gomoku_board),
            'message': f'게임 종료 — {r}',
            'game_over': True,
            'winner': int(winner),
            'winning_line': winning_line,
            'outcome_text': outcome_text,
            'session_id': gomoku_session_id,
            'state_seq': gomoku_state_seq + 1,
        }, sid)
        gomoku_state_seq += 1
        schedule_gomoku_analysis(gomoku_session_id)
        return

    gomoku_state_seq += 1
    emit('gomoku_state', {
        **board_to_dict(gomoku_board),
        'message': f'당신의 착수: ({row},{col}) — AI 생각 중...',
        'session_id': gomoku_session_id,
        'state_seq': gomoku_state_seq,
    }, to=sid)
    # Flush the human move to the client first, then show the AI response.
    eventlet.sleep(0)
    eventlet.sleep(0.55)
    process_gomoku_ai_turn(gomoku_session_id, len(gomoku_history) - 1, sid)


def process_gomoku_ai_turn(session_id, history_index, sid):
    global gomoku_board, gomoku_history, gomoku_active, ai_player, gomoku_ai_turn_timer, gomoku_state_seq
    try:
        if session_id != gomoku_session_id or not gomoku_active or gomoku_board is None:
            return
        if history_index >= len(gomoku_history):
            return

        ai_move = ai_player.get_action(gomoku_board, temp=1e-3)
        ai_r, ai_c = int(ai_move // BOARD_W), int(ai_move % BOARD_W)
        gomoku_board.do_move(ai_move)
        gomoku_history[history_index]['ai_move'] = ai_move
        print(f"Gomoku AI move emitted: session={session_id}, move=({ai_r},{ai_c})")

        end2, winner2 = gomoku_board.game_end()
        if end2:
            gomoku_active = False
            r = '흑(●) 승리!' if winner2 == 1 else ('백(○) 승리!' if winner2 == 2 else '무승부')
            winning_line, _ = find_gomoku_winning_line(gomoku_board) if winner2 in (1, 2) else (None, -1)
            outcome_text = human_perspective_outcome_text(winner2)
            print(f"Gomoku terminal emit: session={session_id}, winner={winner2}")
            emit_gomoku_terminal_state({
                **board_to_dict(gomoku_board),
                'message': f'AI 착수: ({ai_r},{ai_c}) — {r}',
                'game_over': True,
                'winner': int(winner2),
                'winning_line': winning_line,
                'outcome_text': outcome_text,
                'session_id': session_id,
                'state_seq': gomoku_state_seq + 1,
            }, sid)
            gomoku_state_seq += 1
            schedule_gomoku_analysis(session_id)
            return

        gomoku_state_seq += 1
        socketio.emit('gomoku_state', {
            **board_to_dict(gomoku_board),
            'message': f'AI 착수: ({ai_r},{ai_c})',
            'session_id': session_id,
            'state_seq': gomoku_state_seq,
        }, room=sid, namespace='/')
    except Exception as exc:
        print(f"오목 AI 턴 처리 오류: {exc}")

def run_gomoku_analysis(expected_session_id=None):
    global gomoku_analysis_results, gomoku_session_id
    if expected_session_id is not None and expected_session_id != gomoku_session_id:
        return
    if not gomoku_history:
        return
    analysis_session_id = gomoku_session_id
    total = len(gomoku_history)
    socketio.emit('gomoku_analysis_start', {'total': total, 'session_id': analysis_session_id})
    analyses = []
    for i, entry in enumerate(gomoku_history):
        if analysis_session_id != gomoku_session_id:
            return
        socketio.emit('gomoku_analysis_progress',
                      {'current': i+1, 'total': total, 'move': f"({entry['row']},{entry['col']})", 'session_id': analysis_session_id})
        try:
            res = compute_q_values(entry['board_before'], gomoku_net, n_playout=300)
            ba  = res['best_action']
            br, bc = ba // BOARD_W, ba % BOARD_W
            aq = float(res['q_values'][entry['move']]) \
                 if not np.isnan(res['q_values'][entry['move']]) else None
            bq = float(res['q_values'][ba])
            analyses.append({
                'move_num': i+1, 'row': entry['row'], 'col': entry['col'],
                'actual_q': round(aq, 3) if aq is not None else None,
                'best_q': round(bq, 3), 'best_row': br, 'best_col': bc,
                'loss': round(bq - aq, 3) if aq is not None else None,
                'v_s': round(float(res['v_s']), 3),
                'is_best': (entry['move'] == ba),
            })
        except Exception as e:
            print(f"오목 분석 오류 {i+1}수: {e}")
            analyses.append({'move_num': i+1, 'row': entry['row'], 'col': entry['col'],
                             'actual_q': None, 'best_q': None, 'loss': None,
                             'v_s': None, 'is_best': False})

    losses   = [a['loss'] for a in analyses if a['loss'] is not None]
    avg_loss = round(sum(losses)/len(losses), 3) if losses else 0
    worst    = max((a for a in analyses if a['loss'] is not None),
                   key=lambda a: a['loss'], default=None)
    worst_10 = sorted((a for a in analyses if a['loss'] is not None), key=lambda a: a['loss'], reverse=True)[:5]
    if analysis_session_id != gomoku_session_id:
        return
    gomoku_analysis_results = analyses
    socketio.emit('gomoku_analysis_done', {
        'analyses': analyses, 'worst': worst, 'worst_10': worst_10,
        'avg_loss': avg_loss, 'total_moves': total,
        'session_id': analysis_session_id,
    })
    print("✅ Gomoku 분석 완료")


@socketio.on('gomoku_request_counterfactual')
def handle_gomoku_counterfactual(data):
    global gomoku_counterfactual_cache, gomoku_session_id
    if not gomoku_history or not gomoku_analysis_results:
        emit('gomoku_counterfactual_error', {'message': '오목 counterfactual 데이터를 찾지 못했습니다.'})
        return

    move_num = int(data.get('move_num', -1))
    request_session_id = int(data.get('session_id', gomoku_session_id))
    if request_session_id != gomoku_session_id:
        emit('gomoku_counterfactual_error', {'message': '이전 판의 비교 요청이라 취소되었습니다.', 'session_id': request_session_id, 'move_num': move_num})
        return
    cache_key = (request_session_id, move_num)
    if cache_key in gomoku_counterfactual_cache:
        emit('gomoku_counterfactual_ready', gomoku_counterfactual_cache[cache_key])
        return
    candidate = next((a for a in gomoku_analysis_results if a['move_num'] == move_num), None)
    history_entry = next((h for idx, h in enumerate(gomoku_history, start=1) if idx == move_num), None)
    if candidate is None or history_entry is None:
        emit('gomoku_counterfactual_error', {'message': '선택한 착수를 찾지 못했습니다.', 'move_num': move_num})
        return
    human_board = copy.deepcopy(history_entry['board_before'])
    agent_board = copy.deepcopy(history_entry['board_before'])
    human_frames = []
    agent_frames = []
    human_sequence_labels = []
    agent_sequence_labels = []
    human_outcome = None
    agent_outcome = None

    replay_turn_limit = 10

    remaining_history = gomoku_history[move_num - 1:]
    human_turns = 0
    for turn_entry in remaining_history:
        end_h, winner_h = human_board.game_end()
        if end_h or human_turns >= replay_turn_limit:
            break

        human_move = turn_entry['move']
        if human_move in human_board.availables:
            human_board.do_move(human_move)
            human_sequence_labels.append(f"({turn_entry['row']}, {turn_entry['col']})")
            human_frames.append(render_gomoku_board(human_board, highlight_move=human_move))
            human_turns += 1

        end_h, winner_h = human_board.game_end()
        if end_h or human_turns >= replay_turn_limit:
            if end_h:
                human_outcome = human_perspective_outcome_text(winner_h)
            break

        actual_ai_move = turn_entry.get('ai_move')
        if actual_ai_move is not None and actual_ai_move in human_board.availables:
            ai_r, ai_c = divmod(actual_ai_move, BOARD_W)
            human_board.do_move(actual_ai_move)
            human_sequence_labels.append(f"({ai_r}, {ai_c})")
            human_frames.append(render_gomoku_board(human_board, highlight_move=actual_ai_move))
            human_turns += 1
            end_h, winner_h = human_board.game_end()
            if end_h:
                human_outcome = human_perspective_outcome_text(winner_h)
                break

    best_move = candidate['best_row'] * BOARD_W + candidate['best_col']
    if best_move in agent_board.availables:
        # 첫 흑 착수 프레임 (히트맵 없음 — 직전 백 프레임이 없어 기준 없음)
        agent_board.do_move(best_move)
        agent_sequence_labels.append(f"({candidate['best_row']}, {candidate['best_col']})")
        agent_frames.append(render_gomoku_board(
            agent_board, highlight_move=best_move))
        end_a, winner_a = agent_board.game_end()
        if end_a:
            agent_outcome = human_perspective_outcome_text(winner_a)

    agent_turns = 1 if agent_frames else 0
    while agent_turns < replay_turn_limit:
        end_a, winner_a = agent_board.game_end()
        if end_a:
            break
        current_player = agent_board.current_player
        next_a, _ = choose_gomoku_best_move(agent_board, n_playout=80)
        if next_a not in agent_board.availables:
            break
        next_r, next_c = divmod(next_a, BOARD_W)
        agent_board.do_move(next_a)
        agent_sequence_labels.append(f"({next_r}, {next_c})")
        agent_turns += 1
        end_a, winner_a = agent_board.game_end()
        if end_a:
            # 게임 종료 프레임: 히트맵 없음
            agent_outcome = human_perspective_outcome_text(winner_a)
            agent_frames.append(render_gomoku_board(
                agent_board, highlight_move=next_a))
            break
        if current_player == 2:
            # 백 착수 직후 → 흑 차례 상태에서 Q-value 계산해서 백 프레임에 히트맵 합침
            black_q = compute_q_values(agent_board, gomoku_net, n_playout=80)['q_values']
            agent_frames.append(render_gomoku_board(
                agent_board, highlight_move=next_a, q_values=black_q))
        else:
            # 흑 착수 프레임: 히트맵 없음
            agent_frames.append(render_gomoku_board(
                agent_board, highlight_move=next_a))

    if not human_frames:
        human_frames.append(render_gomoku_board(human_board))
    if not agent_frames:
        agent_frames.append(render_gomoku_board(agent_board))

    # 양쪽 프레임 수 맞추기 (자동 재생 동기화용)
    length = max(len(human_frames), len(agent_frames))
    while len(human_frames) < length:
        human_frames.append(human_frames[-1].copy())
    while len(agent_frames) < length:
        agent_frames.append(agent_frames[-1].copy())

    summary = {
        'move_num': move_num,
        'loss': candidate['loss'],
        'actual_row': history_entry['row'],
        'actual_col': history_entry['col'],
        'actual_q': candidate.get('actual_q'),
        'best_q': candidate.get('best_q'),
        'best_row': candidate['best_row'],
        'best_col': candidate['best_col'],
        'human_sequence_labels': human_sequence_labels,
        'agent_sequence_labels': agent_sequence_labels,
        'human_outcome': human_outcome,
        'agent_outcome': agent_outcome,
    }
    feedback, feedback_source, feedback_model, feedback_route = generate_feedback("gomoku", summary)
    payload = {
        'human_frames': encode_frames(human_frames),
        'agent_frames': encode_frames(agent_frames),
        'summary': summary,
        'feedback': feedback,
        'feedback_source': feedback_source,
        'feedback_model': feedback_model,
        'feedback_route': feedback_route,
        'session_id': request_session_id,
        'move_num': move_num,
    }
    gomoku_counterfactual_cache[cache_key] = payload
    emit('gomoku_counterfactual_ready', payload)


if __name__ == '__main__':
    print("🚀 AI Arcade server running on http://127.0.0.1:5001")
    socketio.run(app, host='0.0.0.0', debug=False, port=5001)
