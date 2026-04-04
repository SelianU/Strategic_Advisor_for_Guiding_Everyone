import eventlet
eventlet.monkey_patch()

import sys, os, copy, pickle, base64
from datetime import datetime

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import gymnasium as gym
import ale_py  # noqa
import cv2
import numpy as np

# ── AlphaZero (Gomoku) ───────────────────────────────────────
ALPHAZERO_DIR = os.path.join(os.path.dirname(__file__), 'AlphaZero_Gomoku')
sys.path.insert(0, ALPHAZERO_DIR)
from game import Board
from mcts_alphaZero import MCTSPlayer
from analyze_qvalue import load_model, make_policy_value_fn, compute_q_values

# ── D3QN (Space Invaders) ────────────────────────────────────
from d3qn_helper import load_d3qn, get_q_values, ACTION_NAMES

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

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

# ── D3QN 모델 ────────────────────────────────────────────────
D3QN_MODEL_PATH = os.path.join(os.path.dirname(__file__),
                               'checkpoints_v3_logs', 'best_model.pth')
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

gomoku_board  = None
gomoku_history = []
gomoku_active  = False
ai_player     = None


def encode_frame(frame):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')


# ══ 라우트 ════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/space-invaders')
def space_invaders_page():
    return render_template('space_invaders.html', has_d3qn=(d3qn_net is not None))

@app.route('/gomoku')
def gomoku_page():
    return render_template('gomoku.html')


# ══ Space Invaders ════════════════════════════════════════════

def basic_analyze(data):
    acts   = [d['action'] for d in data if d['action'] is not None]
    total  = len(acts) or 1
    reward = sum(d['reward'] for d in data)
    left, right, fire = acts.count(3), acts.count(2), acts.count(1)
    agg = fire / total * 100
    if agg > 40:   style, advice = '공격형 🔥', '발사 빈도가 높습니다. 이동으로 외계인을 유인하세요.'
    elif agg < 15: style, advice = '수비형 🛡️', '발사 빈도가 낮습니다. 더 공격적으로 처치하세요.'
    else:          style, advice = '균형형 ⚖️', '발사와 이동의 균형이 좋습니다!'
    bias = '좌편향' if left > right * 1.3 else ('우편향' if right > left * 1.3 else '균형')
    return {
        'play_style': style, 'total_reward': int(reward), 'total_steps': total,
        'left_ratio': f'{left/total*100:.1f}%', 'right_ratio': f'{right/total*100:.1f}%',
        'fire_ratio': f'{agg:.1f}%', 'move_bias': bias, 'ai_feedback': advice,
    }

@socketio.on('si_start')
def handle_si_start():
    global si_episode_data, si_env_ready, si_stacked_state
    si_env_ready = False
    si_episode_data = []

    obs_raw, _ = _si_env.reset()
    proc = _preprocess(obs_raw)
    for _ in range(4):
        _si_frames.append(proc)

    si_episode_data.append({
        'step': 0, 'rgb': obs_raw,
        'stacked_state': _get_stacked().copy(),
        'action': None, 'reward': 0.0, 'done': False
    })
    si_env_ready = True
    emit('si_frame', {'image': encode_frame(obs_raw), 'done': False})

@socketio.on('si_action')
def handle_si_action(data):
    global si_episode_data, si_env_ready
    if not si_env_ready:
        return

    action = int(data.get('action', 0))
    obs_raw, reward, terminated, truncated, _ = _si_env.step(action)
    done = terminated or truncated

    _si_frames.append(_preprocess(obs_raw))

    si_episode_data.append({
        'step': len(si_episode_data),
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
        emit('si_frame', {'image': encode_frame(obs_raw), 'done': False})

def run_si_analysis():
    valid = [d for d in si_episode_data if d['action'] is not None]
    if not valid:
        return
    total = len(valid)
    socketio.emit('si_analysis_start', {'total': total})

    analyses = []
    for i, entry in enumerate(valid):
        if i % 50 == 0:
            socketio.emit('si_analysis_progress',
                          {'current': i, 'total': total, 'pct': round(i/total*100)})
        q_vals   = get_q_values(d3qn_net, entry['stacked_state'], DEVICE)
        action   = entry['action']
        best_act = int(np.argmax(q_vals))
        player_q = float(q_vals[action])
        best_q   = float(q_vals[best_act])
        analyses.append({
            'step': i, 'action': action,
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
    worst_10  = sorted(analyses, key=lambda a: a['loss'], reverse=True)[:10]
    agree     = round(sum(1 for a in analyses if a['is_best']) / len(analyses) * 100, 1)

    from collections import Counter
    p_cnt  = Counter(a['action_name']      for a in analyses)
    ai_cnt = Counter(a['best_action_name'] for a in analyses)

    socketio.emit('si_analysis_done', {
        'avg_loss': avg_loss, 'worst': worst, 'worst_10': worst_10,
        'total_steps': total, 'agree_rate': agree,
        'player_actions': dict(p_cnt), 'ai_actions': dict(ai_cnt),
    })
    print(f"✅ SI 분석 완료 avg_loss={avg_loss:.3f} agree={agree}%")


# ══ Gomoku ════════════════════════════════════════════════════

def board_to_dict(board):
    cells = [int(board.states.get(r * BOARD_W + c, 0))
             for r in range(BOARD_H) for c in range(BOARD_W)]
    return {'cells': cells, 'current_player': int(board.current_player),
            'last_move': int(board.last_move) if board.last_move != -1 else -1,
            'width': BOARD_W, 'height': BOARD_H}

@socketio.on('gomoku_start')
def handle_gomoku_start():
    global gomoku_board, gomoku_history, gomoku_active, ai_player
    gomoku_board = Board(width=BOARD_W, height=BOARD_H, n_in_row=N_IN_ROW)
    gomoku_board.init_board(start_player=0)
    gomoku_history = []
    gomoku_active  = True
    ai_player = MCTSPlayer(make_policy_value_fn(gomoku_net), c_puct=5,
                           n_playout=400, is_selfplay=0)
    emit('gomoku_state', {**board_to_dict(gomoku_board), 'message': '당신은 흑(●)입니다!'})

@socketio.on('gomoku_move')
def handle_gomoku_move(data):
    global gomoku_board, gomoku_history, gomoku_active, ai_player
    if not gomoku_active:
        return
    row, col = data.get('row'), data.get('col')
    move = row * BOARD_W + col
    if move not in gomoku_board.availables:
        emit('gomoku_error', {'message': '이미 돌이 놓인 위치입니다.'})
        return

    board_before = copy.deepcopy(gomoku_board)
    gomoku_board.do_move(move)
    gomoku_history.append({'move': move, 'row': row, 'col': col,
                           'board_before': board_before})

    end, winner = gomoku_board.game_end()
    if end:
        gomoku_active = False
        r = '흑(●) 승리!' if winner==1 else ('백(○) 승리!' if winner==2 else '무승부')
        emit('gomoku_state', {**board_to_dict(gomoku_board), 'message': f'게임 종료 — {r}',
                              'game_over': True, 'winner': int(winner)})
        socketio.start_background_task(run_gomoku_analysis)
        return

    ai_move = ai_player.get_action(gomoku_board, temp=1e-3)
    ai_r, ai_c = int(ai_move // BOARD_W), int(ai_move % BOARD_W)
    gomoku_board.do_move(ai_move)
    end2, winner2 = gomoku_board.game_end()
    if end2:
        gomoku_active = False
        r = '흑(●) 승리!' if winner2==1 else ('백(○) 승리!' if winner2==2 else '무승부')
        emit('gomoku_state', {**board_to_dict(gomoku_board),
                              'message': f'AI 착수: ({ai_r},{ai_c}) — {r}',
                              'game_over': True, 'winner': int(winner2)})
        socketio.start_background_task(run_gomoku_analysis)
        return

    emit('gomoku_state', {**board_to_dict(gomoku_board), 'message': f'AI 착수: ({ai_r},{ai_c})'})

def run_gomoku_analysis():
    if not gomoku_history:
        return
    total = len(gomoku_history)
    socketio.emit('gomoku_analysis_start', {'total': total})
    analyses = []
    for i, entry in enumerate(gomoku_history):
        socketio.emit('gomoku_analysis_progress',
                      {'current': i+1, 'total': total, 'move': f"({entry['row']},{entry['col']})"})
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
    socketio.emit('gomoku_analysis_done', {
        'analyses': analyses, 'worst': worst,
        'avg_loss': avg_loss, 'total_moves': total,
    })
    print("✅ Gomoku 분석 완료")


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)