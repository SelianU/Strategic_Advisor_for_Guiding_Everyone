"""
handlers.py (gomoku.handlers) — Gomoku 관련 SocketIO 핸들러 전부.

원본 app.py의 다음 섹션을 합친 모듈:
  - Socket: Gomoku Sessions  (list / save / load / delete)
  - Socket: Gomoku Game      (start / move / AI 턴)
  - Gomoku 분석              (run_gomoku_analysis + 스케줄링)
  - Gomoku counterfactual    (request handler)

기존 `global gomoku_*` 선언은 모두 제거되었고, 가변 상태는
싱글턴 `state` 객체의 속성을 직접 대입/조회한다.
"""
import os
import copy
import json
import uuid
import shutil
import pickle
from datetime import datetime

import eventlet
import numpy as np
from flask         import request
from flask_socketio import emit

from ai_agents.gomoku import GomokuRLBoard
from llm_feedback     import generate_feedback

from extensions     import socketio, SAVED_SESSIONS_DIR
from .state   import state, gomoku_net, BOARD_W, BOARD_H, N_IN_ROW
from .engine  import (compute_q_values, compute_human_winrate, choose_gomoku_best_move,
                            find_gomoku_winning_line, human_perspective_outcome_text, board_to_dict)
from .render  import encode_frames, render_gomoku_board
from .sessions import (list_saved_sessions, serialize_gomoku_history, rebuild_gomoku_history,
                             gomoku_analysis_snapshot, serialize_gomoku_cf_cache)


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────
def emit_gomoku_terminal_state(payload, sid):
    socketio.emit('gomoku_state', payload, room=sid, namespace='/')


# ══════════════════════════════════════════════════════════════════════════════
# Gomoku 세션 관리 (저장 / 불러오기 / 삭제 / 목록)
# ══════════════════════════════════════════════════════════════════════════════
@socketio.on('gomoku_list_sessions')
def handle_gomoku_list_sessions():
    emit('gomoku_sessions_list', {'sessions': list_saved_sessions('gomoku')})


@socketio.on('gomoku_save_session')
def handle_gomoku_save_session(data):
    if not state.history or not state.analysis_results:
        emit('gomoku_session_saved', {'ok': False, 'message': '저장할 분석 결과가 없습니다.'}); return
    title    = (data.get('title') or '').strip() or 'Gomoku 기록'
    sid      = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
    sess_dir = os.path.join(SAVED_SESSIONS_DIR, 'gomoku', sid)
    os.makedirs(sess_dir, exist_ok=True)
    snapshot = gomoku_analysis_snapshot()
    meta = {'game': 'gomoku', 'title': title,
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'summary': f'총 {len(state.history)}수 · 평균 손실 {snapshot["avg_loss"]}'}
    with open(os.path.join(sess_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(sess_dir, 'session.pkl'), 'wb') as f:
        pickle.dump({'history': serialize_gomoku_history(), 'analysis_results': state.analysis_results,
                     'analysis_snapshot': snapshot, 'cached_counterfactuals': serialize_gomoku_cf_cache()}, f)
    emit('gomoku_session_saved', {'ok': True, 'message': '기록을 저장했습니다.', 'sessions': list_saved_sessions('gomoku')})


@socketio.on('gomoku_load_session')
def handle_gomoku_load_session(data):
    sess_dir = os.path.join(SAVED_SESSIONS_DIR, 'gomoku', data.get('session_id', ''))
    meta_p, data_p = os.path.join(sess_dir, 'meta.json'), os.path.join(sess_dir, 'session.pkl')
    if not (os.path.isfile(meta_p) and os.path.isfile(data_p)):
        emit('gomoku_session_loaded', {'ok': False, 'message': '기록을 찾지 못했습니다.'}); return
    with open(meta_p, 'r', encoding='utf-8') as f: meta = json.load(f)
    with open(data_p, 'rb') as f: payload = pickle.load(f)
    state.session_id += 1
    state.board, state.history = rebuild_gomoku_history(payload['history'])
    state.analysis_results = payload['analysis_results']
    state.counterfactual_cache = {}
    cached = payload.get('cached_counterfactuals') or []
    for item in cached:
        mn = int(item.get('move_num', -1))
        restored = copy.deepcopy(item); restored['session_id'] = state.session_id
        state.counterfactual_cache[(state.session_id, mn)] = restored
    state.active = False
    emit('gomoku_session_loaded', {'ok': True, 'meta': meta,
        'analysis': payload.get('analysis_snapshot') or gomoku_analysis_snapshot(),
        'board': board_to_dict(state.board), 'cached_counterfactuals': cached,
        'session_id': state.session_id})


@socketio.on('gomoku_delete_session')
def handle_gomoku_delete_session(data):
    sess_dir = os.path.join(SAVED_SESSIONS_DIR, 'gomoku', data.get('session_id', ''))
    if os.path.isdir(sess_dir): shutil.rmtree(sess_dir)
    emit('gomoku_session_deleted', {'ok': True, 'message': '기록을 삭제했습니다.', 'sessions': list_saved_sessions('gomoku')})


# ══════════════════════════════════════════════════════════════════════════════
# Gomoku 게임 진행 (start / move / AI 턴)
# ══════════════════════════════════════════════════════════════════════════════
def schedule_gomoku_analysis(session_id):
    if state.analysis_timer is not None:
        try: state.analysis_timer.cancel()
        except Exception: pass
    state.analysis_timer = eventlet.spawn_after(3.2, run_gomoku_analysis, session_id)


@socketio.on('gomoku_start')
def handle_gomoku_start(data=None):
    state.cancel_timers()
    data = data or {}
    state.human_color = int(data.get('human_color', 1))
    state.session_id += 1
    state.board = GomokuRLBoard(board_size=BOARD_W, n_in_row=N_IN_ROW)
    state.board.init_board(start_player=0)
    state.history = []
    state.analysis_results = []
    state.counterfactual_cache = {}
    state.state_seq = 1
    state.active = True
    state.ai_player = gomoku_net
    color_text = '흑(●)' if state.human_color == 1 else '백(○)'
    emit('gomoku_state', {**board_to_dict(state.board), 'message': f'당신은 {color_text}입니다!',
        'human_color': state.human_color, 'session_id': state.session_id, 'state_seq': state.state_seq,
        'win_rate': compute_human_winrate(state.board)})
    if state.human_color == 2:
        sid = request.sid; eventlet.sleep(0); eventlet.sleep(0.4)
        _process_gomoku_ai_opening(state.session_id, sid)


def _process_gomoku_ai_opening(session_id, sid):
    if session_id != state.session_id or not state.active or state.board is None: return
    try:
        ai_move = state.ai_player.get_action(state.board, temp=1e-3)
        ai_r, ai_c = int(ai_move // BOARD_W), int(ai_move % BOARD_W)
        state.board.do_move(ai_move); state.state_seq += 1
        socketio.emit('gomoku_state', {**board_to_dict(state.board),
            'message': f'AI 선착: ({BOARD_H - ai_r},{ai_c + 1}) — 당신의 차례입니다.',
            'human_color': state.human_color, 'session_id': session_id, 'state_seq': state.state_seq,
            'win_rate': compute_human_winrate(state.board)},
            room=sid, namespace='/')
    except Exception as exc: print(f'AI 선착 오류: {exc}')


@socketio.on('gomoku_move')
def handle_gomoku_move(data):
    if not state.active: return
    sid = request.sid
    row, col = data.get('row'), data.get('col')
    move = row * BOARD_W + col
    if move not in state.board.availables:
        emit('gomoku_error', {'message': '이미 돌이 놓인 위치입니다.'}, to=sid); return
    if state.board.current_player != state.human_color: return
    board_before = copy.deepcopy(state.board)
    state.board.do_move(move)
    state.history.append({'move': move, 'row': row, 'col': col, 'board_before': board_before})
    end, winner = state.board.game_end()
    if end:
        state.active = False
        r = '흑(●) 승리!' if winner==1 else ('백(○) 승리!' if winner==2 else '무승부')
        wl, _ = find_gomoku_winning_line(state.board) if winner in (1,2) else (None,-1)
        emit_gomoku_terminal_state({**board_to_dict(state.board), 'message': f'게임 종료 — {r}',
            'game_over': True, 'winner': int(winner), 'winning_line': wl,
            'outcome_text': human_perspective_outcome_text(winner),
            'human_color': state.human_color, 'session_id': state.session_id,
            'state_seq': state.state_seq+1,
            'win_rate': 100.0 if winner == state.human_color else (50.0 if winner == -1 else 0.0)}, sid)
        state.state_seq += 1; schedule_gomoku_analysis(state.session_id); return
    state.state_seq += 1
    emit('gomoku_state', {**board_to_dict(state.board),
        'message': f'당신의 착수: ({BOARD_H - row},{col + 1}) — AI 생각 중...',
        'human_color': state.human_color, 'session_id': state.session_id, 'state_seq': state.state_seq,
        'win_rate': compute_human_winrate(state.board)}, to=sid)
    eventlet.sleep(0); eventlet.sleep(0.55)
    process_gomoku_ai_turn(state.session_id, len(state.history)-1, sid)


def process_gomoku_ai_turn(session_id, history_index, sid):
    try:
        if session_id != state.session_id or not state.active or history_index >= len(state.history): return
        ai_move = state.ai_player.get_action(state.board, temp=1e-3)
        ai_r, ai_c = int(ai_move // BOARD_W), int(ai_move % BOARD_W)
        state.board.do_move(ai_move); state.history[history_index]['ai_move'] = ai_move
        end2, winner2 = state.board.game_end()
        if end2:
            state.active = False
            r = '흑(●) 승리!' if winner2==1 else ('백(○) 승리!' if winner2==2 else '무승부')
            wl, _ = find_gomoku_winning_line(state.board) if winner2 in (1,2) else (None,-1)
            emit_gomoku_terminal_state({**board_to_dict(state.board), 'message': f'AI 착수: ({BOARD_H - ai_r},{ai_c + 1}) — {r}',
                'game_over': True, 'winner': int(winner2), 'winning_line': wl,
                'outcome_text': human_perspective_outcome_text(winner2),
                'human_color': state.human_color, 'session_id': session_id,
                'state_seq': state.state_seq+1,
                'win_rate': 100.0 if winner2 == state.human_color else (50.0 if winner2 == -1 else 0.0)}, sid)
            state.state_seq += 1; schedule_gomoku_analysis(session_id); return
        state.state_seq += 1
        socketio.emit('gomoku_state', {**board_to_dict(state.board), 'message': f'AI 착수: ({BOARD_H - ai_r},{ai_c + 1})',
            'human_color': state.human_color, 'session_id': session_id, 'state_seq': state.state_seq,
            'win_rate': compute_human_winrate(state.board)},
            room=sid, namespace='/')
    except Exception as exc: print(f'오목 AI 턴 처리 오류: {exc}')


# ══════════════════════════════════════════════════════════════════════════════
# Gomoku 게임 후 분석
# ══════════════════════════════════════════════════════════════════════════════
def run_gomoku_analysis(expected_session_id=None):
    if (expected_session_id is not None and expected_session_id != state.session_id) or not state.history: return
    sid   = state.session_id
    total = len(state.history)
    socketio.emit('gomoku_analysis_start', {'total': total, 'session_id': sid})
    analyses = []
    for i, entry in enumerate(state.history):
        if sid != state.session_id: return
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
    if sid != state.session_id: return
    state.analysis_results = analyses
    losses = [a['loss'] for a in analyses if a['loss'] is not None]
    socketio.emit('gomoku_analysis_done', {
        'analyses': analyses,
        'worst':    max((a for a in analyses if a['loss'] is not None), key=lambda a: a['loss'], default=None),
        'worst_10': sorted((a for a in analyses if a['loss'] is not None), key=lambda a: a['loss'], reverse=True)[:5],
        'avg_loss': round(sum(losses)/len(losses), 3) if losses else 0,
        'total_moves': total, 'session_id': sid,
    })
    print('✅ Gomoku 분석 완료')


# ══════════════════════════════════════════════════════════════════════════════
# Gomoku Counterfactual (선택 착수와 PPO 최선수의 결과 비교 리플레이)
# ══════════════════════════════════════════════════════════════════════════════
@socketio.on('gomoku_request_counterfactual')
def handle_gomoku_counterfactual(data):
    if not state.history or not state.analysis_results:
        emit('gomoku_counterfactual_error', {'message': '오목 counterfactual 데이터를 찾지 못했습니다.'}); return
    move_num = int(data.get('move_num', -1))
    req_sid  = int(data.get('session_id', state.session_id))
    if req_sid != state.session_id:
        emit('gomoku_counterfactual_error', {'message': '이전 판의 요청입니다.', 'session_id': req_sid, 'move_num': move_num}); return
    cache_key = (req_sid, move_num)
    if cache_key in state.counterfactual_cache:
        emit('gomoku_counterfactual_ready', state.counterfactual_cache[cache_key]); return
    candidate     = next((a for a in state.analysis_results if a['move_num'] == move_num), None)
    history_entry = next((h for idx, h in enumerate(state.history, start=1) if idx == move_num), None)
    if candidate is None or history_entry is None:
        emit('gomoku_counterfactual_error', {'message': '선택한 착수를 찾지 못했습니다.', 'move_num': move_num}); return

    human_board = copy.deepcopy(history_entry['board_before'])
    agent_board = copy.deepcopy(history_entry['board_before'])
    hf, af      = [], []
    h_labels, a_labels = [], []
    h_outcome = a_outcome = None
    LIMIT = 10

    remaining = state.history[move_num-1:]
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
    state.counterfactual_cache[cache_key] = payload
    emit('gomoku_counterfactual_ready', payload)