"""
sessions.py (gomoku.sessions) — Gomoku 세션 디스크 영속화 + 직렬화 헬퍼.

소켓 핸들러(gomoku/handlers.py)에서 호출되는 순수 함수들:
  - 디렉토리 스캔/메타 읽기 (list_saved_sessions)
  - history → 직렬화 가능한 dict 변환 (serialize_gomoku_history)
  - 직렬화된 history → 보드 + history 재구성 (rebuild_gomoku_history)
  - 분석 결과 스냅샷 (gomoku_analysis_snapshot)
  - counterfactual 캐시 직렬화 (serialize_gomoku_cf_cache)
"""
import os
import copy
import json

from ai_agents.gomoku import GomokuRLBoard

from extensions   import SAVED_SESSIONS_DIR
from .state import state, BOARD_W, N_IN_ROW


def list_saved_sessions(game_name):
    path = os.path.join(SAVED_SESSIONS_DIR, game_name)
    os.makedirs(path, exist_ok=True)
    sessions = []
    for sid in os.listdir(path):
        meta_path = os.path.join(path, sid, 'meta.json')
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            sessions.append({'id': sid, 'title': meta.get('title') or '이름 없음',
                             'saved_at': meta.get('saved_at') or '', 'summary': meta.get('summary') or ''})
        except Exception:
            continue
    sessions.sort(key=lambda x: x['saved_at'], reverse=True)
    return sessions


def serialize_gomoku_history():
    return [{'move': e['move'], 'row': e['row'], 'col': e['col'],
             **({'ai_move': e['ai_move']} if e.get('ai_move') is not None else {})}
            for e in state.history]


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
    if not state.analysis_results:
        return None
    losses = [a['loss'] for a in state.analysis_results if a['loss'] is not None]
    return {
        'analyses':    state.analysis_results,
        'worst':       max((a for a in state.analysis_results if a['loss'] is not None), key=lambda a: a['loss'], default=None),
        'worst_10':    sorted((a for a in state.analysis_results if a['loss'] is not None), key=lambda a: a['loss'], reverse=True)[:5],
        'avg_loss':    round(sum(losses)/len(losses), 3) if losses else 0,
        'total_moves': len(state.history),
    }


def serialize_gomoku_cf_cache():
    saved = []
    for (sid, move_num), payload in state.counterfactual_cache.items():
        if sid != state.session_id:
            continue
        item = copy.deepcopy(payload)
        item['move_num'] = move_num
        saved.append(item)
    return saved