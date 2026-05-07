"""
engine.py (gomoku.engine) — Gomoku 순수 로직 함수들.

cv2 / 디스크 / 소켓에 의존하지 않는 계산 함수만 모은다:
  - Q-value / best move
  - 인간 관점 승률
  - 승리선 검출
  - 결과 텍스트
  - 보드 → dict 직렬화 (소켓 페이로드용 단순 변환)
"""
import numpy as np

from .state import state, gomoku_net, BOARD_W, BOARD_H


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
    v_human = v_s if board.current_player == state.human_color else -v_s
    # v_human ∈ [-1, 1]을 [0, 100]으로 변환
    win_rate = (v_human + 1.0) / 2.0 * 100.0
    return round(float(win_rate), 1)


def choose_gomoku_best_move(board, n_playout=80):
    res = compute_q_values(board, gomoku_net, n_playout=n_playout)
    return int(res['best_action']), res


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
    if winner == -1:                    return 'DRAW'
    if winner == state.human_color:     return 'WIN'
    return 'LOSE'