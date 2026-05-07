"""
render.py (gomoku.render) — cv2 기반 Gomoku 보드 시각화 + 프레임 인코딩.

순수 로직(gomoku/engine.py)과 분리한 이유: cv2/이미지 처리 의존성을
한 곳에 격리하고, 핸들러 코드에서는 결과(numpy 배열)만 다룰 수 있도록.
"""
import base64

import cv2
import numpy as np

from .state import BOARD_W, BOARD_H


# ── 프레임 인코딩 ────────────────────────────────────────────────────────────
def encode_frame(frame):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')


def encode_frames(frames):
    return [encode_frame(f) for f in frames]


# ── Q-value 히트맵 색상 ─────────────────────────────────────────────────────
def _q_to_bgr(t):
    if t < 0.5:
        s = t * 2
        return (int(210-160*s), int(80+130*s), int(30+20*s))
    s = (t-0.5)*2
    return (int(50-50*s), int(210-180*s), int(50+205*s))


# ── 보드 렌더링 ─────────────────────────────────────────────────────────────
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