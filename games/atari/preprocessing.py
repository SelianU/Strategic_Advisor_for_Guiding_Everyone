"""
preprocessing.py (games.atari.preprocessing)

프레임/색상 관련 순수 유틸 함수.

- preprocess         : RGB → grayscale 84×84 (DQN 입력 표준)
- encode_frame(s)    : RGB → base64 JPEG (소켓 전송)
- compose_compare_frame : 두 프레임을 좌우로 이어 붙임 (counterfactual용)
- hex_to_rgb         : '#ff8a1c' → '255,138,28' (CSS 변수 주입용)

이 모듈은 atari 패키지 내부 어디서도 import 가능하며,
다른 모듈(gradcam 포함)에 의존하지 않는다.
"""
import base64

import cv2
import numpy as np


def preprocess(frame: np.ndarray) -> np.ndarray:
    """RGB 프레임 → grayscale 84×84 (DQN 입력 표준)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


def encode_frame(frame: np.ndarray) -> str:
    """RGB 프레임 → base64 JPEG 문자열."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')


def encode_frames(frames: list) -> list:
    return [encode_frame(f) for f in frames]


def compose_compare_frame(human_frame: np.ndarray, agent_frame: np.ndarray) -> np.ndarray:
    """두 프레임을 좌우로 이어 붙이고 가운데 분리선을 그린다."""
    h, w = human_frame.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = human_frame
    canvas[:, w:] = agent_frame
    cv2.line(canvas, (w, 0), (w, h), (80, 80, 80), 2)
    return canvas


def hex_to_rgb(hex_color: str) -> str:
    """'#ff8a1c' → '255,138,28'"""
    h = hex_color.lstrip('#')
    return ','.join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ── 호환성 alias ─────────────────────────────────────────────────────────────
# 기존 코드(atari_base.py)에서 _preprocess, _hex_to_rgb 같은 underscore 이름을
# 외부에서 직접 임포트하던 경우를 대비해 별칭을 유지.
_preprocess = preprocess
_hex_to_rgb = hex_to_rgb