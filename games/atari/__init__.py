"""
games.atari — Atari 게임 베이스 패키지.

서브클래스(games/space_invaders.py 등)에서:
    from games.atari import AtariGame

으로 사용한다 (기존: `from games.atari_base import AtariGame` 에서 변경).

자주 쓰는 유틸도 패키지 레벨에서 직접 임포트할 수 있도록 re-export:
    from games.atari import preprocess, encode_frame, AtariGradCAM, ...
"""

from .base           import AtariGame
from .gradcam        import AtariGradCAM, encode_frame_with_gradcam
from .preprocessing  import (preprocess, encode_frame, encode_frames,
                             compose_compare_frame, hex_to_rgb)

# 호환성 alias (구 atari_base.py 의 underscore 이름들)
from .preprocessing import _preprocess, _hex_to_rgb  # noqa: F401

__all__ = [
    'AtariGame',
    'AtariGradCAM',
    'encode_frame_with_gradcam',
    'preprocess', 'encode_frame', 'encode_frames',
    'compose_compare_frame', 'hex_to_rgb',
]