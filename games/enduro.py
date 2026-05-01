"""games/enduro.py — Enduro 게임 설정"""
import os
from games.atari_base import AtariGame


class EnduroGame(AtariGame):
    game_id    = 'enduro'
    game_title = 'ENDURO'
    game_icon  = '🏎️'
    env_name   = 'ALE/Enduro-v5'
    prefix     = 'en_'
    theme_color = '#ff2255'
    model_path_parts = ('ai_agents', 'enduro', 'checkpoints', 'best_model.pth')

    action_names = {
        0: 'NOOP',
        1: 'FIRE',
        2: 'RIGHT',
        3: 'LEFT',
        4: 'DOWN',
        5: 'DOWNRIGHT',
        6: 'DOWNLEFT',
        7: 'RIGHTFIRE',
        8: 'LEFTFIRE',
    }

    keyboard_keys = [
        {'id': 'left',      'label': '←',    'actions': [3, 6, 8]},   # LEFT, DOWNLEFT, LEFTFIRE
        {'id': 'right',     'label': '→',    'actions': [2, 5, 7]},   # RIGHT, DOWNRIGHT, RIGHTFIRE
        {'id': 'down',      'label': '↓',    'actions': [4, 5, 6]},   # DOWN, DOWNRIGHT, DOWNLEFT
        {'id': 'fire',      'label': 'GAS',  'actions': [1, 7, 8]},   # FIRE, RIGHTFIRE, LEFTFIRE
    ]

    key_combos = {
        # 3키 조합
        'down+fire+left':  6,   # DOWNLEFT  (fire 무시, down+left 우선)
        'down+fire+right': 5,   # DOWNRIGHT
        # 2키 조합 (알파벳순)
        'down+left':       6,   # DOWNLEFT
        'down+right':      5,   # DOWNRIGHT
        'fire+left':       8,   # LEFTFIRE
        'fire+right':      7,   # RIGHTFIRE
        # 단일 키
        'down':            4,
        'fire':            1,
        'left':            3,
        'right':           2,
        # 아무것도 안 누름
        '':                0,
    }

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.enduro import load_enduro_d3qn
        net, _ = load_enduro_d3qn(path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.enduro import get_q_values
        return get_q_values(self.net, stacked_state, self.device)