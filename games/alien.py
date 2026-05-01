"""games/alien.py — Alien 게임 설정"""
import os
from games.atari_base import AtariGame

class AlienGame(AtariGame):
    game_id    = 'alien'
    game_title = 'ALIEN'
    game_icon  = '👽'
    env_name   = 'ALE/Alien-v5'
    prefix     = 'al_'
    theme_color = '#a855f7'   # 보라색 계열
    model_path_parts = ('ai_agents', 'alien', 'checkpoints', 'best_model.pth')

    action_names = {
        0: 'NOOP',      1: 'FIRE',
        2: 'UP',        3: 'RIGHT',       4: 'LEFT',        5: 'DOWN',
        6: 'UPRIGHT',   7: 'UPLEFT',      8: 'DOWNRIGHT',   9: 'DOWNLEFT',
        10: 'UPFIRE',   11: 'RIGHTFIRE',  12: 'LEFTFIRE',   13: 'DOWNFIRE',
        14: 'UPRIGHTFIRE', 15: 'UPLEFTFIRE',
        16: 'DOWNRIGHTFIRE', 17: 'DOWNLEFTFIRE',
    }

    keyboard_keys = [
        {'id': 'up',    'label': '↑',    'actions': [2,  6,  7,  10, 14, 15]},
        {'id': 'down',  'label': '↓',    'actions': [5,  8,  9,  13, 16, 17]},
        {'id': 'left',  'label': '←',    'actions': [4,  7,  9,  12, 15, 17]},
        {'id': 'right', 'label': '→',    'actions': [3,  6,  8,  11, 14, 16]},
        {'id': 'fire',  'label': 'FIRE', 'actions': [1, 10, 11, 12, 13, 14, 15, 16, 17]},
    ]

    key_combos = {
        # 3키 조합 (알파벳 순: down < fire < left < right < up)
        'down+fire+right':  16,  # DOWNRIGHTFIRE
        'down+fire+left':   17,  # DOWNLEFTFIRE
        'fire+right+up':    14,  # UPRIGHTFIRE
        'fire+left+up':     15,  # UPLEFTFIRE
        # 2키 조합
        'down+fire':        13,  # DOWNFIRE
        'down+right':        8,  # DOWNRIGHT
        'down+left':         9,  # DOWNLEFT
        'fire+right':       11,  # RIGHTFIRE
        'fire+left':        12,  # LEFTFIRE
        'fire+up':          10,  # UPFIRE
        'right+up':          6,  # UPRIGHT
        'left+up':           7,  # UPLEFT
        # 단독
        'down':              5,
        'fire':              1,
        'left':              4,
        'right':             3,
        'up':                2,
        '':                  0,  # NOOP
    }

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.d3qn_helper import load_d3qn
        net, _ = load_d3qn('alien', path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.d3qn_helper import get_q_values
        return get_q_values(self.net, stacked_state, self.device)