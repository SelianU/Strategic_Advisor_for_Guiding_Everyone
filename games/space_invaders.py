"""games/space_invaders.py — Space Invaders 게임 설정"""
import os
from games.atari_base import AtariGame


class SpaceInvadersGame(AtariGame):
    game_id    = 'space_invaders'
    game_title = 'SPACE INVADERS'
    game_icon  = '👾'
    env_name   = 'ALE/SpaceInvaders-v5'
    prefix     = 'si_'
    theme_color = '#39ff14'
    model_path_parts = ('ai_agents', 'space_invaders', 'checkpoints', 'best_model.pth')

    action_names = {
        0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT', 4: 'RIGHTFIRE', 5: 'LEFTFIRE',
    }

    keyboard_keys = [
        {'id': 'left',  'label': '←',    'actions': [3, 5]},       # LEFT, LEFTFIRE
        {'id': 'right', 'label': '→',    'actions': [2, 4]},       # RIGHT, RIGHTFIRE
        {'id': 'fire',  'label': 'FIRE', 'actions': [1, 4, 5]},    # FIRE, RIGHTFIRE, LEFTFIRE
    ]

    key_combos = {
        'fire+left':  5,   # LEFTFIRE  (알파벳순 정렬)
        'fire+right': 4,   # RIGHTFIRE
        'left':       3,
        'right':      2,
        'fire':       1,
        '':           0,
    }

    @property
    def url_path(self):
        return '/space-invaders'   # 기존 URL 유지

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.space_invaders import load_d3qn
        net, _ = load_d3qn(path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.space_invaders import get_q_values
        return get_q_values(self.net, stacked_state, self.device)