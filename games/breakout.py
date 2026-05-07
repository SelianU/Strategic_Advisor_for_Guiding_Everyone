"""games/breakout.py — Breakout 게임 설정"""
import os
from games.atari import AtariGame


class BreakoutGame(AtariGame):
    game_id    = 'breakout'
    game_title = 'BREAKOUT'
    game_icon  = '🎯'
    env_name   = 'ALE/Breakout-v5'
    prefix     = 'bo_'
    theme_color = '#ff8a1c'
    model_path_parts = ('ai_agents', 'breakout', 'checkpoints', 'best_model.pth')

    action_names = {
        0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT',
    }

    keyboard_keys = [
        {'id': 'left',  'label': '←',    'actions': [3]},
        {'id': 'right', 'label': '→',    'actions': [2]},
        {'id': 'fire',  'label': 'FIRE', 'actions': [1]},
    ]

    key_combos = {
        'left':  3,
        'right': 2,
        'fire':  1,
        '':      0,
    }

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.breakout import load_breakout_d3qn
        net, _ = load_breakout_d3qn(path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.breakout import get_q_values as bo_get_q_values
        return bo_get_q_values(self.net, stacked_state, self.device)