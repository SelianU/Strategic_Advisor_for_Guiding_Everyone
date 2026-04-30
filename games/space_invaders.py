"""games/space_invaders.py — Space Invaders 게임 설정"""
import os
import numpy as np
from games.atari_base import AtariGame


def extract_si_game_state(rgb_frame: np.ndarray) -> dict:
    """RGB 프레임(210×160×3)에서 Space Invaders 게임 상태 피처를 추출합니다."""
    brightness = rgb_frame.max(axis=2)
    active = brightness > 40
    h, w = active.shape

    enemy_zone = active[20:145, :]
    row_activity = enemy_zone.sum(axis=1)
    enemy_rows = np.where(row_activity > 3)[0]
    if len(enemy_rows) > 0:
        enemy_lowest_y = int(enemy_rows.max()) + 20
        total_enemy_px = int(enemy_zone.sum())
        enemy_count = max(1, min(55, total_enemy_px // 45))
    else:
        enemy_lowest_y = 0
        enemy_count = 0
    danger_distance = max(0, 145 - enemy_lowest_y) if enemy_count > 0 else 999

    # 방패 감지 (실측 캘리브레이션: col 42-49, 74-81, 106-113)
    shield_zone = active[142:175, :]
    shield_starts = [42, 74, 106]  # 실측 방패 x 시작 위치
    shield_width = 8               # 실측 방패 너비
    max_px = 112                   # 온전한 방패 1개의 실측 픽셀 (8+16×6+8)
    shield_integrity = []
    for sx in shield_starts:
        px = int(shield_zone[:, sx:sx + shield_width].sum())
        shield_integrity.append(round(min(100, px * 100 // max(1, max_px))))

    player_zone = active[168:182, :]
    player_cols = np.where(player_zone.any(axis=0))[0]
    if len(player_cols) > 0:
        player_x = int(player_cols.mean())
        player_x_pct = round(player_x * 100 // w)
    else:
        player_x = w // 2
        player_x_pct = 50
    player_under_shield = any(sx <= player_x <= sx + shield_width for sx in shield_starts)

    bullet_zone = active[145:168, :]
    bullet_rows = np.where(bullet_zone.any(axis=1))[0]
    incoming_proximity = (168 - (int(bullet_rows.max()) + 145)) if len(bullet_rows) > 0 else 23

    # 탄환이 실제로 방패 X범위 위에 있는지 확인 (단순 Y근접과 구분)
    bullet_over_shield = False
    if len(bullet_rows) > 0:
        bullet_cols = np.where(bullet_zone.any(axis=0))[0]
        for bx in bullet_cols:
            if any(sx <= bx <= sx + shield_width for sx in shield_starts):
                bullet_over_shield = True
                break

    if enemy_count <= 3:
        speed_phase = 'critical'
    elif enemy_count <= 9:
        speed_phase = 'fast'
    else:
        speed_phase = 'normal'

    col_has_enemy = enemy_zone.any(axis=0)
    col_size = max(1, w // 11)
    cleared_columns = sum(
        1 for i in range(11)
        if not col_has_enemy[i * col_size:(i + 1) * col_size].any()
    )

    return {
        'enemy_count':         enemy_count,
        'enemy_lowest_row':    enemy_lowest_y,
        'danger_distance':     danger_distance,
        'shield_integrity':    shield_integrity,
        'player_x_pct':        player_x_pct,
        'player_under_shield': player_under_shield,
        'enemy_speed_phase':   speed_phase,
        'cleared_columns':     cleared_columns,
        'incoming_proximity':   incoming_proximity,
        'bullet_over_shield':   bullet_over_shield,
    }


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

    def _extra_summary(self, entry: dict) -> dict:
        """pre_rgb 프레임에서 게임 상태 피처를 추출해 summary에 추가합니다."""
        pre_rgb = entry.get('pre_rgb')
        if pre_rgb is None:
            return {}
        return {'game_state': extract_si_game_state(pre_rgb)}
