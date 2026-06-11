"""games/atlantis.py — Atlantis 게임 설정 (선언적 스펙 기반)"""
import numpy as np
from games.atari import AtariGame


def extract_atlantis_game_state(ram: np.ndarray | None, rgb_frame: np.ndarray | None) -> dict:
    """
    Atlantis RAM 기반 게임 상태 피처 추출.

    RAM 레이아웃 (OC_Atari / NVlabs cule 기준):
      ram[36:40]  — 적 우주선 X 좌표 (4 슬롯, 0 = 비활성)
      ram[71:75]  — 적 우주선 고도 레인 인덱스 (% 4 → 0=상단, 3=도시 근접)
      ram[79:83]  — 적 우주선 종류 값 (< 80 = 고곤류 소형, == 80 = 밴디트, 나머지 = 타나토이드)
      ram[84:91]  — 도시 구역 파괴 플래그 (0 = 온전, 1+ = 파괴됨), 7개 구역
      ram[62]     — 데스레이(적 폭탄) X좌표, 0이 아니면 활성
    """
    if ram is None:
        return {}

    try:
        # ── 적 우주선 슬롯 파싱 ──────────────────────────────────────────────────
        SCREEN_W = 160
        enemy_x   = [int(ram[36 + i]) for i in range(4)]
        enemy_alt = [int(ram[71 + i]) % 4 for i in range(4)]  # 0=상단, 3=도시 근접
        enemy_type_val = [int(ram[79 + i]) for i in range(4)]

        def _type_name(v: int) -> str:
            if v == 0:
                return 'none'
            if v == 80:
                return 'bandit'   # 밴디트: 중형, 빠름
            if v < 80:
                return 'gorgon'   # 고곤: 소형, 가장 빠름, 폭탄 투하
            return 'thanatoid'    # 타나토이드: 대형, 느림, 고점수

        def _x_zone(x: int) -> str:
            if x < 55:
                return 'left'
            if x < 110:
                return 'center'
            return 'right'

        # 활성 적 (x != 0)
        active_enemies = []
        for i in range(4):
            if enemy_x[i] > 0:
                active_enemies.append({
                    'x': enemy_x[i],
                    'x_pct': round(enemy_x[i] * 100 // SCREEN_W),
                    'zone': _x_zone(enemy_x[i]),
                    'altitude_lane': enemy_alt[i],       # 3 = 도시에 가장 가까움
                    'type': _type_name(enemy_type_val[i]),
                })

        # ── 최고 위협 적 (고도 레인이 가장 높은 = 3에 가까울수록 위험) ──────────
        most_threatening = max(active_enemies, key=lambda e: e['altitude_lane'], default=None)
        most_threatening_zone     = most_threatening['zone'] if most_threatening else 'none'
        most_threatening_altitude = most_threatening['altitude_lane'] if most_threatening else -1
        most_threatening_type     = most_threatening['type'] if most_threatening else 'none'

        # ── 도시 구역 잔존 수 ────────────────────────────────────────────────────
        city_flags = [int(ram[84 + i]) for i in range(7)]
        structures_remaining = city_flags.count(0)   # 0 = 온전, 7개 중 몇 개 남았나

        # ── 데스레이(적 폭탄) 활성 여부 ──────────────────────────────────────────
        deathray_active = int(ram[62]) != 0

        return {
            'enemy_count':               len(active_enemies),
            'most_threatening_zone':     most_threatening_zone,
            'most_threatening_altitude': most_threatening_altitude,  # 0–3
            'most_threatening_type':     most_threatening_type,
            'city_structures_remaining': structures_remaining,       # 0–7
            'deathray_active':           deathray_active,
            'active_enemy_zones':        [e['zone'] for e in active_enemies],
        }
    except Exception:
        return {}

class AtlantisGame(AtariGame):
    game_id    = 'atlantis'
    game_title = 'ATLANTIS'
    game_icon  = '🌊'
    env_name   = 'ALE/Atlantis-v5'
    frame_skip  = 4
    prefix     = 'at_'
    theme_color = '#4488ff'
    model_path_parts = ('data', 'checkpoints', 'atlantis', 'best_model_atlantis.pth')

    action_names = {
        0: 'NOOP',  1: 'FIRE',  2: 'RIGHTFIRE',  3: 'LEFTFIRE',
    }

    keyboard_keys = [
        {'id': 'left',  'label': '←',    'actions': [3]},
        {'id': 'right', 'label': '→',    'actions': [2]},
        {'id': 'fire',  'label': 'SPACE', 'actions': [1, 2, 3]},
    ]

    key_combos = {
        'fire+right': 2,
        'fire+left':  3,
        'fire':       1,
        '':           0,
    }

    achievement_specs = [
        # bronze
        {'id': 'score_3000',  'title': '첫 방어',        'tier': 'bronze',   'desc': '3000점 달성',
         'metric': 'score', 'value': 3000},
        {'id': 'score_10000', 'title': '도시 수호자',    'tier': 'bronze',   'desc': '10000점 달성',
         'metric': 'score', 'value': 10000},
        {'id': 'combo_3',     'title': '연속 격파 I',    'tier': 'bronze',   'desc': '200스텝 내 적 3회 격파',
         'metric': 'combo', 'value': 3, 'window': 200},
        # silver
        {'id': 'score_20000', 'title': '방위군',          'tier': 'silver',   'desc': '20000점 달성',
         'metric': 'score', 'value': 20000},
        {'id': 'bench_human', 'title': '인간 평균 돌파',  'tier': 'silver',   'desc': '인간 평균(29,028점) 초과',
         'metric': 'score', 'value': 29028, 'cmp': '>'},
        {'id': 'combo_8',     'title': '연속 격파 II',   'tier': 'silver',   'desc': '200스텝 내 적 8회 격파',
         'metric': 'combo', 'value': 8, 'window': 200},
        {'id': 'survive_wave','title': '웨이브 생존',     'tier': 'silver',   'desc': '목숨 잃지 않고 300스텝 생존',
         'metric': 'no_death', 'value': 300},
        # gold
        {'id': 'score_50000', 'title': '아틀란티스 영웅', 'tier': 'gold',     'desc': '50000점 달성',
         'metric': 'score', 'value': 50000},
        {'id': 'combo_15',    'title': '연속 격파 III',  'tier': 'gold',     'desc': '200스텝 내 적 15회 격파',
         'metric': 'combo', 'value': 15, 'window': 200},
        {'id': 'bench_dqn',   'title': 'DQN 돌파',       'tier': 'gold',     'desc': 'DQN 기준(85,950점) 초과',
         'metric': 'score', 'value': 85950, 'cmp': '>'},
        # platinum
        {'id': 'bench_ddqn',  'title': 'DDQN 돌파',      'tier': 'platinum', 'desc': 'DDQN 기준(64,758점) 초과',
         'metric': 'score', 'value': 64758, 'cmp': '>'},
    ]

    def _extra_summary(self, entry: dict) -> dict:
        pre_ram = entry.get('pre_ram')
        pre_rgb = entry.get('pre_rgb')
        if pre_ram is None and pre_rgb is None:
            return {}
        gs = extract_atlantis_game_state(pre_ram, pre_rgb)
        return {'game_state': gs} if gs else {}

    game_info = {
        'summary': '해저 도시 아틀란티스를 공습하는 외계 함대로부터 도시를 방어하는 게임입니다.',
        'objective': '도시의 핵심 시설 3곳을 마지막까지 지키세요.',
        'how_to_play': [
            'SPACE: 중앙 포탑 발사',
            '← + SPACE: 왼쪽 포탑 발사',
            '→ + SPACE: 오른쪽 포탑 발사',
        ],
        'scoring': [
            '적 함선 격추: 50~250점 (크기·속도에 따라)',
            '웨이브 클리어 보너스',
        ],
        'end_condition': '도시 시설 3곳이 모두 파괴되면 게임이 종료됩니다.',
        'tips': [
            '가장 낮은 고도로 내려온 적을 우선 격추하세요.',
            '좌우 포탑의 사각지대를 중앙 포탑으로 커버하세요.',
        ],
        'controls': {
            'left':  'SPACE와 함께 왼쪽 포탑 발사',
            'right': 'SPACE와 함께 오른쪽 포탑 발사',
            'fire':  '중앙 포탑 발사',
        },
    }
