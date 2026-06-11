"""games/alien.py — Alien 게임 설정 (선언적 스펙 기반)"""
import numpy as np
from games.atari import AtariGame


def _alien_dir(dx: int, dy: int) -> str:
    """플레이어 기준 외계인 방향 (한국어)."""
    adx, ady = abs(dx), abs(dy)
    if adx < 15 and ady < 15:
        return '바로 옆'
    if adx > ady * 2:
        return '오른쪽' if dx > 0 else '왼쪽'
    if ady > adx * 2:
        return '아래쪽' if dy > 0 else '위쪽'
    if dx > 0 and dy > 0:
        return '오른쪽 아래'
    if dx > 0 and dy < 0:
        return '오른쪽 위'
    if dx < 0 and dy > 0:
        return '왼쪽 아래'
    return '왼쪽 위'


def extract_alien_game_state(ram: np.ndarray | None) -> dict:
    """
    Alien RAM 기반 게임 상태 피처 추출.

    RAM 레이아웃 (OC_Atari 기준):
      ram[0]      — 게임 모드 (0=일반, 32/48=다크)
      ram[42:45]  — 외계인 0~2 Y raw  (screen_y = 196 - val*2),  0이면 비활성
      ram[45]     — 플레이어 Y raw     (screen_y = 196 - val*2 + 1)
      ram[49:52]  — 외계인 0~2 X raw  (screen_x = val + 17),     0이면 비활성
      ram[52]     — 플레이어 X raw     (screen_x = val + 18)
      ram[64]     — 잔여 목숨 (표시값 = val - 1)
      ram[65:78]  — 왼쪽 에그 비트필드 (13행 × bits 2~7 → 6열)
      ram[78:91]  — 오른쪽 에그 비트필드 (동일 구조)
      ram[103]    — 펄사 위치 (0=없음, 1=오른쪽, 2=왼쪽, 3=위쪽)
      ram[117]    — 외계인 약화 상태 (139 = 약화 중)
    """
    if ram is None:
        return {}

    try:
        game_mode_raw = int(ram[0])
        game_mode     = 'normal' if game_mode_raw == 0 else 'dark'

        # ── 플레이어 위치 ───────────────────────────────────────────────────────
        px = int(ram[52]) + 18
        py = 196 - int(ram[45]) * 2 + 1

        # ── 외계인 위치 (일반 모드 전용, 최대 3마리) ─────────────────────────────
        aliens = []
        if game_mode == 'normal':
            for i in range(3):
                ax_raw = int(ram[49 + i])
                ay_raw = int(ram[42 + i])
                if ax_raw != 0 and ay_raw != 0:
                    ax  = ax_raw + 17
                    ay  = 196 - ay_raw * 2
                    dx  = ax - px
                    dy  = ay - py
                    dist = int((dx ** 2 + dy ** 2) ** 0.5)
                    aliens.append({
                        'x': ax, 'y': ay,
                        'dist': dist,
                        'dir':  _alien_dir(dx, dy),
                    })

        # ── 펄사 상태 ──────────────────────────────────────────────────────────
        pulsar_raw    = int(ram[103])
        pulsar_active = pulsar_raw != 0
        _PULSAR_POS   = {0: '없음', 1: '오른쪽', 2: '왼쪽', 3: '위쪽'}
        pulsar_pos_ko = _PULSAR_POS.get(pulsar_raw, '없음')

        # ── 외계인 약화 여부 ───────────────────────────────────────────────────
        alien_vulnerable = int(ram[117]) == 139

        # ── 남은 에그 수 (비트필드 파싱) ─────────────────────────────────────────
        # ram[65~77]: 왼쪽 13행 / ram[78~90]: 오른쪽 13행
        # 각 바이트의 bit2~bit7 (마스크 4,8,16,32,64,128) 이 에그 1개
        eggs_remaining = 0
        for i in range(13):
            for base in (65, 78):
                byte = int(ram[base + i])
                for mask in (4, 8, 16, 32, 64, 128):
                    if byte & mask:
                        eggs_remaining += 1

        # ── 잔여 목숨 ──────────────────────────────────────────────────────────
        lives = max(0, int(ram[64]) - 1)

        # ── 가장 가까운 외계인 ─────────────────────────────────────────────────
        nearest = min(aliens, key=lambda a: a['dist'], default=None)

        return {
            'game_mode':          game_mode,
            'player_x':           px,
            'player_y':           py,
            'alien_count':        len(aliens),
            'nearest_alien_dist': nearest['dist'] if nearest else None,
            'nearest_alien_dir':  nearest['dir']  if nearest else None,
            'pulsar_active':      pulsar_active,
            'pulsar_position':    pulsar_pos_ko,
            'alien_vulnerable':   alien_vulnerable,
            'eggs_remaining':     eggs_remaining,
            'lives':              lives,
            'aliens':             aliens,
        }
    except Exception:
        return {}

class AlienGame(AtariGame):
    game_id    = 'alien'
    game_title = 'ALIEN'
    game_icon  = '👽'
    env_name   = 'ALE/Alien-v5'
    frame_skip  = 4
    prefix     = 'al_'
    theme_color = '#a855f7'   # 보라색 계열
    model_path_parts = ('data', 'checkpoints', 'alien', 'best_model_alien.pth')

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
        {'id': 'fire',  'label': 'SPACE', 'actions': [1, 10, 11, 12, 13, 14, 15, 16, 17]},
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

    # 에그·펄사: reward == 10 / 외계인 처치·프라이즈: reward >= 400
    achievement_specs = [
        # bronze
        {'id': 'egg_10',        'title': '아이템 수집가',    'tier': 'bronze',   'desc': '에그·펄사 10개 수집',
         'metric': 'reward_count', 'value': 10, 'reward_exact': 10},
        {'id': 'score_800',     'title': '생존자',            'tier': 'bronze',   'desc': '800점 달성',
         'metric': 'score', 'value': 800},
        {'id': 'big_1',         'title': '찬스 포착',         'tier': 'bronze',   'desc': '외계인 처치 또는 프라이즈 획득 1회',
         'metric': 'reward_count', 'value': 1, 'reward_min': 400},
        # silver
        {'id': 'egg_30',        'title': '미로 탐험가',      'tier': 'silver',   'desc': '에그·펄사 30개 수집',
         'metric': 'reward_count', 'value': 30, 'reward_exact': 10},
        {'id': 'big_2',         'title': '기회주의자',        'tier': 'silver',   'desc': '외계인 처치 또는 프라이즈 획득 2회',
         'metric': 'reward_count', 'value': 2, 'reward_min': 400},
        {'id': 'big_3',         'title': '찬스 메이커',      'tier': 'silver',   'desc': '외계인 처치 또는 프라이즈 획득 3회',
         'metric': 'reward_count', 'value': 3, 'reward_min': 400},
        {'id': 'no_death_1500', 'title': '무결점 탐험',      'tier': 'silver',   'desc': '목숨 잃지 않고 1500스텝 생존',
         'metric': 'no_death', 'value': 1500},
        {'id': 'score_2000',    'title': '에이리언 헌터',    'tier': 'silver',   'desc': '2000점 달성',
         'metric': 'score', 'value': 2000},
        # gold
        {'id': 'egg_70',        'title': '고지가 코앞',      'tier': 'gold',     'desc': '에그·펄사 70개 수집',
         'metric': 'reward_count', 'value': 70, 'reward_exact': 10},
        {'id': 'score_4000',    'title': '우주 전사',         'tier': 'gold',     'desc': '4000점 달성',
         'metric': 'score', 'value': 4000},
        {'id': 'score_6000',    'title': '외계 정복자',       'tier': 'gold',     'desc': '6000점 달성',
         'metric': 'score', 'value': 6000},
        {'id': 'bench_dqn',     'title': 'DQN 돌파',          'tier': 'platinum', 'desc': 'DQN 기준(3,069점) 초과',
         'metric': 'score', 'value': 3069, 'cmp': '>'},
        {'id': 'bench_human',   'title': '인간 평균 돌파',    'tier': 'gold',     'desc': '인간 평균(6,875점) 초과',
         'metric': 'score', 'value': 6875, 'cmp': '>'},
        # platinum
        {'id': 'bench_ddqn',    'title': 'DDQN 돌파',         'tier': 'gold',     'desc': 'DDQN 기준(2,907점) 초과',
         'metric': 'score', 'value': 2907, 'cmp': '>'},
    ]

    def _extra_summary(self, entry: dict) -> dict:
        pre_ram = entry.get('pre_ram')
        if pre_ram is None:
            return {}
        gs = extract_alien_game_state(pre_ram)
        return {'game_state': gs} if gs else {}

    game_info = {
        'summary': '미로형 우주선을 탐험하며 에그를 제거하고 외계인을 피하는 생존 게임입니다.',
        'objective': '미로 안의 에그를 모두 제거해 라운드를 클리어하고 최고 점수를 달성하세요.',
        'how_to_play': [
            '방향키(↑↓←→)로 미로를 이동합니다. 대각선 이동도 가능합니다.',
            '에그(흰 점) 위를 지나가면 자동 제거되며 +10점. 미로의 에그를 모두 제거하면 라운드 클리어.',
            '펄사(파워업)는 미로당 최대 3개 등장. 획득 시 +10점 + 외계인 일시 약화. 약화 중 외계인에게 닿으면 +500점.',
            '프라이즈(노란 아이템)는 미로당 최대 2개 등장. 획득 시 +500점.',
            '스페이스 바: 화염 방사기 — 외계인을 밀어내는 방어 수단. 외계인을 직접 제거하지는 못합니다.',
            '미로 중앙의 좌우 통로는 연결되어 있습니다. 왼쪽 끝으로 나가면 오른쪽 끝에서 등장하고, 반대도 마찬가지.',
        ],
        'scoring': [
            '에그 제거: +10점 (라운드 클리어 조건)',
            '펄사 획득: +10점 + 외계인 약화 효과 (미로당 최대 3개)',
            '외계인 처치: +500점 (펄사 효과 중 외계인에게 닿기)',
            '프라이즈 획득: +500점 (미로당 최대 2개)',
        ],
        'bonus_round': (
            '미로 클리어 후 배경이 검게 변하는 보너스 라운드가 등장합니다. '
            '여러 색깔의 외계인들이 화면을 자유롭게 돌아다니며, '
            '화면 위쪽에 프라이즈 우주선이 날아다닙니다. '
            '프라이즈 우주선에 닿으면 고득점 보너스를 얻을 수 있습니다. '
            '외계인이 많아 위험하므로 이동 경로를 신중하게 잡고, '
            '펄사가 있다면 활용해 외계인을 약화시키며 틈을 노리세요.'
        ),
        'end_condition': '목숨 3개를 모두 소진하면 종료됩니다. 에그를 모두 제거하면 다음 라운드로 진행합니다.',
        'tips': [
            '외계인이 가까이 오면 스페이스 바로 밀어내고 도망가세요.',
            '펄사를 먹은 직후가 외계인을 처치할 수 있는 유일한 기회입니다.',
            '프라이즈는 고점수 기회이니 놓치지 마세요.',
            '좌우 통로로 도망치면 화면 반대편에서 나와 외계인을 따돌릴 수 있습니다.',
            '보너스 라운드에서는 외계인 수가 많으므로 무리하지 말고 안전한 경로로 프라이즈를 노리세요.',
        ],
        'controls': {
            'up':    '이동',
            'down':  '이동',
            'left':  '이동',
            'right': '이동',
            'fire':  '화염방사',
        },
    }
