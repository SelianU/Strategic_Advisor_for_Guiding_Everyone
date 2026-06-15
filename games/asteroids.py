"""games/asteroids.py — Asteroids 게임 설정 (선언적 스펙 기반)"""
from games.atari import AtariGame

class AsteroidsGame(AtariGame):
    game_id    = 'asteroids'
    game_title = 'ASTEROIDS'
    game_icon  = '☄️'
    env_name   = 'ALE/Asteroids-v5'
    frame_skip  = 4
    prefix     = 'ao_'
    theme_color = '#00bfff'
    model_path_parts = ('data', 'checkpoints', 'asteroids', 'best_model_asteroids.pth')

    action_names = {
        0: 'NOOP',   1: 'FIRE',        2: 'UP',        3: 'RIGHT',
        4: 'LEFT',   5: 'DOWN',        6: 'UPRIGHT',   7: 'UPLEFT',
        8: 'UPFIRE', 9: 'RIGHTFIRE',  10: 'LEFTFIRE', 11: 'DOWNFIRE',
        12: 'UPRIGHTFIRE', 13: 'UPLEFTFIRE',
    }

    keyboard_keys = [
        {'id': 'up',    'label': '↑',    'actions': [2, 6, 7, 8, 12, 13]},
        {'id': 'down',  'label': '↓',    'actions': [5, 11]},
        {'id': 'left',  'label': '←',    'actions': [4, 7, 10, 13]},
        {'id': 'right', 'label': '→',    'actions': [3, 6, 9, 12]},
        {'id': 'fire',  'label': 'SPACE', 'actions': [1, 8, 9, 10, 11, 12, 13]},
    ]

    key_combos = {
        'fire+right+up': 12,
        'fire+left+up':  13,
        'fire+up':        8,
        'fire+right':     9,
        'fire+left':     10,
        'down+fire':     11,
        'right+up':       6,
        'left+up':        7,
        'up':             2,
        'right':          3,
        'left':           4,
        'down':           5,
        'fire':           1,
        '':               0,
    }

    achievement_specs = [
        # bronze
        {'id': 'score_300',   'title': '첫 충돌',        'tier': 'bronze',   'desc': '300점 달성',
         'metric': 'score', 'value': 300},
        {'id': 'score_1000',  'title': '우주 청소부',    'tier': 'bronze',   'desc': '1000점 달성',
         'metric': 'score', 'value': 1000},
        {'id': 'combo_3',     'title': '연속 격파 I',    'tier': 'bronze',   'desc': '200스텝 내 소행성 3개 격파',
         'metric': 'combo', 'value': 3, 'window': 200},
        # silver
        {'id': 'score_3000',  'title': '유성 사냥꾼',    'tier': 'silver',   'desc': '3000점 달성',
         'metric': 'score', 'value': 3000},
        {'id': 'score_6000',  'title': '소행성 파괴자',  'tier': 'silver',   'desc': '6000점 달성',
         'metric': 'score', 'value': 6000},
        {'id': 'combo_6',     'title': '연속 격파 II',   'tier': 'silver',   'desc': '200스텝 내 소행성 6개 격파',
         'metric': 'combo', 'value': 6, 'window': 200},
        {'id': 'survive_500', 'title': '생존자',          'tier': 'silver',   'desc': '500스텝 생존',
         'metric': 'survive', 'value': 500},
        # gold
        {'id': 'score_10000', 'title': '별의 지배자',    'tier': 'gold',     'desc': '10000점 달성',
         'metric': 'score', 'value': 10000},
        {'id': 'combo_10',    'title': '연속 격파 III',  'tier': 'gold',     'desc': '200스텝 내 소행성 10개 격파',
         'metric': 'combo', 'value': 10, 'window': 200},
        {'id': 'bench_human', 'title': '인간 평균 돌파', 'tier': 'gold',     'desc': '인간 평균(13,157점) 초과',
         'metric': 'score', 'value': 13157, 'cmp': '>'},
        # platinum
        {'id': 'score_20000', 'title': '우주 전설',      'tier': 'platinum', 'desc': '20000점 달성',
         'metric': 'score', 'value': 20000},
    ]

    game_info = {
        'summary': '우주를 떠도는 소행성과 UFO를 파괴하는 클래식 우주 슈팅 게임입니다.',
        'objective': '소행성과 UFO를 모두 파괴해 최고 점수를 달성하세요.',
        'how_to_play': [
            '← : 우주선 반시계 방향 회전',
            '→ : 우주선 시계 방향 회전',
            '↑ : 앞쪽으로 추진 (관성으로 계속 이동)',
            '↓ : 워프 — 잠시 사라진 뒤 다른 위치에 재등장',
            '스페이스바: 전방 발사',
        ],
        'scoring': [
            '대형 소행성: 20점',
            '중형 소행성: 50점',
            '소형 소행성: 100점',
            'UFO(대): 200점 / UFO(소): 1000점',
        ],
        'end_condition': '목숨 3개를 모두 소진하면 게임이 종료됩니다.',
        'tips': [
            '큰 소행성을 쪼개면 작은 조각이 더 빠르게 날아오니 주의하세요.',
            '관성을 이용해 최소 연료로 이동하면 오래 생존할 수 있습니다.',
        ],
        'controls': {
            'up':    '전방 추진',
            'down':  '워프',
            'left':  '반시계 방향 회전',
            'right': '시계 방향 회전',
            'fire':  '전방 발사',
        },
    }
