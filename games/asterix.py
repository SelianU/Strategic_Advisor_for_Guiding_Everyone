"""games/asterix.py — Asterix 게임 설정 (선언적 스펙 기반)"""
from games.atari import AtariGame

class AsterixGame(AtariGame):
    game_id    = 'asterix'
    game_title = 'ASTERIX'
    game_icon  = '⭐'
    env_name   = 'ALE/Asterix-v5'
    frame_skip  = 4
    prefix     = 'ax_'
    theme_color = '#ffd700'
    model_path_parts = ('data', 'checkpoints', 'asterix', 'best_model_asterix.pth')

    action_names = {
        0: 'NOOP',  1: 'UP',    2: 'RIGHT',     3: 'LEFT',
        4: 'DOWN',  5: 'UPRIGHT', 6: 'UPLEFT',  7: 'DOWNRIGHT',
        8: 'DOWNLEFT',
    }

    keyboard_keys = [
        {'id': 'up',    'label': '↑',    'actions': [1, 5, 6]},
        {'id': 'down',  'label': '↓',    'actions': [4, 7, 8]},
        {'id': 'left',  'label': '←',    'actions': [3, 6, 8]},
        {'id': 'right', 'label': '→',    'actions': [2, 5, 7]},
    ]

    key_combos = {
        'right+up':  5,
        'left+up':   6,
        'down+right': 7,
        'down+left':  8,
        'up':         1,
        'right':      2,
        'left':       3,
        'down':       4,
        '':           0,
    }

    achievement_specs = [
        # bronze
        {'id': 'score_500',    'title': '첫 별 수집',      'tier': 'bronze',   'desc': '500점 달성',
         'metric': 'score', 'value': 500},
        {'id': 'score_1500',   'title': '별 사냥꾼',        'tier': 'bronze',   'desc': '1500점 달성',
         'metric': 'score', 'value': 1500},
        {'id': 'streak_5',     'title': '연속 획득 I',      'tier': 'bronze',   'desc': '빠르게 5연속 아이템 획득',
         'metric': 'streak', 'value': 5, 'gap': 60},
        # silver
        {'id': 'score_4000',   'title': '아스테릭스',       'tier': 'silver',   'desc': '4000점 달성',
         'metric': 'score', 'value': 4000},
        {'id': 'score_6000',   'title': '용감한 전사',      'tier': 'silver',   'desc': '6000점 달성',
         'metric': 'score', 'value': 6000},
        {'id': 'streak_10',    'title': '연속 획득 II',     'tier': 'silver',   'desc': '빠르게 10연속 아이템 획득',
         'metric': 'streak', 'value': 10, 'gap': 60},
        {'id': 'no_death_1000','title': '귀신 같은 회피',   'tier': 'silver',   'desc': '목숨 잃지 않고 1000스텝 생존',
         'metric': 'no_death', 'value': 1000},
        # gold
        {'id': 'bench_dqn',    'title': 'DQN 돌파',         'tier': 'gold',     'desc': 'DQN 기준(6,012점) 초과',
         'metric': 'score', 'value': 6012, 'cmp': '>'},
        {'id': 'bench_human',  'title': '인간 평균 돌파',   'tier': 'gold',     'desc': '인간 평균(8,503점) 초과',
         'metric': 'score', 'value': 8503, 'cmp': '>'},
        {'id': 'score_12000',  'title': '전설의 전사',      'tier': 'gold',     'desc': '12000점 달성',
         'metric': 'score', 'value': 12000},
        # platinum
        {'id': 'bench_ddqn',   'title': 'DDQN 돌파',        'tier': 'platinum', 'desc': 'DDQN 기준(15,150점) 초과',
         'metric': 'score', 'value': 15150, 'cmp': '>'},
    ]

    game_info = {
        'summary': '마법 물약을 수집하며 끝없이 이동하는 적을 피하는 아케이드 게임입니다.',
        'objective': '화면 속 반짝이는 물약을 최대한 많이 수집하세요.',
        'how_to_play': [
            '방향키(↑↓←→)로 캐릭터를 이동합니다.',
            '물약(빛나는 아이템)에 닿으면 자동으로 수집됩니다.',
            '적 캐릭터와 무기에는 절대 닿지 마세요.',
        ],
        'scoring': [
            '물약 수집: 10점',
            '빠른 연속 수집 시 보너스 점수',
        ],
        'end_condition': '목숨 3개를 모두 소진하면 게임이 종료됩니다.',
        'tips': [
            '물약과 적의 움직임 패턴을 파악해 안전한 경로를 선택하세요.',
            '화면 가장자리 루트를 활용하면 적을 쉽게 피할 수 있습니다.',
        ],
        'controls': {
            'up':    '위로 이동',
            'down':  '아래로 이동',
            'left':  '왼쪽 이동',
            'right': '오른쪽 이동',
        },
    }
