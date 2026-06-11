"""games/mariobros.py — Mario Bros 게임 설정 (선언적 스펙 기반)"""
from games.atari import AtariGame

class MarioBrosGame(AtariGame):
    game_id    = 'mariobros'
    game_title = 'MARIO BROS'
    game_icon  = '🍄'
    env_name   = 'ALE/MarioBros-v5'
    frame_skip  = 4
    prefix     = 'mb_'
    theme_color = '#ff0000'
    model_path_parts = ('data', 'checkpoints', 'mariobros', 'best_model.pth')

    action_names = {
        0: 'NOOP',   1: 'FIRE',          2: 'UP',            3: 'RIGHT',
        4: 'LEFT',   5: 'DOWN',          6: 'UPRIGHT',       7: 'UPLEFT',
        8: 'DOWNRIGHT', 9: 'DOWNLEFT',  10: 'UPFIRE',       11: 'RIGHTFIRE',
        12: 'LEFTFIRE', 13: 'DOWNFIRE', 14: 'UPRIGHTFIRE',  15: 'UPLEFTFIRE',
        16: 'DOWNRIGHTFIRE', 17: 'DOWNLEFTFIRE',
    }

    keyboard_keys = [
        {'id': 'up',    'label': '↑',    'actions': [2,  6,  7, 10, 14, 15]},
        {'id': 'down',  'label': '↓',    'actions': [5,  8,  9, 13, 16, 17]},
        {'id': 'left',  'label': '←',    'actions': [4,  7,  9, 12, 15, 17]},
        {'id': 'right', 'label': '→',    'actions': [3,  6,  8, 11, 14, 16]},
        {'id': 'fire',  'label': 'SPACE', 'actions': [1, 10, 11, 12, 13, 14, 15, 16, 17]},
    ]

    key_combos = {
        'down+fire+right':  16,
        'down+fire+left':   17,
        'fire+right+up':    14,
        'fire+left+up':     15,
        'down+fire':        13,
        'down+right':        8,
        'down+left':         9,
        'fire+right':       11,
        'fire+left':        12,
        'fire+up':          10,
        'right+up':          6,
        'left+up':           7,
        'down':              5,
        'fire':              1,
        'left':              4,
        'right':             3,
        'up':                2,
        '':                  0,
    }

    achievement_specs = [
        # bronze
        {'id': 'score_200',     'title': '첫 적 처치',       'tier': 'bronze',   'desc': '200점 달성',
         'metric': 'score', 'value': 200},
        {'id': 'score_800',     'title': '배관공',            'tier': 'bronze',   'desc': '800점 달성',
         'metric': 'score', 'value': 800},
        {'id': 'combo_2',       'title': '콤보 킬 I',        'tier': 'bronze',   'desc': '빠르게 적 2마리 연속 처치',
         'metric': 'streak', 'value': 2, 'gap': 60},
        # silver
        {'id': 'score_2000',    'title': '마리오 전사',      'tier': 'silver',   'desc': '2000점 달성',
         'metric': 'score', 'value': 2000},
        {'id': 'score_4000',    'title': '버섯 왕국 수호자', 'tier': 'silver',   'desc': '4000점 달성',
         'metric': 'score', 'value': 4000},
        {'id': 'combo_3',       'title': '콤보 킬 II',       'tier': 'silver',   'desc': '빠르게 적 3마리 연속 처치',
         'metric': 'streak', 'value': 3, 'gap': 60},
        {'id': 'no_death_stage','title': '완벽한 스테이지',  'tier': 'silver',
         'desc': '목숨 잃지 않고 스테이지 클리어 (500스텝)',
         'metric': 'no_death', 'value': 500},
        # gold
        {'id': 'score_6000',    'title': '전설의 마리오',    'tier': 'gold',     'desc': '6000점 달성',
         'metric': 'score', 'value': 6000},
        {'id': 'bench_human',   'title': '인간 평균 돌파',   'tier': 'gold',     'desc': '인간 평균(7,777점) 초과',
         'metric': 'score', 'value': 7777, 'cmp': '>'},
        {'id': 'combo_5',       'title': '콤보 킬 III',      'tier': 'gold',     'desc': '빠르게 적 5마리 연속 처치',
         'metric': 'streak', 'value': 5, 'gap': 60},
        # platinum
        {'id': 'score_10000',   'title': '슈퍼 마리오 전설', 'tier': 'platinum', 'desc': '10000점 달성',
         'metric': 'score', 'value': 10000},
    ]

    game_info = {
        'summary': '하수구 배관에 나타난 적들을 처치하는 플랫폼 액션 게임입니다.',
        'objective': '각 라운드에 등장하는 모든 적을 처치해 라운드를 클리어하세요.',
        'how_to_play': [
            '← → : 좌우 이동',
            '스페이스바: 점프',
            '적 발판 아래서 점프 → 적이 뒤집힘 → 발로 차서 처치',
            '뒤집힌 적을 빨리 차지 않으면 다시 일어납니다.',
        ],
        'scoring': [
            '적 처치: 800점~',
            '동시 다수 처치 시 콤보 보너스',
            '라운드 클리어 보너스',
        ],
        'end_condition': '목숨 3개를 모두 소진하면 게임이 종료됩니다.',
        'tips': [
            '적을 뒤집은 직후 빠르게 처리하면 콤보 점수가 올라갑니다.',
            '같은 종류의 적을 연속 처치하면 보너스가 더 높습니다.',
        ],
        'controls': {
            'up':    '이동',
            'down':  '이동',
            'left':  '왼쪽 이동',
            'right': '오른쪽 이동',
            'fire':  '점프',
        },
    }
