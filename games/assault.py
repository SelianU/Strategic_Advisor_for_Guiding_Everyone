"""games/assault.py — Assault 게임 설정 (선언적 스펙 기반)"""
from games.atari import AtariGame

class AssaultGame(AtariGame):
    game_id    = 'assault'
    game_title = 'ASSAULT'
    game_icon  = '🚀'
    env_name   = 'ALE/Assault-v5'
    frame_skip  = 4
    prefix     = 'as_'
    theme_color = '#00ff88'
    model_path_parts = ('data', 'checkpoints', 'assault', 'best_model_assault.pth')

    action_names = {
        0: 'NOOP',  1: 'FIRE',  2: 'UP',  3: 'RIGHT',
        4: 'LEFT',  5: 'RIGHTFIRE',  6: 'LEFTFIRE',
    }

    keyboard_keys = [
        {'id': 'up',    'label': '↑',    'actions': [2]},
        {'id': 'left',  'label': '←',    'actions': [4, 6]},
        {'id': 'right', 'label': '→',    'actions': [3, 5]},
        {'id': 'fire',  'label': 'SPACE', 'actions': [1, 5, 6]},
    ]

    key_combos = {
        'fire+right': 5,
        'fire+left':  6,
        'up':         2,
        'right':      3,
        'left':       4,
        'fire':       1,
        '':           0,
    }

    achievement_specs = [
        # bronze
        {'id': 'score_200',  'title': '첫 포격',       'tier': 'bronze',   'desc': '200점 달성',
         'metric': 'score', 'value': 200},
        {'id': 'score_600',  'title': '포격수',         'tier': 'bronze',   'desc': '600점 달성',
         'metric': 'score', 'value': 600},
        {'id': 'combo_3',    'title': '집중 포화 I',    'tier': 'bronze',   'desc': '200스텝 내 3회 명중',
         'metric': 'combo', 'value': 3, 'window': 200},
        # silver
        {'id': 'bench_human','title': '인간 평균 돌파', 'tier': 'silver',   'desc': '인간 평균(1,496점) 초과',
         'metric': 'score', 'value': 1496, 'cmp': '>'},
        {'id': 'score_1500', 'title': '에이스 포수',    'tier': 'silver',   'desc': '1500점 달성',
         'metric': 'score', 'value': 1500},
        {'id': 'combo_6',    'title': '집중 포화 II',   'tier': 'silver',   'desc': '200스텝 내 6회 명중',
         'metric': 'combo', 'value': 6, 'window': 200},
        {'id': 'wave_clear', 'title': '웨이브 격파',    'tier': 'silver',   'desc': '연속 5회 명중',
         'metric': 'streak', 'value': 5, 'gap': 60},
        # gold
        {'id': 'score_3000', 'title': '맹공격',         'tier': 'gold',     'desc': '3000점 달성',
         'metric': 'score', 'value': 3000},
        {'id': 'combo_10',   'title': '집중 포화 III',  'tier': 'gold',     'desc': '200스텝 내 10회 명중',
         'metric': 'combo', 'value': 10, 'window': 200},
        {'id': 'bench_dqn',  'title': 'DQN 돌파',       'tier': 'gold',     'desc': 'DQN 기준(3,359점) 초과',
         'metric': 'score', 'value': 3359, 'cmp': '>'},
        # platinum
        {'id': 'bench_ddqn', 'title': 'DDQN 돌파',      'tier': 'platinum', 'desc': 'DDQN 기준(5,023점) 초과',
         'metric': 'score', 'value': 5023, 'cmp': '>'},
    ]

    game_info = {
        'summary': '지상 포대로 하늘에서 침략하는 적 함선들을 격추하는 슈팅 게임입니다.',
        'objective': '화면을 가득 채운 적 UFO와 포탑을 최대한 많이 파괴하세요.',
        'how_to_play': [
            '방향키(←→)로 포대를 좌우로 이동합니다.',
            '스페이스바로 미사일을 발사해 적을 격추합니다.',
            '↑ + 스페이스바 조합으로 고각 발사도 가능합니다.',
        ],
        'scoring': [
            '일반 UFO 격추: 10~25점',
            '중간 포탑 격추: 80점',
            '고속 UFO 격추: 150점',
        ],
        'end_condition': '목숨 3개를 모두 소진하면 게임이 종료됩니다.',
        'tips': [
            '화면 중앙의 큰 적을 먼저 제거해 위협을 줄이세요.',
            'AI 에이전트의 발사 타이밍 패턴을 관찰해 보세요.',
        ],
        'controls': {
            'up':    '위로 발사',
            'left':  '왼쪽 이동',
            'right': '오른쪽 이동',
            'fire':  '앞 발사 (←/→ 동시: 사선 발사)',
        },
    }
