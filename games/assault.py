"""games/assault.py — Assault 게임 설정"""
import os
from games.atari import AtariGame
from games.atari.ach_helper import episode_score, combo_max, streak_max

class AssaultGame(AtariGame):
    game_id    = 'assault'
    game_title = 'ASSAULT'
    game_icon  = '🚀'
    env_name   = 'ALE/Assault-v5'
    frame_skip  = 4
    prefix     = 'as_'
    theme_color = '#00ff88'
    model_path_parts = ('ai_agents', 'assault', 'checkpoints', 'best_model_assault.pth')

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

    achievements = [
        # bronze
        {'id': 'score_200',  'title': '첫 포격',       'tier': 'bronze',   'desc': '200점 달성'},
        {'id': 'score_600',  'title': '포격수',         'tier': 'bronze',   'desc': '600점 달성'},
        {'id': 'combo_3',    'title': '집중 포화 I',    'tier': 'bronze',   'desc': '200스텝 내 3회 명중'},
        # silver
        {'id': 'bench_human','title': '인간 평균 돌파', 'tier': 'silver',   'desc': '인간 평균(1,496점) 초과'},
        {'id': 'score_1500', 'title': '에이스 포수',    'tier': 'silver',   'desc': '1500점 달성'},
        {'id': 'combo_6',    'title': '집중 포화 II',   'tier': 'silver',   'desc': '200스텝 내 6회 명중'},
        {'id': 'wave_clear', 'title': '웨이브 격파',    'tier': 'silver',   'desc': '연속 5회 명중'},
        # gold
        {'id': 'score_3000', 'title': '맹공격',         'tier': 'gold',     'desc': '3000점 달성'},
        {'id': 'combo_10',   'title': '집중 포화 III',  'tier': 'gold',     'desc': '200스텝 내 10회 명중'},
        {'id': 'bench_dqn',  'title': 'DQN 돌파',       'tier': 'gold',     'desc': 'DQN 기준(3,359점) 초과'},
        # platinum
        {'id': 'bench_ddqn', 'title': 'DDQN 돌파',      'tier': 'platinum', 'desc': 'DDQN 기준(5,023점) 초과'},
    ]

    def _on_episode_reset(self):
        self._rt = {
            'score': 0.0, 'kill_window': [], 'streak': 0,
            'max_streak': 0, 'last_kill_step': -1,
        }

    def _check_realtime_achievements(self, entry: dict) -> list:
        if not hasattr(self, '_rt'):
            return []
        rt = self._rt
        new = []

        def unlock(id_, title, desc, tier, condition):
            if condition and id_ not in self._ach_unlocked:
                self._ach_unlocked.add(id_)
                new.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        reward = entry.get('reward', 0)
        step   = entry.get('step', 0)
        rt['score'] += reward
        if reward > 0:
            rt['kill_window'].append(step)
            gap = step - rt['last_kill_step'] if rt['last_kill_step'] >= 0 else 999
            rt['streak'] = rt['streak'] + 1 if gap <= 60 else 1
            rt['max_streak'] = max(rt['max_streak'], rt['streak'])
            rt['last_kill_step'] = step
        rt['kill_window'] = [s for s in rt['kill_window'] if step - s <= 200]
        s  = rt['score']
        kw = rt['kill_window']
        sk = rt['max_streak']

        unlock('score_200',  '첫 포격',       '200점 달성',               'bronze', s >= 200)
        unlock('score_600',  '포격수',         '600점 달성',               'bronze', s >= 600)
        unlock('combo_3',    '집중 포화 I',    '200스텝 내 3회 명중',      'bronze', len(kw) >= 3)
        unlock('bench_human','인간 평균 돌파', '인간 평균(1,496점) 초과',    'silver', s > 1496)
        unlock('score_1500', '에이스 포수',    '1500점 달성',              'silver', s >= 1500)
        unlock('combo_6',    '집중 포화 II',   '200스텝 내 6회 명중',      'silver', len(kw) >= 6)
        unlock('wave_clear', '웨이브 격파',    '연속 5회 명중',            'silver', sk >= 5)
        unlock('score_3000', '맹공격',         '3000점 달성',              'gold', s >= 3000)
        unlock('combo_10',   '집중 포화 III',  '200스텝 내 10회 명중',     'gold', len(kw) >= 10)
        unlock('bench_dqn',  'DQN 돌파',       'DQN 기준(3,359점) 초과',   'gold', s > 3359)
        unlock('bench_ddqn', 'DDQN 돌파',      'DDQN 기준(5,023점) 초과', 'platinum', s > 5023)
        return new

    def _compute_achievements(self) -> list:
        data = [d for d in self.episode_data if d.get('action') is not None]
        if not data:
            return []
        s  = episode_score(data)
        cm = combo_max(data, window=200, min_r=0, max_r=1e9)
        sk = streak_max(data, gap=60, min_r=0, max_r=1e9)
        achieved = []

        def add(id_, title, desc, tier, cond):
            if cond:
                achieved.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        add('score_200',  '첫 포격',       '200점 달성',               'bronze', s >= 200)
        add('score_600',  '포격수',         '600점 달성',               'bronze', s >= 600)
        add('combo_3',    '집중 포화 I',    '200스텝 내 3회 명중',      'bronze', cm >= 3)
        add('bench_human','인간 평균 돌파', '인간 평균(1,496점) 초과',    'silver', s > 1496)
        add('score_1500', '에이스 포수',    '1500점 달성',              'silver', s >= 1500)
        add('combo_6',    '집중 포화 II',   '200스텝 내 6회 명중',      'silver', cm >= 6)
        add('wave_clear', '웨이브 격파',    '연속 5회 명중',            'silver', sk >= 5)
        add('score_3000', '맹공격',         '3000점 달성',              'gold', s >= 3000)
        add('combo_10',   '집중 포화 III',  '200스텝 내 10회 명중',     'gold', cm >= 10)
        add('bench_dqn',  'DQN 돌파',       'DQN 기준(3,359점) 초과',   'gold', s > 3359)
        add('bench_ddqn', 'DDQN 돌파',      'DDQN 기준(5,023점) 초과', 'platinum', s > 5023)
        return achieved

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.d3qn_helper import load_d3qn
        net, _ = load_d3qn('assault', path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.d3qn_helper import get_q_values
        return get_q_values(self.net, stacked_state, self.device)

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
