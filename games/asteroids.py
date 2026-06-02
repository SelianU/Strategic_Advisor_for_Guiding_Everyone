"""games/asteroids.py — Asteroids 게임 설정"""
import os
from games.atari import AtariGame
from games.atari.ach_helper import episode_score, combo_max

class AsteroidsGame(AtariGame):
    game_id    = 'asteroids'
    game_title = 'ASTEROIDS'
    game_icon  = '☄️'
    env_name   = 'ALE/Asteroids-v5'
    frame_skip  = 4
    prefix     = 'ao_'
    theme_color = '#00bfff'
    model_path_parts = ('ai_agents', 'asteroids', 'checkpoints', 'best_model.pth')

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

    achievements = [
        # bronze
        {'id': 'score_300',   'title': '첫 충돌',        'tier': 'bronze',   'desc': '300점 달성'},
        {'id': 'score_1000',  'title': '우주 청소부',    'tier': 'bronze',   'desc': '1000점 달성'},
        {'id': 'combo_3',     'title': '연속 격파 I',    'tier': 'bronze',   'desc': '200스텝 내 소행성 3개 격파'},
        # silver
        {'id': 'score_3000',  'title': '유성 사냥꾼',    'tier': 'silver',   'desc': '3000점 달성'},
        {'id': 'score_6000',  'title': '소행성 파괴자',  'tier': 'silver',   'desc': '6000점 달성'},
        {'id': 'combo_6',     'title': '연속 격파 II',   'tier': 'silver',   'desc': '200스텝 내 소행성 6개 격파'},
        {'id': 'survive_500', 'title': '생존자',          'tier': 'silver',   'desc': '500스텝 생존'},
        # gold
        {'id': 'score_10000', 'title': '별의 지배자',    'tier': 'gold',     'desc': '10000점 달성'},
        {'id': 'combo_10',    'title': '연속 격파 III',  'tier': 'gold',     'desc': '200스텝 내 소행성 10개 격파'},
        {'id': 'bench_human', 'title': '인간 평균 돌파', 'tier': 'gold',     'desc': '인간 평균(13,157점) 초과'},
        # platinum
        {'id': 'score_20000', 'title': '우주 전설',      'tier': 'platinum', 'desc': '20000점 달성'},
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
        rt['kill_window'] = [s for s in rt['kill_window'] if step - s <= 200]
        s  = rt['score']
        kw = rt['kill_window']
        steps_total = len(self.episode_data)

        unlock('score_300',   '첫 충돌',        '300점 달성',                   'bronze', s >= 300)
        unlock('score_1000',  '우주 청소부',    '1000점 달성',                  'bronze', s >= 1000)
        unlock('combo_3',     '연속 격파 I',    '200스텝 내 소행성 3개 격파',   'bronze', len(kw) >= 3)
        unlock('score_3000',  '유성 사냥꾼',    '3000점 달성',                  'silver', s >= 3000)
        unlock('score_6000',  '소행성 파괴자',  '6000점 달성',                  'silver', s >= 6000)
        unlock('combo_6',     '연속 격파 II',   '200스텝 내 소행성 6개 격파',   'silver', len(kw) >= 6)
        unlock('survive_500', '생존자',          '500스텝 생존',                 'silver', steps_total >= 500)
        unlock('score_10000', '별의 지배자',    '10000점 달성',                 'gold', s >= 10000)
        unlock('combo_10',    '연속 격파 III',  '200스텝 내 소행성 10개 격파',  'gold', len(kw) >= 10)
        unlock('bench_human', '인간 평균 돌파', '인간 평균(13,157점) 초과',     'gold', s > 13157)
        unlock('score_20000', '우주 전설',      '20000점 달성',                 'platinum', s >= 20000)
        return new

    def _compute_achievements(self) -> list:
        data = [d for d in self.episode_data if d.get('action') is not None]
        if not data:
            return []
        s   = episode_score(data)
        cm  = combo_max(data, window=200, min_r=0, max_r=1e9)
        steps = len(data)
        achieved = []

        def add(id_, title, desc, tier, cond):
            if cond:
                achieved.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        add('score_300',   '첫 충돌',        '300점 달성',                   'bronze', s >= 300)
        add('score_1000',  '우주 청소부',    '1000점 달성',                  'bronze', s >= 1000)
        add('combo_3',     '연속 격파 I',    '200스텝 내 소행성 3개 격파',   'bronze', cm >= 3)
        add('score_3000',  '유성 사냥꾼',    '3000점 달성',                  'silver', s >= 3000)
        add('score_6000',  '소행성 파괴자',  '6000점 달성',                  'silver', s >= 6000)
        add('combo_6',     '연속 격파 II',   '200스텝 내 소행성 6개 격파',   'silver', cm >= 6)
        add('survive_500', '생존자',          '500스텝 생존',                 'silver', steps >= 500)
        add('score_10000', '별의 지배자',    '10000점 달성',                 'gold', s >= 10000)
        add('combo_10',    '연속 격파 III',  '200스텝 내 소행성 10개 격파',  'gold', cm >= 10)
        add('bench_human', '인간 평균 돌파', '인간 평균(13,157점) 초과',     'gold', s > 13157)
        add('score_20000', '우주 전설',      '20000점 달성',                 'platinum', s >= 20000)
        return achieved

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.d3qn_helper import load_d3qn
        net, _ = load_d3qn('asteroids', path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.d3qn_helper import get_q_values
        return get_q_values(self.net, stacked_state, self.device)

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
