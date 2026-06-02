"""games/asterix.py — Asterix 게임 설정"""
import os
from games.atari import AtariGame
from games.atari.ach_helper import episode_score, streak_max, life_losses

class AsterixGame(AtariGame):
    game_id    = 'asterix'
    game_title = 'ASTERIX'
    game_icon  = '⭐'
    env_name   = 'ALE/Asterix-v5'
    frame_skip  = 4
    prefix     = 'ax_'
    theme_color = '#ffd700'
    model_path_parts = ('ai_agents', 'asterix', 'checkpoints', 'best_model.pth')

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

    achievements = [
        # bronze
        {'id': 'score_500',    'title': '첫 별 수집',      'tier': 'bronze',   'desc': '500점 달성'},
        {'id': 'score_1500',   'title': '별 사냥꾼',        'tier': 'bronze',   'desc': '1500점 달성'},
        {'id': 'streak_5',     'title': '연속 획득 I',      'tier': 'bronze',   'desc': '빠르게 5연속 아이템 획득'},
        # silver
        {'id': 'score_4000',   'title': '아스테릭스',       'tier': 'silver',   'desc': '4000점 달성'},
        {'id': 'score_6000',   'title': '용감한 전사',      'tier': 'silver',   'desc': '6000점 달성'},
        {'id': 'streak_10',    'title': '연속 획득 II',     'tier': 'silver',   'desc': '빠르게 10연속 아이템 획득'},
        {'id': 'no_death_1000','title': '귀신 같은 회피',   'tier': 'silver',   'desc': '목숨 잃지 않고 1000스텝 생존'},
        # gold
        {'id': 'bench_dqn',    'title': 'DQN 돌파',         'tier': 'gold',     'desc': 'DQN 기준(6,012점) 초과'},
        {'id': 'bench_human',  'title': '인간 평균 돌파',   'tier': 'gold',     'desc': '인간 평균(8,503점) 초과'},
        {'id': 'score_12000',  'title': '전설의 전사',      'tier': 'gold',     'desc': '12000점 달성'},
        # platinum
        {'id': 'bench_ddqn',   'title': 'DDQN 돌파',        'tier': 'platinum', 'desc': 'DDQN 기준(15,150점) 초과'},
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
        lives  = entry.get('lives')
        rt['score'] += reward
        if lives is not None and hasattr(self, '_rt_prev_lives') and self._rt_prev_lives is not None and lives < self._rt_prev_lives:
            rt['streak'] = 0
            rt['last_kill_step'] = -1
        if lives is not None:
            self._rt_prev_lives = lives
        if reward > 0:
            gap = step - rt['last_kill_step'] if rt['last_kill_step'] >= 0 else 999
            rt['streak'] = rt['streak'] + 1 if gap <= 60 else 1
            rt['max_streak'] = max(rt['max_streak'], rt['streak'])
            rt['last_kill_step'] = step
        s   = rt['score']
        sk  = rt['max_streak']
        steps_total = len(self.episode_data)

        unlock('score_500',    '첫 별 수집',      '500점 달성',                   'bronze', s >= 500)
        unlock('score_1500',   '별 사냥꾼',        '1500점 달성',                  'bronze', s >= 1500)
        unlock('streak_5',     '연속 획득 I',      '빠르게 5연속 아이템 획득',     'bronze', sk >= 5)
        unlock('score_4000',   '아스테릭스',       '4000점 달성',                  'silver', s >= 4000)
        unlock('score_6000',   '용감한 전사',      '6000점 달성',                  'silver', s >= 6000)
        unlock('streak_10',    '연속 획득 II',     '빠르게 10연속 아이템 획득',    'silver', sk >= 10)
        unlock('no_death_1000','귀신 같은 회피',   '목숨 잃지 않고 1000스텝 생존', 'silver',
               steps_total >= 1000 and life_losses(self.episode_data) == 0)
        unlock('bench_dqn',    'DQN 돌파',         'DQN 기준(6,012점) 초과',       'gold', s > 6012)
        unlock('bench_human',  '인간 평균 돌파',   '인간 평균(8,503점) 초과',      'gold', s > 8503)
        unlock('score_12000',  '전설의 전사',      '12000점 달성',                 'gold', s >= 12000)
        unlock('bench_ddqn',   'DDQN 돌파',        'DDQN 기준(15,150점) 초과',     'platinum', s > 15150)
        return new

    def _compute_achievements(self) -> list:
        data = [d for d in self.episode_data if d.get('action') is not None]
        if not data:
            return []
        s   = episode_score(data)
        sk  = streak_max(data, gap=60, min_r=0, max_r=1e9)
        ll  = life_losses(data)
        steps = len(data)
        achieved = []

        def add(id_, title, desc, tier, cond):
            if cond:
                achieved.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        add('score_500',    '첫 별 수집',      '500점 달성',                   'bronze', s >= 500)
        add('score_1500',   '별 사냥꾼',        '1500점 달성',                  'bronze', s >= 1500)
        add('streak_5',     '연속 획득 I',      '빠르게 5연속 아이템 획득',     'bronze', sk >= 5)
        add('score_4000',   '아스테릭스',       '4000점 달성',                  'silver', s >= 4000)
        add('score_6000',   '용감한 전사',      '6000점 달성',                  'silver', s >= 6000)
        add('streak_10',    '연속 획득 II',     '빠르게 10연속 아이템 획득',    'silver', sk >= 10)
        add('no_death_1000','귀신 같은 회피',   '목숨 잃지 않고 1000스텝 생존', 'silver', ll == 0 and steps >= 1000)
        add('bench_dqn',    'DQN 돌파',         'DQN 기준(6,012점) 초과',       'gold', s > 6012)
        add('bench_human',  '인간 평균 돌파',   '인간 평균(8,503점) 초과',      'gold', s > 8503)
        add('score_12000',  '전설의 전사',      '12000점 달성',                 'gold', s >= 12000)
        add('bench_ddqn',   'DDQN 돌파',        'DDQN 기준(15,150점) 초과',     'platinum', s > 15150)
        return achieved

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.d3qn_helper import load_d3qn
        net, _ = load_d3qn('asterix', path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.d3qn_helper import get_q_values
        return get_q_values(self.net, stacked_state, self.device)

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
