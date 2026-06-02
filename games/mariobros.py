"""games/mariobros.py — Mario Bros 게임 설정"""
import os
from games.atari import AtariGame
from games.atari.ach_helper import episode_score, streak_max, life_losses

class MarioBrosGame(AtariGame):
    game_id    = 'mariobros'
    game_title = 'MARIO BROS'
    game_icon  = '🍄'
    env_name   = 'ALE/MarioBros-v5'
    frame_skip  = 4
    prefix     = 'mb_'
    theme_color = '#ff0000'
    model_path_parts = ('ai_agents', 'mariobros', 'checkpoints', 'best_model.pth')

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

    achievements = [
        # bronze
        {'id': 'score_200',     'title': '첫 적 처치',       'tier': 'bronze',   'desc': '200점 달성'},
        {'id': 'score_800',     'title': '배관공',            'tier': 'bronze',   'desc': '800점 달성'},
        {'id': 'combo_2',       'title': '콤보 킬 I',        'tier': 'bronze',   'desc': '빠르게 적 2마리 연속 처치'},
        # silver
        {'id': 'score_2000',    'title': '마리오 전사',      'tier': 'silver',   'desc': '2000점 달성'},
        {'id': 'score_4000',    'title': '버섯 왕국 수호자', 'tier': 'silver',   'desc': '4000점 달성'},
        {'id': 'combo_3',       'title': '콤보 킬 II',       'tier': 'silver',   'desc': '빠르게 적 3마리 연속 처치'},
        {'id': 'no_death_stage','title': '완벽한 스테이지',  'tier': 'silver',   'desc': '목숨 잃지 않고 스테이지 클리어 (500스텝)'},
        # gold
        {'id': 'score_6000',    'title': '전설의 마리오',    'tier': 'gold',     'desc': '6000점 달성'},
        {'id': 'bench_human',   'title': '인간 평균 돌파',   'tier': 'gold',     'desc': '인간 평균(7,777점) 초과'},
        {'id': 'combo_5',       'title': '콤보 킬 III',      'tier': 'gold',     'desc': '빠르게 적 5마리 연속 처치'},
        # platinum
        {'id': 'score_10000',   'title': '슈퍼 마리오 전설', 'tier': 'platinum', 'desc': '10000점 달성'},
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

        unlock('score_200',     '첫 적 처치',       '200점 달성',                   'bronze', s >= 200)
        unlock('score_800',     '배관공',            '800점 달성',                   'bronze', s >= 800)
        unlock('combo_2',       '콤보 킬 I',        '빠르게 적 2마리 연속 처치',    'bronze', sk >= 2)
        unlock('score_2000',    '마리오 전사',      '2000점 달성',                  'silver', s >= 2000)
        unlock('score_4000',    '버섯 왕국 수호자', '4000점 달성',                  'silver', s >= 4000)
        unlock('combo_3',       '콤보 킬 II',       '빠르게 적 3마리 연속 처치',   'silver', sk >= 3)
        unlock('no_death_stage','완벽한 스테이지',  '목숨 잃지 않고 스테이지 클리어 (500스텝)', 'silver',
               steps_total >= 500 and life_losses(self.episode_data) == 0)
        unlock('score_6000',    '전설의 마리오',    '6000점 달성',                  'gold', s >= 6000)
        unlock('bench_human',   '인간 평균 돌파',   '인간 평균(7,777점) 초과',      'gold', s > 7777)
        unlock('combo_5',       '콤보 킬 III',      '빠르게 적 5마리 연속 처치',   'gold', sk >= 5)
        unlock('score_10000',   '슈퍼 마리오 전설', '10000점 달성',                 'platinum', s >= 10000)
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

        add('score_200',     '첫 적 처치',       '200점 달성',                   'bronze', s >= 200)
        add('score_800',     '배관공',            '800점 달성',                   'bronze', s >= 800)
        add('combo_2',       '콤보 킬 I',        '빠르게 적 2마리 연속 처치',    'bronze', sk >= 2)
        add('score_2000',    '마리오 전사',      '2000점 달성',                  'silver', s >= 2000)
        add('score_4000',    '버섯 왕국 수호자', '4000점 달성',                  'silver', s >= 4000)
        add('combo_3',       '콤보 킬 II',       '빠르게 적 3마리 연속 처치',   'silver', sk >= 3)
        add('no_death_stage','완벽한 스테이지',  '목숨 잃지 않고 스테이지 클리어 (500스텝)', 'silver', ll == 0 and steps >= 500)
        add('score_6000',    '전설의 마리오',    '6000점 달성',                  'gold', s >= 6000)
        add('bench_human',   '인간 평균 돌파',   '인간 평균(7,777점) 초과',      'gold', s > 7777)
        add('combo_5',       '콤보 킬 III',      '빠르게 적 5마리 연속 처치',   'gold', sk >= 5)
        add('score_10000',   '슈퍼 마리오 전설', '10000점 달성',                 'platinum', s >= 10000)
        return achieved

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.d3qn_helper import load_d3qn
        net, _ = load_d3qn('mariobros', path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.d3qn_helper import get_q_values
        return get_q_values(self.net, stacked_state, self.device)

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
