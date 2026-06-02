"""games/space_invaders.py — Space Invaders 게임 설정"""
import os
import numpy as np
from games.atari import AtariGame


_SHIELD_STARTS = [42, 74, 106]  # 방패 3개 X 시작 좌표 (픽셀)
_SHIELD_WIDTH  = 8              # 방패 감지 폭
_SHIELD_MAX_PX = 112            # 완전한 방패 1개의 최대 픽셀 수


def extract_si_game_state(ram: np.ndarray | None, rgb_frame: np.ndarray | None) -> dict:
    """
    Space Invaders 게임 상태 피처 추출.
    RAM 기반: enemy_count, enemy_lowest_row, danger_distance,
              player_x_pct, player_under_shield, incoming_proximity
    RGB 기반: shield_integrity, cleared_columns, bullet_over_shield
    """
    if ram is None and rgb_frame is None:
        return {}

    W = 160

    # ── RAM 기반 피처 ────────────────────────────────────────────────────────
    if ram is not None:
        enemy_count    = int(ram[0x11])
        player_x       = int(ram[0x1C])
        player_x_pct   = round(player_x * 100 // W)
        enemy_lowest_y = 40 + int(ram[0x1A]) * 2
        danger_distance = max(0, 145 - enemy_lowest_y) if enemy_count > 0 else 999
        player_under_shield = any(
            sx <= player_x <= sx + _SHIELD_WIDTH for sx in _SHIELD_STARTS
        )
        live_bullets = [int(ram[a]) for a in (0x4C, 0x4E, 0x50) if ram[a] > 0]
        incoming_proximity = (
            max(0, 23 - max(live_bullets) * 23 // 80) if live_bullets else 23
        )
    else:
        # 구 세션 로드 등 RAM 없는 경우 — RGB 폴백
        brightness   = rgb_frame.max(axis=2)
        active_fb    = brightness > 40
        enemy_zone   = active_fb[20:145, :]
        enemy_rows   = np.where(enemy_zone.sum(axis=1) > 3)[0]
        if len(enemy_rows) > 0:
            enemy_lowest_y = int(enemy_rows.max()) + 20
            enemy_count    = max(1, min(55, int(enemy_zone.sum()) // 45))
        else:
            enemy_lowest_y = 0
            enemy_count    = 0
        danger_distance = max(0, 145 - enemy_lowest_y) if enemy_count > 0 else 999
        player_cols  = np.where(active_fb[168:182, :].any(axis=0))[0]
        player_x     = int(player_cols.mean()) if len(player_cols) > 0 else W // 2
        player_x_pct = round(player_x * 100 // W)
        player_under_shield = any(
            sx <= player_x <= sx + _SHIELD_WIDTH for sx in _SHIELD_STARTS
        )
        bullet_rows = np.where(active_fb[145:168, :].any(axis=1))[0]
        incoming_proximity = (168 - (int(bullet_rows.max()) + 145)) if len(bullet_rows) > 0 else 23

    # ── RGB 기반 피처 (방패 손상도 · 클리어 열 · 탄환-방패 겹침) ─────────────
    if rgb_frame is not None:
        brightness  = rgb_frame.max(axis=2)
        active      = brightness > 40
        shield_zone = active[142:175, :]

        if enemy_count > 0 and enemy_lowest_y >= 140:
            shield_integrity = [0, 0, 0]
        else:
            shield_integrity = [
                min(100, int(shield_zone[:, sx:sx + _SHIELD_WIDTH].sum()) * 100 // _SHIELD_MAX_PX)
                for sx in _SHIELD_STARTS
            ]

        enemy_zone    = active[20:145, :]
        col_has_enemy = enemy_zone.any(axis=0)
        col_size      = max(1, W // 11)
        cleared_columns = sum(
            1 for i in range(11)
            if not col_has_enemy[i * col_size:(i + 1) * col_size].any()
        )

        bullet_zone        = active[145:168, :]
        bullet_over_shield = False
        if bullet_zone.any():
            for bx in np.where(bullet_zone.any(axis=0))[0]:
                if any(sx <= bx <= sx + _SHIELD_WIDTH for sx in _SHIELD_STARTS):
                    bullet_over_shield = True
                    break
    else:
        shield_integrity   = [0, 0, 0]
        cleared_columns    = 0
        bullet_over_shield = False

    # ── 파생 피처 ────────────────────────────────────────────────────────────
    if enemy_count <= 3:
        speed_phase = 'critical'
    elif enemy_count <= 9:
        speed_phase = 'fast'
    else:
        speed_phase = 'normal'

    return {
        'enemy_count':         enemy_count,
        'enemy_lowest_row':    enemy_lowest_y,
        'danger_distance':     danger_distance,
        'shield_integrity':    shield_integrity,
        'player_x_pct':        player_x_pct,
        'player_under_shield': player_under_shield,
        'enemy_speed_phase':   speed_phase,
        'cleared_columns':     cleared_columns,
        'incoming_proximity':  incoming_proximity,
        'bullet_over_shield':  bullet_over_shield,
    }


class SpaceInvadersGame(AtariGame):
    game_id    = 'space_invaders'
    game_title = 'SPACE INVADERS'
    game_icon  = '👾'
    env_name   = 'ALE/SpaceInvaders-v5'
    frame_skip  = 15   # 에이전트 학습/분석 기준
    prefix     = 'si_'
    theme_color = '#39ff14'
    model_path_parts = ('ai_agents', 'space_invaders', 'checkpoints', 'best_model_insane.pth')

    action_names = {
        0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT', 4: 'RIGHTFIRE', 5: 'LEFTFIRE',
    }

    keyboard_keys = [
        {'id': 'left',  'label': '←',    'actions': [3, 5]},       # LEFT, LEFTFIRE
        {'id': 'right', 'label': '→',    'actions': [2, 4]},       # RIGHT, RIGHTFIRE
        {'id': 'fire',  'label': 'SPACE', 'actions': [1, 4, 5]},    # FIRE, RIGHTFIRE, LEFTFIRE
    ]

    key_combos = {
        'fire+left':  5,   # LEFTFIRE  (알파벳순 정렬)
        'fire+right': 4,   # RIGHTFIRE
        'left':       3,
        'right':      2,
        'fire':       1,
        '':           0,
    }

    @property
    def url_path(self):
        return '/space-invaders'   # 기존 URL 유지

    achievements = [
        # bronze
        {'id': 'score_100',  'title': '첫 발걸음',      'tier': 'bronze',   'desc': '100점 달성'},
        {'id': 'score_300',  'title': '자리 잡기',       'tier': 'bronze',   'desc': '300점 달성'},
        {'id': 'combo_3',    'title': '연속 격추 I',     'tier': 'bronze',   'desc': '200스텝 내 3마리 격추'},
        {'id': 'streak_3',   'title': '연사 콤보 I',     'tier': 'bronze',   'desc': '간격 50스텝 이내 3연속 격추'},
        {'id': 'round_1',    'title': '1라운드 클리어',  'tier': 'bronze',   'desc': '1라운드 적 전멸'},
        # silver
        {'id': 'score_500',  'title': '격투사',          'tier': 'silver',   'desc': '500점 달성'},
        {'id': 'score_1000', 'title': '에이스',          'tier': 'silver',   'desc': '1000점 달성'},
        {'id': 'combo_5',    'title': '연속 격추 II',    'tier': 'silver',   'desc': '200스텝 내 5마리 격추'},
        {'id': 'streak_5',   'title': '연사 콤보 II',    'tier': 'silver',   'desc': '간격 50스텝 이내 5연속 격추'},
        {'id': 'ufo_1',      'title': 'UFO 격추',        'tier': 'silver',   'desc': '미스터리 쉽을 격추'},
        {'id': 'final_3',    'title': '핀포인트',        'tier': 'silver',   'desc': '적 3마리 이하 남았을 때 격추'},
        {'id': 'no_death_1', 'title': '완벽한 수비',     'tier': 'silver',   'desc': '목숨 잃지 않고 1라운드 클리어'},
        {'id': 'round_2',    'title': '2라운드 클리어',  'tier': 'silver',   'desc': '2라운드 적 전멸'},
        # gold
        {'id': 'score_2000', 'title': '전설의 사수',     'tier': 'gold',     'desc': '2000점 달성'},
        {'id': 'score_3000', 'title': '우주 최강',       'tier': 'gold',     'desc': '3000점 달성'},
        {'id': 'combo_8',    'title': '연속 격추 III',   'tier': 'gold',     'desc': '200스텝 내 8마리 격추'},
        {'id': 'streak_7',   'title': '연사 콤보 III',   'tier': 'gold',     'desc': '간격 50스텝 이내 7연속 격추'},
        {'id': 'ufo_2',      'title': 'UFO 헌터',        'tier': 'gold',     'desc': '한 판에 UFO 2번 격추'},
        {'id': 'no_death_2', 'title': '철벽 방어선',     'tier': 'gold',     'desc': '목숨 잃지 않고 2라운드 클리어'},
        {'id': 'no_death_3', 'title': '전설의 방어선',   'tier': 'gold',     'desc': '목숨 잃지 않고 3라운드 클리어'},
        {'id': 'bench_human', 'title': '인간 평균 돌파', 'tier': 'gold',     'desc': '논문 기준 인간 평균(1,652점) 초과'},
        {'id': 'bench_dqn',  'title': 'DQN 돌파',        'tier': 'gold',     'desc': '2015 DQN 기준 점수(1,976점) 초과'},
        # platinum
        {'id': 'bench_ddqn', 'title': 'DDQN 돌파',       'tier': 'platinum', 'desc': 'DDQN 기준 점수(3,155점) 초과'},
    ]

    game_info = {
        'summary': '지구를 침공하는 외계인 함대를 격추하는 고전 슈팅 게임입니다.',
        'objective': '방어막을 활용해 외계인 공격을 피하면서 모든 적을 격추하세요.',
        'how_to_play': [
            '← → : 포대를 좌우로 이동',
            '스페이스바: 총알 발사',
            '방어막은 외계인 총알과 내 총알 모두 소모됩니다.',
            '외계인이 화면 아래로 내려오면 게임 오버입니다.',
        ],
        'scoring': [
            '1행(하단, 흰색): 5점',
            '2행: 10점',
            '3행: 15점',
            '4행: 20점',
            '5행: 25점',
            '6행(상단): 30점 — 위쪽일수록 고점수!',
            'UFO(화면 상단): 200점',
        ],
        'end_condition': '목숨 3개를 모두 소진하거나 외계인이 지면에 도달하면 종료됩니다.',
        'tips': [
            '외계인이 적을수록 이동 속도가 빨라지니 주의하세요.',
            '방어막은 소모되므로 너무 의존하지 마세요.',
        ],
        'controls': {
            'left':  '왼쪽 이동',
            'right': '오른쪽 이동',
            'fire':  '발사',
        },
    }

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.space_invaders import load_d3qn
        net, _ = load_d3qn(path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.space_invaders import get_q_values
        return get_q_values(self.net, stacked_state, self.device)

    # 6×6 적 배치 (줄당 6마리)
    # 연사 콤보 간격 허용치 (스텝): 총알 비행+재장전 고려
    _STREAK_GAP = 50

    def _on_episode_reset(self):
        self._rt = {
            'score':           0.0,
            'ufo_kills':       0,
            'rounds':          0,
            'cum_deaths':      0,
            'prev_lives':      None,
            'prev_ec':         None,
            'in_clear':        False,
            'kill_window':     [],   # step 인덱스 기록 (200스텝 윈도우)
            'kills_this_life': 0,    # 현재 목숨 격추 수 (줄 격파 판정)
            'streak':          0,    # 현재 연사 콤보 길이
            'max_streak':      0,    # 이번 판 최대 연사 콤보
            'last_kill_step':  -1,   # 마지막 격추 스텝
            'last_3_kill':     False,
        }

    def _check_realtime_achievements(self, entry: dict) -> list:
        if not hasattr(self, '_rt'):
            return []
        rt  = self._rt
        new = []

        def unlock(id_, title, desc, tier, condition):
            if condition and id_ not in self._ach_unlocked:
                self._ach_unlocked.add(id_)
                new.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        reward = entry.get('reward', 0)
        step   = entry.get('step', 0)
        lives  = entry.get('lives')
        ram    = entry.get('pre_ram')

        # 점수 누적
        rt['score'] += reward

        # 목숨 손실 — kills_this_life·streak 리셋
        if lives is not None and rt['prev_lives'] is not None and lives < rt['prev_lives']:
            rt['cum_deaths']     += rt['prev_lives'] - lives
            rt['kills_this_life'] = 0
            rt['streak']          = 0
            rt['last_kill_step']  = -1
        if lives is not None:
            rt['prev_lives'] = lives

        # UFO 격추
        if reward >= 100:
            rt['ufo_kills'] += 1

        # 일반 적 격추
        if 0 < reward < 100:
            rt['kill_window'].append(step)
            rt['kills_this_life'] += 1

            # 연사 콤보: 마지막 킬로부터 _STREAK_GAP 이내면 streak 유지
            gap = step - rt['last_kill_step'] if rt['last_kill_step'] >= 0 else 999
            rt['streak'] = rt['streak'] + 1 if gap <= self._STREAK_GAP else 1
            rt['max_streak'] = max(rt['max_streak'], rt['streak'])
            rt['last_kill_step'] = step

            # 남은 적 ≤3일 때 격추
            if ram is not None and int(ram[0x11]) <= 3:
                rt['last_3_kill'] = True

        # 200스텝 윈도우 유지
        rt['kill_window'] = [s for s in rt['kill_window'] if step - s <= 200]

        # 라운드 클리어
        if ram is not None:
            ec = int(ram[0x11])
            if rt['prev_ec'] is not None and rt['prev_ec'] > 0 and ec == 0 and not rt['in_clear']:
                rt['rounds'] += 1
                rt['in_clear'] = True
            elif ec > 0:
                rt['in_clear'] = False
            rt['prev_ec'] = ec

        s   = rt['score']
        r   = rt['rounds']
        d   = rt['cum_deaths']
        u   = rt['ufo_kills']
        kw  = rt['kill_window']
        ktl = rt['kills_this_life']
        sk  = rt['max_streak']
        # ── 초급 ──
        unlock('score_100',  '첫 발걸음',      '100점 달성',                          'bronze', s >= 100)
        unlock('score_300',  '자리 잡기',       '300점 달성',                          'bronze', s >= 300)
        unlock('combo_3',    '연속 격추 I',     '200스텝 내 3마리 격추',               'bronze', len(kw) >= 3)
        unlock('streak_3',   '연사 콤보 I',     f'간격 {self._STREAK_GAP}스텝 이내 3연속 격추', 'bronze', sk >= 3)
        unlock('round_1',    '1라운드 클리어',  '1라운드 적 전멸',                     'bronze', r >= 1)
        # ── 중급 ──
        unlock('score_500',  '격투사',          '500점 달성',                          'silver', s >= 500)
        unlock('score_1000', '에이스',          '1000점 달성',                         'silver', s >= 1000)
        unlock('combo_5',    '연속 격추 II',    '200스텝 내 5마리 격추',               'silver', len(kw) >= 5)
        unlock('streak_5',   '연사 콤보 II',    f'간격 {self._STREAK_GAP}스텝 이내 5연속 격추', 'silver', sk >= 5)
        unlock('ufo_1',      'UFO 격추',        '미스터리 쉽을 격추',                  'silver', u >= 1)
        unlock('final_3',    '핀포인트',        '적 3마리 이하 남았을 때 격추',        'silver', rt['last_3_kill'])
        unlock('no_death_1', '완벽한 수비',     '목숨 잃지 않고 1라운드 클리어',      'silver', r >= 1 and d == 0)
        unlock('round_2',    '2라운드 클리어',  '2라운드 적 전멸',                     'silver', r >= 2)
        # ── 고급 ──
        unlock('score_2000', '전설의 사수',     '2000점 달성',                         'gold', s >= 2000)
        unlock('score_3000', '우주 최강',       '3000점 달성',                         'gold', s >= 3000)
        unlock('combo_8',    '연속 격추 III',   '200스텝 내 8마리 격추',               'gold', len(kw) >= 8)
        unlock('streak_7',   '연사 콤보 III',   f'간격 {self._STREAK_GAP}스텝 이내 7연속 격추', 'gold', sk >= 7)
        unlock('ufo_2',      'UFO 헌터',        '한 판에 UFO 2번 격추',                'gold', u >= 2)
        unlock('no_death_2', '철벽 방어선',     '목숨 잃지 않고 2라운드 클리어',      'gold', r >= 2 and d == 0)
        unlock('no_death_3', '전설의 방어선',   '목숨 잃지 않고 3라운드 클리어',      'gold', r >= 3 and d == 0)
        unlock('bench_human','인간 평균 돌파',  '논문 기준 인간 평균(1,652점) 초과',   'gold', s > 1652)
        unlock('bench_dqn',  'DQN 돌파',        '2015 DQN 기준 점수(1,976점) 초과',    'gold', s > 1976)
        # ── 최고급 ──
        unlock('bench_ddqn', 'DDQN 돌파',       'DDQN 기준 점수(3,155점) 초과',        'platinum', s > 3155)

        return new

    def _extra_summary(self, entry: dict) -> dict:
        pre_ram = entry.get('pre_ram')
        pre_rgb = entry.get('pre_rgb')
        if pre_ram is None and pre_rgb is None:
            return {}
        return {'game_state': extract_si_game_state(pre_ram, pre_rgb)}

    def _compute_achievements(self) -> list:
        data = [d for d in self.episode_data if d.get('action') is not None]
        if not data:
            return []

        total_score = sum(d['reward'] for d in data)

        # ── 목숨 손실 누적 추적 ────────────────────────────────────────────
        lives_vals  = [d['lives'] for d in data if d.get('lives') is not None]
        life_losses = sum(
            max(0, lives_vals[i - 1] - lives_vals[i])
            for i in range(1, len(lives_vals))
        )

        # ── 라운드 클리어 감지 + 라운드 시점별 누적 목숨 손실 ───────────────
        rounds_cleared          = 0
        round_clear_life_losses = []   # 각 라운드 클리어 시점의 누적 life_losses
        prev_ec    = None
        in_clear   = False
        cum_deaths = 0
        prev_lives = None

        for d in data:
            lives = d.get('lives')
            if lives is not None and prev_lives is not None and lives < prev_lives:
                cum_deaths += prev_lives - lives
            prev_lives = lives if lives is not None else prev_lives

            ram = d.get('pre_ram')
            if ram is None:
                continue
            ec = int(ram[0x11])
            if prev_ec is not None and prev_ec > 0 and ec == 0 and not in_clear:
                rounds_cleared += 1
                round_clear_life_losses.append(cum_deaths)
                in_clear = True
            elif ec > 0:
                in_clear = False
            prev_ec = ec

        # ── UFO 격추 (reward >= 100 = UFO hit) ────────────────────────────
        ufo_kills = sum(1 for d in data if d.get('reward', 0) >= 100)

        # ── 연속 격추: 슬라이딩 윈도우 200스텝 ───────────────────────────
        kill_steps = [d['step'] for d in data if 0 < d.get('reward', 0) < 100]

        def max_kills_in_window(w: int) -> int:
            best, li = 0, 0
            for ri, s in enumerate(kill_steps):
                while kill_steps[li] < s - w:
                    li += 1
                best = max(best, ri - li + 1)
            return best

        max_combo = max_kills_in_window(200)

        # ── 연사 콤보: 킬 간격 _STREAK_GAP 이내 연속 ─────────────────────
        max_streak = 0
        streak = 0
        last_kill_s = -1
        prev_lives_sk = None
        for d in data:
            lives = d.get('lives')
            if lives is not None and prev_lives_sk is not None and lives < prev_lives_sk:
                streak, last_kill_s = 0, -1
            prev_lives_sk = lives if lives is not None else prev_lives_sk
            if 0 < d.get('reward', 0) < 100:
                gap = d['step'] - last_kill_s if last_kill_s >= 0 else 999
                streak = streak + 1 if gap <= self._STREAK_GAP else 1
                max_streak = max(max_streak, streak)
                last_kill_s = d['step']

        # ── 마지막 3마리 이하 상태에서 격추 여부 ──────────────────────────
        final_3 = any(
            0 < d.get('reward', 0) < 100
            and d.get('pre_ram') is not None
            and int(d['pre_ram'][0x11]) <= 3
            for d in data
        )

        # ── 도전과제 판정 ─────────────────────────────────────────────────
        achieved = []

        def add(id_, title, desc, tier, cond):
            if cond:
                achieved.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        # 초급
        add('score_100',  '첫 발걸음',      '100점 달성',                              'bronze', total_score >= 100)
        add('score_300',  '자리 잡기',       '300점 달성',                              'bronze', total_score >= 300)
        add('combo_3',    '연속 격추 I',     '200스텝 내 3마리 격추',                   'bronze', max_combo >= 3)
        add('streak_3',   '연사 콤보 I',     f'간격 {self._STREAK_GAP}스텝 이내 3연속', 'bronze', max_streak >= 3)
        add('round_1',    '1라운드 클리어',  '1라운드 적 전멸',                         'bronze', rounds_cleared >= 1)

        # 중급
        add('score_500',  '격투사',          '500점 달성',                              'silver', total_score >= 500)
        add('score_1000', '에이스',          '1000점 달성',                             'silver', total_score >= 1000)
        add('combo_5',    '연속 격추 II',    '200스텝 내 5마리 격추',                   'silver', max_combo >= 5)
        add('streak_5',   '연사 콤보 II',    f'간격 {self._STREAK_GAP}스텝 이내 5연속', 'silver', max_streak >= 5)
        add('ufo_1',      'UFO 격추',        '미스터리 쉽을 격추',                      'silver', ufo_kills >= 1)
        add('final_3',    '핀포인트',        '적 3마리 이하 남았을 때 격추',            'silver', final_3)
        add('no_death_1', '완벽한 수비',     '목숨 잃지 않고 1라운드 클리어',          'silver',
            rounds_cleared >= 1 and round_clear_life_losses[0] == 0)
        add('round_2',    '2라운드 클리어',  '2라운드 적 전멸',                         'silver', rounds_cleared >= 2)

        # 고급
        add('score_2000', '전설의 사수',    '2000점 달성',                              'gold', total_score >= 2000)
        add('score_3000', '우주 최강',      '3000점 달성',                              'gold', total_score >= 3000)
        add('combo_8',    '연속 격추 III',  '200스텝 내 8마리 격추',                    'gold', max_combo >= 8)
        add('streak_7',   '연사 콤보 III',  f'간격 {self._STREAK_GAP}스텝 이내 7연속',  'gold', max_streak >= 7)
        add('ufo_2',      'UFO 헌터',       '한 판에 UFO 2번 격추',                     'gold', ufo_kills >= 2)
        add('no_death_2', '철벽 방어선',    '목숨 잃지 않고 2라운드 클리어',           'gold',
            rounds_cleared >= 2 and round_clear_life_losses[1] == 0)
        add('no_death_3', '전설의 방어선',  '목숨 잃지 않고 3라운드 클리어',           'gold',
            rounds_cleared >= 3 and (len(round_clear_life_losses) > 2 and round_clear_life_losses[2] == 0))
        add('bench_human','인간 평균 돌파', '논문 기준 인간 평균(1,652점) 초과',        'gold', total_score > 1652)
        add('bench_dqn',  'DQN 돌파',       '2015 DQN 기준 점수(1,976점) 초과',         'gold', total_score > 1976)
        # 최고급
        add('bench_ddqn', 'DDQN 돌파',      'DDQN 기준 점수(3,155점) 초과',             'platinum', total_score > 6427)

        return achieved
