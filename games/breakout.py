"""games/breakout.py — Breakout 게임 설정"""
import numpy as np
from games.atari import AtariGame
from games.atari.ach_helper import episode_score, streak_max, life_losses


# ── 벽돌 색상 (아래→위 순서, row index 0=파랑 ... 5=빨강) ──────────────────
_BRICK_ROW_COLORS = ['파랑', '초록', '노랑', '연한 주황', '진한 주황', '빨강']
_BRICK_ROW_PTS    = [1,      1,      1,      4,           4,           7]


def _brick_bitmap(ram: np.ndarray) -> np.ndarray:
    """
    OC_Atari 방식으로 RAM 0~35에서 6×18 벽돌 비트맵 생성.
    반환: shape (6, 18), 1=벽돌 있음, 0=제거됨. row0=파랑(하단), row5=빨강(상단).
    """
    array = ram[:36].reshape(-1, 6)
    rows = []
    for col in np.array(array).T:
        row_str = ''
        for j, val in enumerate(col):
            if j == 0:
                row_str = '{0:06b}'.format(int(val))[::-2] + row_str
            elif j == 5:
                row_str = '{0:08b}'.format(int(val))[1::-2] + row_str
            else:
                row_str = '{0:08b}'.format(int(val))[::-2] + row_str
        rows.append(row_str)
    # rows[0] = 맨 위(빨강), rows[-1] = 맨 아래(파랑)
    rows_int = np.array([[int(c) for c in r] for r in rows], dtype=np.int8)
    # correct column order (OC_Atari 기준)
    order = [0, 4, 3, 2, 1, 5, 6, 7, 8, 11, 12, 16, 15, 14, 13, 17, 18, 19]
    rows_int = rows_int.T[order].T
    # 행 순서를 아래→위로 뒤집어 row0=파랑(하단)으로
    return rows_int[::-1]


def extract_breakout_game_state(ram: np.ndarray, prev_ball_y: int | None = None) -> dict:
    """
    RAM 기반 Breakout 게임 상태 추출.
      ram[72]  = paddle x (raw)
      ram[99]  = ball x (raw)
      ram[101] = ball y (raw, 0 = 공 없음)
    """
    if ram is None:
        return {}
    try:
        ball_raw_y = int(ram[101])
        ball_in_play = (ball_raw_y != 0 and ball_raw_y + 9 <= 196)

        # 화면 좌표 (OC_Atari 오프셋)
        ball_x  = int(ram[99])  - 49
        ball_y  = ball_raw_y    + 9  # 화면 상의 y
        paddle_x = int(ram[72]) - 47

        # 공 방향 (y 증가 = 아래로, y 감소 = 위로)
        if not ball_in_play:
            ball_dir = '새 공 대기 — 패들 위에 공이 놓인 상태, SPACE로 출발'
        elif prev_ball_y is not None and ball_raw_y != 0:
            delta = ball_raw_y - prev_ball_y
            if delta < -1:
                ball_dir = '상승중(벽돌 쪽)'
            elif delta > 1:
                ball_dir = '하강중(패들 쪽)'
            else:
                ball_dir = '수평이동'
        else:
            ball_dir = '불명'

        # 패들 중심 기준 공 상대 위치
        if ball_in_play:
            rel = ball_x - paddle_x
            if rel < -20:
                ball_vs_paddle = '패들 왼쪽'
            elif rel > 20:
                ball_vs_paddle = '패들 오른쪽'
            else:
                ball_vs_paddle = '패들 중앙 근처'
        else:
            ball_vs_paddle = '—'

        # 벽돌 비트맵 분석
        bmap = _brick_bitmap(ram)  # shape (6, 18), row0=파랑, row5=빨강
        total_bricks = int(bmap.sum())
        bricks_per_row = bmap.sum(axis=1).tolist()  # [파랑수, ..., 빨강수]

        # 터널 분석: 각 열에서 하단(파랑 row=0)부터 연속으로 비어 있는 행 수
        max_cleared_depth = 0
        tunnel_col_count   = 0
        for col in range(bmap.shape[1]):
            depth = 0
            for row in range(6):  # 0=파랑(아래)→5=빨강(위)
                if bmap[row, col] == 0:
                    depth += 1
                else:
                    break
            if depth >= 2:
                tunnel_col_count += 1
            max_cleared_depth = max(max_cleared_depth, depth)

        tunnel_forming = max_cleared_depth >= 2  # ≥2행 연속 개통된 열 존재
        in_tunnel = ball_in_play and ball_raw_y < 57  # 공이 벽돌 위(천장 근처)

        # 공이 상승 중일 때 가장 먼저 닿을 벽돌 행 (하단부터 첫 비어있지 않은 행)
        nearest_row_name = None
        nearest_row_pts  = None
        for ri in range(6):  # 0=파랑(최하단) → 5=빨강(최상단)
            if int(bricks_per_row[ri]) > 0:
                nearest_row_name = _BRICK_ROW_COLORS[ri]
                nearest_row_pts  = _BRICK_ROW_PTS[ri]
                break

        return {
            'ball_x':             ball_x,
            'ball_y':             ball_y,
            'ball_dir':           ball_dir,
            'ball_vs_paddle':     ball_vs_paddle,
            'ball_in_play':       ball_in_play,
            'paddle_x':           paddle_x,
            'total_bricks':       total_bricks,
            'bricks_per_row':     [int(v) for v in bricks_per_row],
            'nearest_row_name':   nearest_row_name,
            'nearest_row_pts':    nearest_row_pts,
            'tunnel_forming':     tunnel_forming,
            'tunnel_col_count':   tunnel_col_count,
            'max_cleared_depth':  max_cleared_depth,
            'in_tunnel':          in_tunnel,
        }
    except Exception:
        return {}


class BreakoutGame(AtariGame):
    game_id    = 'breakout'
    game_title = 'BREAKOUT'
    game_icon  = '🎯'
    env_name   = 'ALE/Breakout-v5'
    frame_skip  = 3
    prefix     = 'bo_'
    theme_color = '#ff8a1c'
    model_path_parts = ('data', 'checkpoints', 'breakout', 'best_model.pth')

    action_names = {
        0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT',
    }

    keyboard_keys = [
        {'id': 'left',  'label': '←',    'actions': [3]},
        {'id': 'right', 'label': '→',    'actions': [2]},
        {'id': 'fire',  'label': 'SPACE', 'actions': [1]},
    ]

    key_combos = {
        'left':  3,
        'right': 2,
        'fire':  1,
        '':      0,
    }

    achievements = [
        # bronze
        {'id': 'score_5',    'title': '첫 타격',         'tier': 'bronze',   'desc': '5점 달성'},
        {'id': 'score_15',   'title': '벽돌 깨기',       'tier': 'bronze',   'desc': '15점 달성'},
        {'id': 'combo_3',    'title': '연속 파괴 I',     'tier': 'bronze',   'desc': '벽돌 3개 연속 파괴 (간격 150프레임 이내)'},
        # silver
        {'id': 'bench_human','title': '인간 평균 돌파',  'tier': 'silver',   'desc': '인간 평균(31점) 초과'},
        {'id': 'score_50',   'title': '브레이커',         'tier': 'silver',   'desc': '50점 달성'},
        {'id': 'combo_6',    'title': '연속 파괴 II',    'tier': 'silver',   'desc': '벽돌 6개 연속 파괴 (간격 150프레임 이내)'},
        {'id': 'survive_20', 'title': '여유로운 플레이', 'tier': 'silver',   'desc': '목숨 3개 이상 유지하며 20점 달성'},
        {'id': 'survive_50', 'title': '안정적인 클리어', 'tier': 'silver',  'desc': '목숨 3개 이상 유지하며 50점 달성'},
        # gold
        {'id': 'score_100',  'title': '마스터 브레이커', 'tier': 'gold',     'desc': '100점 달성'},
        {'id': 'combo_12',   'title': '연속 파괴 III',   'tier': 'gold',     'desc': '벽돌 12개 연속 파괴 (간격 150프레임 이내)'},
        {'id': 'bench_dqn',  'title': 'DQN 돌파',        'tier': 'platinum',     'desc': 'DQN 기준(401점) 초과'},
        # platinum
        {'id': 'bench_ddqn', 'title': 'DDQN 돌파',       'tier': 'gold', 'desc': 'DDQN 기준(375점) 초과'},
    ]

    def _on_episode_reset(self):
        self._rt = {
            'score': 0.0, 'streak': 0,
            'max_streak': 0, 'last_kill_step': -1,
            'survive_20_done': False, 'survive_50_done': False,
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
            rt['streak'] = rt['streak'] + 1 if gap <= 150 else 1
            rt['max_streak'] = max(rt['max_streak'], rt['streak'])
            rt['last_kill_step'] = step
        s    = rt['score']
        sk   = rt['max_streak']
        cur_lives = lives if lives is not None else getattr(self, '_rt_prev_lives', 0)

        # 목숨 3개 이상 유지하며 점수 달성 여부 추적
        if cur_lives is not None and cur_lives >= 3:
            if s >= 20 and not rt['survive_20_done']:
                rt['survive_20_done'] = True
            if s >= 50 and not rt['survive_50_done']:
                rt['survive_50_done'] = True

        unlock('score_5',    '첫 타격',         '5점 달성',                    'bronze',   s >= 5)
        unlock('score_15',   '벽돌 깨기',       '15점 달성',                   'bronze',   s >= 15)
        unlock('combo_3',    '연속 파괴 I',     '벽돌 3개 연속 파괴 (간격 150프레임 이내)', 'bronze', sk >= 3)
        unlock('bench_human','인간 평균 돌파',  '인간 평균(31점) 초과',         'silver',   s > 31)
        unlock('score_50',   '브레이커',        '50점 달성',                    'silver',   s >= 50)
        unlock('combo_6',    '연속 파괴 II',    '벽돌 6개 연속 파괴 (간격 150프레임 이내)', 'silver', sk >= 6)
        unlock('survive_20', '여유로운 플레이', '목숨 3개 이상 유지하며 20점 달성', 'silver', rt['survive_20_done'])
        unlock('survive_50', '안정적인 클리어', '목숨 3개 이상 유지하며 50점 달성', 'silver', rt['survive_50_done'])
        unlock('score_100',  '마스터 브레이커', '100점 달성',                  'gold',     s >= 100)
        unlock('combo_12',   '연속 파괴 III',   '벽돌 12개 연속 파괴 (간격 150프레임 이내)', 'gold', sk >= 12)
        unlock('bench_dqn',  'DQN 돌파',        'DQN 기준(401점) 초과',        'platinum', s > 401)
        unlock('bench_ddqn', 'DDQN 돌파',       'DDQN 기준(375점) 초과',       'gold',     s > 375)
        return new

    def _compute_achievements(self) -> list:
        data = [d for d in self.episode_data if d.get('action') is not None]
        if not data:
            return []
        s  = episode_score(data)
        sk = streak_max(data, gap=150, min_r=0, max_r=1e9)

        # 목숨 3개 이상 유지하며 점수 달성 여부
        survive_20 = survive_50 = False
        cumulative = 0.0
        for d in data:
            cumulative += d.get('reward', 0)
            cur_lives = d.get('lives')
            if cur_lives is not None and cur_lives >= 3:
                if cumulative >= 20:
                    survive_20 = True
                if cumulative >= 50:
                    survive_50 = True

        achieved = []

        def add(id_, title, desc, tier, cond):
            if cond:
                achieved.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        add('score_5',    '첫 타격',         '5점 달성',                    'bronze',   s >= 5)
        add('score_15',   '벽돌 깨기',       '15점 달성',                   'bronze',   s >= 15)
        add('combo_3',    '연속 파괴 I',     '벽돌 3개 연속 파괴 (간격 150프레임 이내)', 'bronze', sk >= 3)
        add('bench_human','인간 평균 돌파',  '인간 평균(31점) 초과',         'silver',   s > 31)
        add('score_50',   '브레이커',        '50점 달성',                    'silver',   s >= 50)
        add('combo_6',    '연속 파괴 II',    '벽돌 6개 연속 파괴 (간격 150프레임 이내)', 'silver', sk >= 6)
        add('survive_20', '여유로운 플레이', '목숨 3개 이상 유지하며 20점 달성', 'silver', survive_20)
        add('survive_50', '안정적인 클리어', '목숨 3개 이상 유지하며 50점 달성', 'silver', survive_50)
        add('score_100',  '마스터 브레이커', '100점 달성',                  'gold',     s >= 100)
        add('combo_12',   '연속 파괴 III',   '벽돌 12개 연속 파괴 (간격 150프레임 이내)', 'gold', sk >= 12)
        add('bench_dqn',  'DQN 돌파',        'DQN 기준(401점) 초과',        'platinum', s > 401)
        add('bench_ddqn', 'DDQN 돌파',       'DDQN 기준(375점) 초과',       'gold',     s > 375)
        return achieved

    def _select_top5(self, analyses: list, k: int = 5, min_gap: int = 60) -> list:
        """전체 loss 상위 k개, 최소 min_gap 스텝 간격 보장 (구간 분할 없음)."""
        if not analyses:
            return []

        # 에피소드 종료 시점 찾기
        max_step = max(a['step'] for a in analyses) if analyses else 0
        # 마지막 30 프레임은 제외 (이미 공이 떨어지고 있어 의미 없음)
        cutoff_step = max_step - 30

        # 종료 직전 프레임 필터링
        valid_analyses = [a for a in analyses if a['step'] <= cutoff_step]

        if not valid_analyses:
            # 모든 후보가 종료 직전이면 그냥 원본 사용
            valid_analyses = analyses

        sorted_all = sorted(valid_analyses, key=lambda a: a['loss'], reverse=True)
        selected = []
        for c in sorted_all:
            if len(selected) >= k:
                break
            if all(abs(c['step'] - s['step']) >= min_gap for s in selected):
                selected.append(c)
        return sorted(selected, key=lambda a: a['loss'], reverse=True)

    def _extra_summary(self, entry: dict) -> dict:
        pre_ram = entry.get('pre_ram')
        if pre_ram is None:
            return {}

        # 이전 프레임 ball_y를 찾아 공 방향 계산
        step = entry.get('step', 0)
        prev_ball_y: int | None = None
        for prev in reversed(self.episode_data[max(0, step - 5):step]):
            pr = prev.get('pre_ram')
            if pr is not None and int(pr[101]) != 0:
                prev_ball_y = int(pr[101])
                break

        gs = extract_breakout_game_state(pre_ram, prev_ball_y)
        return {'game_state': gs} if gs else {}

    # _load_model / _get_q_values 는 베이스 기본 구현(d3qn_helper) 사용

    game_info = {
        'summary': '화면 아래 패들을 좌우로 움직여 공을 튕기고, 위쪽에 배열된 벽돌을 모두 부수는 벽돌깨기 게임입니다.',
        'objective': '공을 놓치지 않으면서 벽돌을 최대한 많이 깨 고득점을 노리세요. 점수는 오직 벽돌을 깼을 때만 발생합니다.',
        'how_to_play': [
            '← → : 패들을 좌우로 이동합니다.',
            '스페이스바 : 공 발사(serve) — 게임 시작 또는 공을 잃은 직후에만 유효합니다. 공이 이미 움직이는 중에는 효과가 없습니다.',
            '공이 패들 아래로 떨어지면 공 하나 감소 (총 5개). 5개 모두 소진 시 게임 오버.',
            '현재 화면의 벽돌을 모두 제거하면 다음 라운드(새 벽돌 배열)로 진행됩니다.',
        ],
        'scoring': [
            '파랑·초록·노랑 (아래 3줄): 각 1점',
            '연한 주황·진한 주황 (중간 2줄): 각 4점',
            '빨강 (맨 위 1줄): 7점 — 위쪽 벽돌일수록 점수가 크게 뜁니다.',
            '점수는 벽돌 파괴 시에만 발생합니다. 패들 랠리만으로는 점수가 오르지 않습니다.',
        ],
        'end_condition': '공 5개를 모두 잃으면 게임 오버. 벽돌을 전부 제거하면 다음 라운드로 이어지며 남은 공은 그대로 유지됩니다.',
        'tips': [
            '공이 내려올 때 이동하면 늦습니다. 공의 궤도를 미리 읽고 착지 지점으로 먼저 패들을 옮기세요.',
            '패들 어느 부분에 공이 닿느냐로 반사 방향이 바뀝니다. 중앙은 수직에 가깝고, 끝으로 갈수록 예리한 각도가 나옵니다.',
            '한쪽 열을 집중 공략해 통로(터널)를 뚫으면 공이 벽돌 위쪽으로 넘어가 천장과 벽 사이에서 연속 파괴가 일어납니다.',
            '벽돌이 많이 제거될수록 공 속도가 점점 빨라집니다. 후반에는 패들을 중앙에 두어 반응 거리를 확보하세요.',
        ],
        # 코칭 전용 (화면에 표시되지 않음)
        'ball_physics': (
            '패들 중앙 → 수직 반사 / 왼쪽 끝 → 왼쪽 위 / 오른쪽 끝 → 오른쪽 위. '
            '가장자리로 받을수록 예리한 각도를 만들 수 있습니다. '
            '벽돌을 많이 부술수록 공 속도가 빨라집니다.'
        ),
        'controls': {
            'left':  '패들 왼쪽 이동',
            'right': '패들 오른쪽 이동',
            'fire':  '새 공 발사',
        },
    }

