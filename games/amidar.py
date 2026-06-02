"""games/amidar.py — Amidar 게임 설정"""
import os
import numpy as np
from games.atari import AtariGame
from games.atari.ach_helper import episode_score, life_losses

# ── Amidar RGB 색상 팔레트 (ALE 렌더링 기준, 5가지뿐) ──────────────────────
_C_PAINTED   = np.array([104,  72, 198], dtype=np.uint8)  # 칠해진 경로 (보라)
_C_UNPAINTED = np.array([162,  98,  33], dtype=np.uint8)  # 미칠한 경로  (갈색)
_C_PLAYER    = np.array([135, 183,  84], dtype=np.uint8)  # 플레이어     (초록)
_C_ENEMY     = np.array([252, 252,  84], dtype=np.uint8)  # 적           (노랑)

_QUAD_KO = {
    'top_left':    '왼쪽 상단',
    'top_right':   '오른쪽 상단',
    'bottom_left': '왼쪽 하단',
    'bottom_right':'오른쪽 하단',
}


def _enemy_dir_ko(dx: int, dy: int) -> str:
    """픽셀 오프셋(dx, dy)을 한국어 방향 문자열로 변환."""
    adx, ady = abs(dx), abs(dy)
    if adx > ady * 1.8:
        return '오른쪽' if dx > 0 else '왼쪽'
    if ady > adx * 1.8:
        return '아래' if dy > 0 else '위'
    if dx > 0:
        return '오른쪽 아래' if dy > 0 else '오른쪽 위'
    return '왼쪽 아래' if dy > 0 else '왼쪽 위'


def extract_amidar_game_state(rgb_frame, ram=None) -> dict:
    """Amidar 게임 상태 추출.

    rgb_frame : 210×160×3 uint8 — 칠하기 진행률 계산에 사용 (항상 필요)
    ram       : 128바이트 ALE RAM — 플레이어/적 픽셀 좌표 정밀 추출에 사용 (선택)

    RAM이 없으면 RGB 픽셀 마스크로 위치를 대략 파악.
    """
    result: dict = {}

    # ── RGB: 칠하기 진행률 (RAM 대체 불가) ──────────────────────────────────
    if rgb_frame is not None:
        frame = np.asarray(rgb_frame, dtype=np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 3:
            game  = frame[15:195, :, :]   # 점수판(상단 15px) 제외
            h, w  = game.shape[:2]
            mh, mw = h // 2, w // 2

            pm = np.all(game == _C_PAINTED,   axis=2)
            um = np.all(game == _C_UNPAINTED, axis=2)

            _quads = {
                'top_left':    (pm[:mh, :mw],  um[:mh, :mw]),
                'top_right':   (pm[:mh, mw:],  um[:mh, mw:]),
                'bottom_left': (pm[mh:, :mw],  um[mh:, :mw]),
                'bottom_right':(pm[mh:, mw:],  um[mh:, mw:]),
            }
            qpct: dict[str, int] = {}
            for key, (p, u) in _quads.items():
                total = int(p.sum()) + int(u.sum())
                qpct[key] = round(int(p.sum()) / total * 100) if total > 0 else 0

            tp, tu  = int(pm.sum()), int(um.sum())
            overall = round(tp / (tp + tu) * 100) if (tp + tu) > 0 else 0
            least_k = min(qpct, key=qpct.get)

            result['painted_pct']        = overall
            result['quad_pct']           = {_QUAD_KO[k]: v for k, v in qpct.items()}
            result['least_painted_zone'] = _QUAD_KO[least_k]
            result['least_painted_pct']  = qpct[least_k]

            # RGB 폴백용 플레이어/적 픽셀 마스크 (RAM 있으면 덮어씀)
            _pl = np.all(game == _C_PLAYER, axis=2)
            _em = np.all(game == _C_ENEMY,  axis=2)
            _rgb_pl_mask = (_pl, mh, mw)
            _rgb_em_mask = (_em, mh, mw)
        else:
            _rgb_pl_mask = _rgb_em_mask = None
    else:
        _rgb_pl_mask = _rgb_em_mask = None

    # ── 플레이어 / 적 위치 ─────────────────────────────────────────────────
    # OC_Atari 확인 주소 (0-indexed):
    #   ram[59..65] = Y 좌표 (인덱스 0=플레이어, 1~6=적), +7 오프셋
    #   ram[66..72] = X 좌표 (인덱스 0=플레이어, 1~6=적), +9 오프셋
    GAME_H, GAME_W = 180, 160
    mid_h_g, mid_w_g = GAME_H // 2, GAME_W // 2

    if ram is not None and len(ram) >= 73:
        px = int(ram[66]) + 9
        py = int(ram[59]) + 7
        py_g = py - 15   # 게임 영역 기준 Y

        lr = 'left'   if px  < mid_w_g else 'right'
        tb = 'top'    if py_g < mid_h_g else 'bottom'
        result['player_zone'] = _QUAD_KO[f'{tb}_{lr}']

        enemies = []
        for i in range(1, 7):
            ex = int(ram[66 + i]) + 9
            ey = int(ram[59 + i]) + 7
            if ex <= 9 and ey <= 7:   # 위치 (0,0) → 비활성 슬롯
                continue
            dx, dy = ex - px, ey - py
            enemies.append({
                'dist': int((dx**2 + dy**2) ** 0.5),
                'dir':  _enemy_dir_ko(dx, dy),
            })
        enemies.sort(key=lambda e: e['dist'])
        result['enemy_count']          = len(enemies)
        result['enemies_desc']         = enemies
        result['nearest_enemy_dist']   = enemies[0]['dist'] if enemies else None
        result['nearest_enemy_dir']    = enemies[0]['dir']  if enemies else None
        result['player_enemy_same_zone'] = bool(enemies and enemies[0]['dist'] < 20)

    else:
        # ── RGB 폴백: 픽셀 마스크로 대략적인 위치 파악 ──────────────────
        if _rgb_pl_mask:
            pl_mask, mh, mw = _rgb_pl_mask
            py_idx, px_idx = np.where(pl_mask)
            if len(py_idx) > 0:
                cy, cx = int(py_idx.mean()), int(px_idx.mean())
                lr = 'left' if cx < mw else 'right'
                tb = 'top'  if cy < mh else 'bottom'
                result['player_zone'] = _QUAD_KO[f'{tb}_{lr}']

        if _rgb_em_mask:
            em_mask, mh, mw = _rgb_em_mask
            enemy_zones: list[str] = []
            try:
                from scipy.ndimage import label as ndlabel
                labeled, n_feat = ndlabel(em_mask)
                for i in range(1, min(n_feat + 1, 8)):
                    ey_i, ex_i = np.where(labeled == i)
                    if len(ey_i) < 3:
                        continue
                    cy, cx = int(ey_i.mean()), int(ex_i.mean())
                    lr = 'left' if cx < mw else 'right'
                    tb = 'top'  if cy < mh else 'bottom'
                    enemy_zones.append(_QUAD_KO[f'{tb}_{lr}'])
            except ImportError:
                if em_mask.sum() > 0:
                    ey_a, ex_a = np.where(em_mask)
                    lr = 'left' if int(ex_a.mean()) < mw else 'right'
                    tb = 'top'  if int(ey_a.mean()) < mh else 'bottom'
                    enemy_zones.append(_QUAD_KO[f'{tb}_{lr}'])

            seen: set[str] = set(); dedup: list[str] = []
            for z in enemy_zones:
                if z not in seen:
                    seen.add(z); dedup.append(z)

            result['enemy_count']          = len(dedup)
            result['enemies_desc']         = [{'dir': z, 'dist': None} for z in dedup]
            result['nearest_enemy_dist']   = None
            result['nearest_enemy_dir']    = dedup[0] if dedup else None
            pz = result.get('player_zone')
            result['player_enemy_same_zone'] = (pz in dedup) if pz else False

    return result

class AmidarGame(AtariGame):
    game_id    = 'amidar'
    game_title = 'AMIDAR'
    game_icon  = '🐒'
    env_name   = 'ALE/Amidar-v5'
    frame_skip  = 4
    prefix     = 'am_'
    theme_color = '#ff6b35'
    model_path_parts = ('ai_agents', 'amidar', 'checkpoints', 'best_model.pth')

    action_names = {
        0: 'NOOP',  1: 'FIRE',      2: 'UP',       3: 'RIGHT',
        4: 'LEFT',  5: 'DOWN',      6: 'UPFIRE',   7: 'RIGHTFIRE',
        8: 'LEFTFIRE', 9: 'DOWNFIRE',
    }

    keyboard_keys = [
        {'id': 'up',    'label': '↑',    'actions': [2, 6]},
        {'id': 'down',  'label': '↓',    'actions': [5, 9]},
        {'id': 'left',  'label': '←',    'actions': [4, 8]},
        {'id': 'right', 'label': '→',    'actions': [3, 7]},
        {'id': 'fire',  'label': 'SPACE', 'actions': [1, 6, 7, 8, 9]},
    ]

    key_combos = {
        'fire+up':    6,
        'fire+right': 7,
        'fire+left':  8,
        'down+fire':  9,
        'up':         2,
        'right':      3,
        'left':       4,
        'down':       5,
        'fire':       1,
        '':           0,
    }

    achievements = [
        # bronze
        {'id': 'box_first',       'title': '첫 박스',         'tier': 'bronze',   'desc': '격자 구역 1개 완성'},
        {'id': 'box_3',           'title': '구역 점령',       'tier': 'bronze',   'desc': '격자 구역 3개 완성'},
        # silver
        {'id': 'box_7',           'title': '박스 수집가',     'tier': 'silver',   'desc': '격자 구역 7개 완성'},
        {'id': 'box_10',          'title': '구역 지배',       'tier': 'silver',   'desc': '격자 구역 10개 완성'},
        {'id': 'no_death_2000',   'title': '무사 귀환',       'tier': 'silver',   'desc': '목숨 잃지 않고 2000스텝 생존'},
        {'id': 'score_650',       'title': '미로 달인',       'tier': 'silver',   'desc': '650점 달성 (DQN 근접)'},
        # gold
        {'id': 'box_15',          'title': '격자 정복자',     'tier': 'gold',     'desc': '격자 구역 15개 완성'},
        {'id': 'bench_dqn',       'title': 'DQN 돌파',        'tier': 'platinum',     'desc': 'DQN 기준(740점) 초과'},
        {'id': 'score_1000',      'title': '천점 돌파',       'tier': 'gold',     'desc': '1000점 달성'},
        {'id': 'no_death_5000',   'title': '불사신',          'tier': 'gold',     'desc': '목숨 잃지 않고 5000스텝 생존'},
        {'id': 'round2_no_death', 'title': '완벽한 1라운드',  'tier': 'gold',     'desc': '목숨 소모 없이 2라운드 진출'},
        {'id': 'score_1200',      'title': '격자 지배자',     'tier': 'gold',     'desc': '1200점 달성'},
        {'id': 'bench_human',     'title': '인간 평균 돌파',  'tier': 'gold',     'desc': '인간 평균(1,676점) 초과'},
        # platinum
        {'id': 'bench_ddqn',      'title': 'DDQN 돌파',       'tier': 'gold', 'desc': 'DDQN 기준(702점) 초과'},
    ]

    def _on_episode_reset(self):
        self._rt = {
            'score': 0.0, 'box_count': 0,
            'prev_lives': None,       # 직전 스텝 목숨 수
            'peak_painted_pct': 0,    # 이 게임 최고 painted_pct (라운드 감지용)
            'round2_entered': False,  # 2라운드 진출 여부
            'round_life_clean': True, # 현재까지 목숨 손실 없음
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

        reward  = entry.get('reward', 0)
        lives   = entry.get('lives')
        steps_total = len(self.episode_data)

        rt['score'] += reward
        if reward >= 40:
            rt['box_count'] += 1

        # ── 목숨 감소 추적 ─────────────────────────────────────────────
        prev_lives = rt['prev_lives']
        if lives is not None and prev_lives is not None and lives < prev_lives:
            rt['round_life_clean'] = False
        if lives is not None:
            rt['prev_lives'] = lives

        # ── 라운드 전환 감지 (painted_pct: 고점→저점) ──────────────────
        pre_rgb = entry.get('pre_rgb')
        if pre_rgb is not None and steps_total > 350:  # 인트로 이후만
            gs = extract_amidar_game_state(pre_rgb)
            pct = gs.get('painted_pct', 0)
            if pct > rt['peak_painted_pct']:
                rt['peak_painted_pct'] = pct
            # 이전 피크가 70% 이상이었다가 현재 10% 이하로 떨어짐 → 라운드 전환
            if rt['peak_painted_pct'] >= 70 and pct <= 10 and not rt['round2_entered']:
                rt['round2_entered'] = True

        s   = rt['score']
        bx  = rt['box_count']
        ll  = life_losses(self.episode_data)

        unlock('box_first',       '첫 박스',         '격자 구역 1개 완성',           'bronze',   bx >= 1)
        unlock('box_3',           '구역 점령',       '격자 구역 3개 완성',           'bronze',   bx >= 3)
        unlock('box_7',           '박스 수집가',     '격자 구역 7개 완성',           'silver',   bx >= 7)
        unlock('box_10',          '구역 지배',       '격자 구역 10개 완성',          'silver',   bx >= 10)
        unlock('no_death_2000',   '무사 귀환',       '목숨 잃지 않고 2000스텝 생존', 'silver',   steps_total >= 2000 and ll == 0)
        unlock('score_650',       '미로 달인',       '650점 달성 (DQN 근접)',         'silver',   s >= 650)
        unlock('box_15',          '격자 정복자',     '격자 구역 15개 완성',             'gold',     bx >= 15)
        unlock('bench_dqn',       'DQN 돌파',        'DQN 기준(740점) 초과',             'platinum',     s > 740)
        unlock('score_1000',      '천점 돌파',       '1000점 달성',                     'gold',     s >= 1000)
        unlock('round2_no_death', '완벽한 1라운드',  '목숨 소모 없이 2라운드 진출',     'gold',     rt['round2_entered'] and rt['round_life_clean'])
        unlock('score_1200',      '격자 지배자',     '1200점 달성',                     'gold',     s >= 1200)
        unlock('bench_human',     '인간 평균 돌파',  '인간 평균(1,676점) 초과',          'gold',     s > 1676)
        unlock('no_death_5000',   '불사신',          '목숨 잃지 않고 5000스텝 생존',    'gold',     steps_total >= 5000 and ll == 0)
        unlock('bench_ddqn',      'DDQN 돌파',       'DDQN 기준(702점) 초과',          'gold', s > 702)
        return new

    def _compute_achievements(self) -> list:
        data = [d for d in self.episode_data if d.get('action') is not None]
        if not data:
            return []
        s     = episode_score(data)
        ll    = life_losses(data)
        steps = len(data)
        box_count = sum(1 for d in data if d.get('reward', 0) >= 40)

        # ── 라운드 전환 감지 (painted_pct 고점→저점) ──────────────────
        round2_entered    = False
        round_life_clean  = True
        peak_pct          = 0
        prev_lives_val    = None
        for i, d in enumerate(data):
            lv = d.get('lives')
            if lv is not None and prev_lives_val is not None and lv < prev_lives_val:
                round_life_clean = False
            prev_lives_val = lv if lv is not None else prev_lives_val

            # painted_pct는 비용이 크므로 100스텝 간격으로 샘플링
            if i % 100 == 0 and d.get('step', 0) > 350:
                rgb = d.get('pre_rgb')
                if rgb is not None:
                    gs = extract_amidar_game_state(rgb)
                    pct = gs.get('painted_pct', 0)
                    if pct > peak_pct:
                        peak_pct = pct
                    if peak_pct >= 70 and pct <= 10 and not round2_entered:
                        round2_entered = True

        achieved = []

        def add(id_, title, desc, tier, cond):
            if cond:
                achieved.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        add('box_first',       '첫 박스',         '격자 구역 1개 완성',           'bronze',   box_count >= 1)
        add('box_3',           '구역 점령',       '격자 구역 3개 완성',           'bronze',   box_count >= 3)
        add('box_7',           '박스 수집가',     '격자 구역 7개 완성',           'silver',   box_count >= 7)
        add('box_10',          '구역 지배',       '격자 구역 10개 완성',          'silver',   box_count >= 10)
        add('no_death_2000',   '무사 귀환',       '목숨 잃지 않고 2000스텝 생존', 'silver',   ll == 0 and steps >= 2000)
        add('score_650',       '미로 달인',       '650점 달성 (DQN 근접)',         'silver',   s >= 650)
        add('box_15',          '격자 정복자',     '격자 구역 15개 완성',              'gold',     box_count >= 15)
        add('bench_dqn',       'DQN 돌파',        'DQN 기준(740점) 초과',              'platinum',     s > 740)
        add('score_1000',      '천점 돌파',       '1000점 달성',                      'gold',     s >= 1000)
        add('round2_no_death', '완벽한 1라운드',  '목숨 소모 없이 2라운드 진출',      'gold',     round2_entered and round_life_clean)
        add('score_1200',      '격자 지배자',     '1200점 달성',                      'gold',     s >= 1200)
        add('bench_human',     '인간 평균 돌파',  '인간 평균(1,676점) 초과',           'gold',     s > 1676)
        add('no_death_5000',   '불사신',          '목숨 잃지 않고 5000스텝 생존',     'gold',     ll == 0 and steps >= 5000)
        add('bench_ddqn',      'DDQN 돌파',       'DDQN 기준(702점) 초과',           'gold', s > 702)
        return achieved

    def _extra_summary(self, entry: dict) -> dict:
        pre_rgb = entry.get('pre_rgb')
        pre_ram = entry.get('pre_ram')
        if pre_rgb is None and pre_ram is None:
            return {}
        gs = extract_amidar_game_state(pre_rgb, pre_ram)
        return {'game_state': gs} if gs else {}

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.d3qn_helper import load_d3qn
        net, _ = load_d3qn('amidar', path, self.device)
        return net

    game_info = {
        'summary': '격자판 위를 이동하며 모든 경로를 칠해 완성하는 전략 게임입니다.',
        'objective': '적에게 잡히지 않고 격자판의 모든 선을 칠하세요.',
        'how_to_play': [
            '방향키(↑↓←→)로 이동하며 지나간 경로를 자동으로 칠합니다.',
            '격자의 각 구역(박스) 네 변을 모두 칠하면 박스가 완성되고 보너스 점수를 얻습니다.',
            '스페이스바: 회피 기동 — 적을 잠깐 통과할 수 있게 됩니다. 목숨당 4회만 사용 가능하며 같은 목숨 안에서 충전되지 않습니다.',
            '홀수 라운드(고릴라) ↔ 짝수 라운드(페인트 롤러)가 번갈아 진행되며, 라운드마다 적의 종류와 수가 달라집니다.',
        ],
        'scoring': [
            '가로선 새로 칠하기: 1~6점 (선 위치에 따라 다름)',
            '세로선 새로 칠하기: 1점',
            '격자 구역(박스) 1개 완성: 약 48점 보너스 (원작 규칙 50점)',
            '모든 구역은 동일한 크기이므로 완성 보너스도 동일합니다.',
            '이미 칠한 선을 다시 지나가도 추가 점수는 없습니다.',
        ],
        'end_condition': '목숨 3개를 모두 소진하면 게임이 종료됩니다. 라운드 클리어 시 남은 목숨이 2개 이상이면 다음 라운드도 3개로 시작하고, 1개면 2개로 시작합니다.',
        'controls': {
            'up':    '이동',
            'down':  '이동',
            'left':  '이동',
            'right': '이동',
            'fire':  '회피 기동',
        },
        'tips': [
            '구역 완성 순서를 미리 계획해 효율적으로 움직이세요.',
            '회피 기동은 목숨당 4회뿐입니다. 막다른 상황에서만 아껴 쓰세요.',
            '라운드가 올라갈수록 적 수와 속도가 증가하므로 초반 라운드에서 최대한 박스를 채우세요.',
        ],
    }

    def _get_q_values(self, stacked_state):
        from ai_agents.d3qn_helper import get_q_values
        return get_q_values(self.net, stacked_state, self.device)
