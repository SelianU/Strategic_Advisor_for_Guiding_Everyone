"""games/enduro.py — Enduro 게임 설정"""
import os
import colorsys
import numpy as np
from games.atari import AtariGame
from games.atari.ach_helper import episode_score

# ── 스텝 기반 시간대 테이블 (JS STEP_PHASES와 동일, 캘리브레이션 실측값) ──
_CYCLE_STEPS = 13312
# 캘리브레이션 실측: 도로 밝기(road_bright)로 단계 구분
#   night1 (7169-9299): sky≈64, road≈10 → 도로 암흑, 후미등만 보임
#   fog    (9300-11200): sky≈74, road≈72 → 하늘·도로 모두 균일 회색
#   night2 (11201-12299): sky≈64, road≈10 → 도로 다시 암흑
#   dawn   (12300-13311): sky≈92, road≈10 → 하늘만 밝아짐
_STEP_PHASES = [
    (0,     3071,  'day',    'normal',      '낮'),
    (3072,  5119,  'day',    'snow_or_ice', '낮 (눈길)'),
    (5120,  6399,  'day',    'normal',      '낮'),
    (6400,  7168,  'sunset', 'normal',      '석양'),
    (7169,  9299,  'night',  'normal',      '야간'),
    (9300,  11200, 'fog',    'fog',         '안개'),
    (11201, 12299, 'night',  'normal',      '야간'),
    (12300, 13311, 'dawn',   'fog',         '새벽'),
]

def phase_from_step(step: int) -> dict:
    """스텝 기반 시간대·도로 상태 반환 (JS STEP_PHASES와 동기화)."""
    s = step % _CYCLE_STEPS
    for from_s, to_s, phase, road, label_ko in _STEP_PHASES:
        if from_s <= s <= to_s:
            return {'display_phase': phase, 'road_condition': road, 'phase_ko': label_ko}
    return {'display_phase': 'day', 'road_condition': 'normal', 'phase_ko': '낮'}


def _rgb_stats(region: np.ndarray) -> dict:
    """RGB 영역의 평균 색상 통계 반환 (0-255 스케일, saturation은 0-1)."""
    mean_rgb = region.reshape(-1, 3).mean(axis=0)       # 0-255
    r_n, g_n, b_n = mean_rgb / 255.0                    # HSV 계산용 정규화
    _, s, _ = colorsys.rgb_to_hsv(float(r_n), float(g_n), float(b_n))
    return {
        'r':          float(mean_rgb[0]),
        'g':          float(mean_rgb[1]),
        'b':          float(mean_rgb[2]),
        'brightness': float(mean_rgb.mean()),
        'saturation': float(s),
    }


def detect_enduro_phase(rgb_frame: np.ndarray) -> dict:
    """
    Enduro ALE RGB 프레임(210×160×3)에서 시간대와 도로 상태를 감지합니다.

    영역 설정 (210 height 기준):
      sky     : rows  17~59  (h×0.08 ~ h×0.28) — 하늘 색으로 시간대 판단
      horizon : rows  46~71  (h×0.22 ~ h×0.34) — 산악 지평선대, 새벽 주황빛 감지
      road    : rows 101~164 (h×0.48 ~ h×0.78) — 도로 밝기로 눈길 판단

    탈채색(sky_sat < 0.15) 구간 3-way 분류:
      dawn    : 지평선대가 따뜻한 주황빛 (hr > hb+28, hr > 80) → 새벽
      night   : 지평선 차갑고 하늘 매우 어두움 (sky_bright < 35) → 야간
      fog     : 나머지 탈채색 → 안개 (회색)

    반환:
      time_phase    : 'day'|'sunset_early'|'sunset_mid'|'sunset_late'|'night'|'fog'|'dawn'
      display_phase : 'day'|'sunset'|'night'|'fog'|'dawn'  (프론트 표시용)
      road_condition: 'normal'|'snow_or_ice'|'fog'
      confidence    : float 0-1
      sky_rgb       : [r, g, b]  (0-255)
      road_rgb      : [r, g, b]  (0-255)
    """
    if rgb_frame is None:
        return {}
    try:
        h, w, _ = rgb_frame.shape
        sky     = rgb_frame[int(h * 0.08):int(h * 0.28), int(w * 0.12):int(w * 0.88)]
        horizon = rgb_frame[int(h * 0.22):int(h * 0.34), int(w * 0.08):int(w * 0.92)]
        road    = rgb_frame[int(h * 0.48):int(h * 0.78), int(w * 0.30):int(w * 0.70)]

        ss = _rgb_stats(sky)
        hs = _rgb_stats(horizon)
        rs = _rgb_stats(road)

        sr, sg, sb   = ss['r'], ss['g'], ss['b']
        sky_bright   = ss['brightness']
        sky_sat      = ss['saturation']
        road_bright  = rs['brightness']
        road_sat     = rs['saturation']
        hr, hg, hb   = hs['r'], hs['g'], hs['b']

        # 지평선 주황빛 판단: 산악대가 따뜻한 색조면 새벽
        # 지평선 영역에 회색 하늘이 섞이므로 평균 R이 높지 않아 임계값을 낮게 설정
        horizon_warm = (hr > hb + 15 and hr > 55)

        # ── 시간대 감지 ────────────────────────────────────────────────────
        # 1순위: 탈채색(채도 < 0.15) → fog 계열을 3단계로 세분화
        #   - 지평선이 주황색  → 새벽(dawn)   : hr > hb+15 AND hr > 55
        #   - 하늘이 어두움    → 야간(night)  : sky_bright < 50
        #   - 나머지           → 안개(fog, 회색)
        # 2순위: 채도 있는 어두운 하늘 → 야간 (Day 2+ 의 짧은 true-night)
        # 3순위: 파란 하늘 → 낮
        # 4순위: 붉은/주황 하늘 → 석양
        if sky_sat < 0.15:
            if horizon_warm:
                time_phase, display_phase, confidence = 'dawn',  'dawn',  0.84
            elif sky_bright < 50:
                time_phase, display_phase, confidence = 'night', 'night', 0.86
            else:
                time_phase, display_phase, confidence = 'fog',   'fog',   0.85
        elif sky_bright < 30:
            # 채도 있는 pitch-dark → true night (석양 후 짧은 구간)
            time_phase, display_phase, confidence = 'night',        'night',  0.90
        elif sb > sr + 25 and sb > sg + 10 and sky_bright >= 55:
            time_phase, display_phase, confidence = 'day',          'day',    0.85
        elif sr >= sb and sky_bright >= 80:
            time_phase, display_phase, confidence = 'sunset_early', 'sunset', 0.75
        elif sr >= sb and 45 <= sky_bright < 80:
            time_phase, display_phase, confidence = 'sunset_mid',   'sunset', 0.75
        elif sr >= sb and sky_bright < 45:
            time_phase, display_phase, confidence = 'sunset_late',  'sunset', 0.70
        else:
            if sky_bright >= 80:
                time_phase, display_phase = 'day',        'day'
            elif sky_bright >= 45:
                time_phase, display_phase = 'sunset_mid', 'sunset'
            else:
                time_phase, display_phase = 'night',      'night'
            confidence = 0.45

        # ── 도로 상태 감지 ────────────────────────────────────────────────
        if road_bright > 110 and road_sat < 0.20:
            road_condition = 'snow_or_ice'
        elif display_phase in ('fog', 'dawn'):
            road_condition = 'fog'
        else:
            road_condition = 'normal'

        return {
            'time_phase':     time_phase,
            'display_phase':  display_phase,
            'road_condition': road_condition,
            'confidence':     round(confidence, 2),
            'sky_rgb':        [round(sr, 1), round(sg, 1), round(sb, 1)],
            'road_rgb':       [round(rs['r'], 1), round(rs['g'], 1), round(rs['b'], 1)],
            'sky_brightness': round(sky_bright, 1),
            'sky_saturation': round(sky_sat, 3),
            'horizon_rgb':    [round(hr, 1), round(hg, 1), round(hb, 1)],
        }
    except Exception:
        return {}


def extract_enduro_game_state(ram: np.ndarray | None) -> dict:
    """
    Enduro RAM 기반 게임 상태 피처 추출.

    RAM 레이아웃 (OC_Atari 기준):
      ram[27:34] — 차량 슬롯 7개 (0이 아니면 해당 슬롯에 차 존재)
      ram[45]    — Day / Level (1 = Day 1, 2 = Day 2, ...)
      ram[46]    — 플레이어 X 드리프트 (부호 있는 값, bit7 = 음수)
      ram[52]    — 플레이어 Y 오프셋 (클수록 화면 아래쪽 = 뒤로 밀린 상태)
      ram[59]    — 스크롤 속도 원시값 (상위 비트 = 현재 게임 속도 프록시)
    """
    if ram is None:
        return {}

    try:
        # ── 플레이어 X 드리프트 (ram[46] = signed byte) ───────────────────────
        # 양수 = 왼쪽으로 치우침, 음수 = 오른쪽으로 치우침 (range: -22 ~ +22)
        drift_raw = int(ram[46])
        signed_drift = drift_raw if drift_raw < 128 else drift_raw - 256
        # px: 도로 중앙≈80, 최대 왼쪽≈58, 최대 오른쪽≈102
        px = 80 - signed_drift

        # ── 도로 내 좌우 위치 ─────────────────────────────────────────────────
        if px < 65:
            lateral = '왼쪽 가장자리'
        elif px < 76:
            lateral = '왼쪽'
        elif px < 85:
            lateral = '중앙'
        elif px < 95:
            lateral = '오른쪽'
        else:
            lateral = '오른쪽 가장자리'

        # ── 화면 내 활성 차량 수 ──────────────────────────────────────────────
        cars_on_screen = sum(1 for i in range(7) if int(ram[27 + i]) != 0)

        # ── 게임 속도 프록시 (스크롤 속도 상위 비트) ────────────────────────────
        speed_raw = int(ram[59]) >> 3

        # ── Day / Level ──────────────────────────────────────────────────────
        day = int(ram[45])   # ROM에서 이미 1-indexed (Day 1 = 1, Day 2 = 2, ...)

        return {
            'player_x':       px,
            'lateral':        lateral,
            'cars_on_screen': cars_on_screen,
            'speed_raw':      speed_raw,
            'day':            day,
        }
    except Exception:
        return {}


class EnduroGame(AtariGame):
    game_id    = 'enduro'
    game_title = 'ENDURO'
    game_icon  = '🏎️'
    env_name   = 'ALE/Enduro-v5'
    frame_skip  = 4
    prefix     = 'en_'
    theme_color = '#ff2255'
    model_path_parts = ('ai_agents', 'enduro', 'checkpoints', 'best_model.pth')

    action_names = {
        0: 'NOOP',
        1: 'FIRE',
        2: 'RIGHT',
        3: 'LEFT',
        4: 'DOWN',
        5: 'DOWNRIGHT',
        6: 'DOWNLEFT',
        7: 'RIGHTFIRE',
        8: 'LEFTFIRE',
    }

    keyboard_keys = [
        {'id': 'left',      'label': '←',    'actions': [3, 6, 8]},   # LEFT, DOWNLEFT, LEFTFIRE
        {'id': 'right',     'label': '→',    'actions': [2, 5, 7]},   # RIGHT, DOWNRIGHT, RIGHTFIRE
        {'id': 'down',      'label': '↓',    'actions': [4, 5, 6]},   # DOWN, DOWNRIGHT, DOWNLEFT
        {'id': 'fire',      'label': 'GAS',  'actions': [1, 7, 8]},   # FIRE, RIGHTFIRE, LEFTFIRE
    ]

    key_combos = {
        # 3키 조합
        'down+fire+left':  6,   # DOWNLEFT  (fire 무시, down+left 우선)
        'down+fire+right': 5,   # DOWNRIGHT
        # 2키 조합 (알파벳순)
        'down+left':       6,   # DOWNLEFT
        'down+right':      5,   # DOWNRIGHT
        'fire+left':       8,   # LEFTFIRE
        'fire+right':      7,   # RIGHTFIRE
        # 단일 키
        'down':            4,
        'fire':            1,
        'left':            3,
        'right':           2,
        # 아무것도 안 누름
        '':                0,
    }

    achievements = [
        # bronze
        {'id': 'score_5',     'title': '첫 추월',          'tier': 'bronze',   'desc': '차 5대 추월'},
        {'id': 'score_20',    'title': '속도광',            'tier': 'bronze',   'desc': '차 20대 추월'},
        {'id': 'score_50',    'title': '레이서',            'tier': 'bronze',   'desc': '차 50대 추월'},
        # silver
        {'id': 'score_100',   'title': '프로 드라이버',     'tier': 'silver',   'desc': '차 100대 추월'},
        {'id': 'score_200',   'title': 'Day 1 클리어',      'tier': 'silver',   'desc': 'Day 1 목표(200대) 달성'},
        {'id': 'bench_dqn',   'title': 'DQN 돌파',          'tier': 'silver',   'desc': 'DQN 기준(302대) 초과'},
        # gold
        {'id': 'bench_human', 'title': '인간 평균 돌파',    'tier': 'gold',     'desc': '인간 평균(309대) 초과'},
        {'id': 'score_500',   'title': '슈퍼 레이서',       'tier': 'gold',     'desc': '차 500대 추월'},
        {'id': 'score_800',   'title': '레이싱 전설',       'tier': 'gold',     'desc': '차 800대 추월'},
        # platinum
        {'id': 'bench_ddqn',  'title': 'DDQN 돌파',         'tier': 'platinum', 'desc': 'DDQN 기준(320대) 초과'},
    ]

    def _on_episode_reset(self):
        self._rt = {'score': 0.0}

    def _check_realtime_achievements(self, entry: dict) -> list:
        if not hasattr(self, '_rt'):
            return []
        rt = self._rt
        new = []

        def unlock(id_, title, desc, tier, condition):
            if condition and id_ not in self._ach_unlocked:
                self._ach_unlocked.add(id_)
                new.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        rt['score'] += entry.get('reward', 0)
        s = rt['score']

        unlock('score_5',     '첫 추월',       '차 5대 추월',             'bronze', s >= 5)
        unlock('score_20',    '속도광',         '차 20대 추월',            'bronze', s >= 20)
        unlock('score_50',    '레이서',         '차 50대 추월',            'bronze', s >= 50)
        unlock('score_100',   '프로 드라이버',  '차 100대 추월',           'silver', s >= 100)
        unlock('score_200',   'Day 1 클리어',   'Day 1 목표(200대) 달성',  'silver', s >= 200)
        unlock('bench_dqn',   'DQN 돌파',       'DQN 기준(302대) 초과',    'silver', s > 302)
        unlock('bench_human', '인간 평균 돌파', '인간 평균(309대) 초과',   'gold',   s > 309)
        unlock('score_500',   '슈퍼 레이서',    '차 500대 추월',           'gold',   s >= 500)
        unlock('score_800',   '레이싱 전설',    '차 800대 추월',           'gold',   s >= 800)
        unlock('bench_ddqn',  'DDQN 돌파',      'DDQN 기준(320대) 초과', 'platinum', s > 320)
        return new

    def _compute_achievements(self) -> list:
        data = [d for d in self.episode_data if d.get('action') is not None]
        if not data:
            return []
        s = episode_score(data)
        achieved = []

        def add(id_, title, desc, tier, cond):
            if cond:
                achieved.append({'id': id_, 'title': title, 'desc': desc, 'tier': tier})

        add('score_5',     '첫 추월',       '차 5대 추월',             'bronze', s >= 5)
        add('score_20',    '속도광',         '차 20대 추월',            'bronze', s >= 20)
        add('score_50',    '레이서',         '차 50대 추월',            'bronze', s >= 50)
        add('score_100',   '프로 드라이버',  '차 100대 추월',           'silver', s >= 100)
        add('score_200',   'Day 1 클리어',   'Day 1 목표(200대) 달성',  'silver', s >= 200)
        add('bench_dqn',   'DQN 돌파',       'DQN 기준(302대) 초과',    'silver', s > 302)
        add('bench_human', '인간 평균 돌파', '인간 평균(309대) 초과',   'gold',   s > 309)
        add('score_500',   '슈퍼 레이서',    '차 500대 추월',           'gold',   s >= 500)
        add('score_800',   '레이싱 전설',    '차 800대 추월',           'gold',   s >= 800)
        add('bench_ddqn',  'DDQN 돌파',      'DDQN 기준(320대) 초과', 'platinum', s > 320)
        return achieved

    # ── 속도 게이지 (ram[39] 스크롤 카운터 기반) ─────────────────────────
    # 실측: GAS 풀가속 +25.1/frame, BRAKE +3.0/frame → 8.4배 차이
    _SPEED_RAM_ADDR = 39
    _SPEED_MAX_RATE = 25.0   # GAS 풀가속 기준 100%
    _SPEED_WIN      = 30     # 이동평균 윈도우

    def _extra_frame_data(self, obs_raw: np.ndarray) -> dict:
        base = detect_enduro_phase(obs_raw)
        base['speed_pct'] = self._compute_speed_from_ram()
        return base

    def _compute_speed_from_ram(self) -> float:
        try:
            ram = self._env.unwrapped.ale.getRAM()
        except Exception:
            return 0.0

        val = int(ram[self._SPEED_RAM_ADDR])

        if not hasattr(self, '_speed_ram_prev'):
            self._speed_ram_prev = val
            self._speed_ram_window: list[float] = []
            return 0.0

        # 8-bit 순환 카운터 wrap-around 보정
        delta = val - self._speed_ram_prev
        if delta > 128:  delta -= 256
        if delta < -128: delta += 256
        self._speed_ram_prev = val

        self._speed_ram_window.append(max(0.0, float(delta)))
        if len(self._speed_ram_window) > self._SPEED_WIN:
            self._speed_ram_window.pop(0)

        smooth = float(np.mean(self._speed_ram_window))
        return round(min(100.0, smooth / self._SPEED_MAX_RATE * 100.0), 1)

    def _extra_summary(self, entry: dict) -> dict:
        result: dict = {}
        # RAM 기반 게임 상태
        pre_ram = entry.get('pre_ram')
        if pre_ram is not None:
            gs = extract_enduro_game_state(pre_ram)
            if gs:
                result['game_state'] = gs
        # 스텝 기반 시간대 (코칭 LLM에게 현재 환경 전달)
        step = entry.get('step', 0)
        result.update(phase_from_step(step))
        return result

    def _load_model(self, path: str):
        if not os.path.exists(path):
            return None
        from ai_agents.enduro import load_enduro_d3qn
        net, _ = load_enduro_d3qn(path, self.device)
        return net

    def _get_q_values(self, stacked_state):
        from ai_agents.enduro import get_q_values
        return get_q_values(self.net, stacked_state, self.device)
    game_info = {
        'summary': '하루 단위로 정해진 수의 차량을 추월하며 최대한 오래 달리는 지구력 레이싱 게임입니다.',
        'objective': 'Day 1은 200대, Day 2부터는 매일 300대를 하루가 끝나기 전에 추월하세요.',
        'how_to_play': [
            'SPACE(GAS) : 가속 — 누르고 있으면 최고 속도까지 올라감. 원하는 속도에서 떼면 그 속도 유지.',
            '↓ : 브레이크 — GAS를 떼고 아래 방향 입력으로 속도를 줄임.',
            '← → : 좌우 조향으로 차량을 피함. GAS와 동시 입력으로 가속 중에도 회피 가능.',
            '목표 대수를 채우면 초록 깃발이 표시되지만, 하루가 끝날 때까지 계속 달리며 추가 추월 가능.',
        ],
        'scoring': [
            '차량 1대 추월: 1점',
            'Day 목표 달성 → 다음 Day 진입 (Day 1: 200대, Day 2 이후: 매일 300대)',
            '화면 상단 숫자 = 주행 거리계(odometer) — 누적 주행 거리를 나타내며 달리는 동안 계속 올라감. 빠를수록 빨리 올라가고, 충돌로 속도가 줄면 즉시 느려짐.',
            '화면 하단 왼쪽 숫자 = 현재 Day / 오른쪽 숫자 = 오늘 남은 목표 추월 대수',
        ],
        'end_condition': '하루가 끝날 때까지 목표 추월 대수를 채우지 못하면 게임이 종료됩니다.',
        'collision': (
            'Enduro에는 목숨 개념이 없습니다. 다른 차에 부딪히면 죽는 것이 아니라 속도가 크게 줄어 '
            '추월 페이스가 떨어집니다. 도로 가장자리로 빠지는 것은 차와 정면 충돌보다 손실이 적습니다.'
        ),
        'tips': [
            '무조건 최고속으로만 달리면 충돌이 잦아 오히려 페이스가 떨어집니다.',
            '야간에는 다른 차의 후미등만 보이므로 속도를 줄여 충돌을 예방하세요.',
            '안개 구간에서는 앞차를 더 늦게 인식하게 되므로 반응 시간이 줄어듭니다.',
            '눈길/얼음길에서는 조향 반응이 둔해지므로 미리 진로를 잡아야 합니다.',
            '차와 충돌할 것 같으면 도로 가장자리로 빠지는 편이 덜 손실입니다.',
        ],
        'controls': {
            'left':  '왼쪽 이동',
            'right': '오른쪽 이동',
            'down':  '브레이크',
            'fire':  '가속 (GAS)',
        },
    }
