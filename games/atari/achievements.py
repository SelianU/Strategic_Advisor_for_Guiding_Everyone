"""
games/atari/achievements.py — 선언적 도전과제 스펙 엔진.

게임 클래스가 `achievement_specs` 리스트만 정의하면
  ① UI용 achievements 목록
  ② 실시간(스텝 단위) 달성 판정
  ③ 게임 종료 후 일괄 판정
을 모두 공통 로직으로 처리한다. (기존에는 게임마다 같은 정보를 3벌씩 중복 정의)

스펙 형식 (dict):
  공통 필드: id, title, tier, desc
  metric 필드:
    {'metric': 'score',        'value': 200}                  # 누적 점수 >= 200
    {'metric': 'score',        'value': 1496, 'cmp': '>'}     # 벤치마크 초과 (>)
    {'metric': 'combo',        'value': 3, 'window': 200}     # window 스텝 내 보상 이벤트 n회
    {'metric': 'streak',       'value': 5, 'gap': 60}         # 간격 gap 이내 연속 보상 n회
                                                              #   (목숨 손실 시 연속 초기화)
    {'metric': 'survive',      'value': 500}                  # n스텝 생존
    {'metric': 'no_death',     'value': 1000}                 # 무사망 + n스텝 생존
    {'metric': 'reward_count', 'value': 10, 'reward_exact': 10}   # reward == x 이벤트 n회
    {'metric': 'reward_count', 'value': 1,  'reward_min': 400}    # reward >= x 이벤트 n회

combo/streak 의 보상 이벤트 범위는 reward > 0 (기존 동작과 동일).
복잡한 게임별 판정(라운드 클리어, RAM 기반 등)은 기존처럼
_check_realtime_achievements / _compute_achievements override 로 처리한다.
"""
from .ach_helper import episode_score, combo_max, streak_max, life_losses


def specs_to_achievement_list(specs: list) -> list:
    """스펙 → UI/템플릿용 도전과제 목록 ({'id','title','tier','desc'})."""
    return [{'id': s['id'], 'title': s['title'], 'tier': s['tier'], 'desc': s['desc']}
            for s in specs]


def _passed(value: float, spec: dict) -> bool:
    threshold = spec['value']
    return value > threshold if spec.get('cmp') == '>' else value >= threshold


class AchievementTracker:
    """실시간(스텝 단위) 도전과제 상태 추적기.

    update() 를 매 스텝 호출하면 점수·콤보 윈도우·연속 기록·보상 카운트·
    목숨 손실을 증분 갱신하고, metric_value() 로 스펙별 현재 값을 조회한다.
    """

    def __init__(self, specs: list):
        self.specs = specs
        self.score = 0.0
        self.life_losses = 0
        self._prev_lives = None
        # (window) → 보상 이벤트 스텝 리스트
        self._windows = {s['window'] for s in specs if s['metric'] == 'combo'}
        self._window_events = {w: [] for w in self._windows}
        # (gap) → 연속 기록 상태
        self._gaps = {s['gap'] for s in specs if s['metric'] == 'streak'}
        self._streaks = {g: {'cur': 0, 'max': 0, 'last': -1} for g in self._gaps}
        # reward_count 스펙별 카운터
        self._counts = {self._count_key(s): 0
                        for s in specs if s['metric'] == 'reward_count'}

    @staticmethod
    def _count_key(spec: dict):
        return ('exact', spec['reward_exact']) if 'reward_exact' in spec \
            else ('min', spec['reward_min'])

    def update(self, entry: dict):
        reward = entry.get('reward', 0)
        step   = entry.get('step', 0)
        lives  = entry.get('lives')

        self.score += reward

        # 목숨 손실 — streak 초기화 + 손실 누적
        if lives is not None:
            if self._prev_lives is not None and lives < self._prev_lives:
                self.life_losses += self._prev_lives - lives
                for st in self._streaks.values():
                    st['cur'], st['last'] = 0, -1
            self._prev_lives = lives

        if reward > 0:
            for w, events in self._window_events.items():
                events.append(step)
            for g, st in self._streaks.items():
                gap = step - st['last'] if st['last'] >= 0 else g + 1
                st['cur'] = st['cur'] + 1 if gap <= g else 1
                st['max'] = max(st['max'], st['cur'])
                st['last'] = step
            for key in self._counts:
                kind, x = key
                if (kind == 'exact' and reward == x) or (kind == 'min' and reward >= x):
                    self._counts[key] += 1

        # 콤보 윈도우 유지
        for w, events in self._window_events.items():
            self._window_events[w] = [s for s in events if step - s <= w]

    def metric_value(self, spec: dict, steps_total: int):
        m = spec['metric']
        if m == 'score':
            return self.score
        if m == 'combo':
            return len(self._window_events[spec['window']])
        if m == 'streak':
            return self._streaks[spec['gap']]['max']
        if m == 'survive':
            return steps_total
        if m == 'no_death':
            return steps_total if self.life_losses == 0 else -1
        if m == 'reward_count':
            return self._counts[self._count_key(spec)]
        raise ValueError(f'알 수 없는 metric: {m}')

    def check(self, steps_total: int, already_unlocked: set) -> list:
        """새로 달성된 도전과제 목록 반환 (이미 달성된 id 는 제외)."""
        new = []
        for spec in self.specs:
            if spec['id'] in already_unlocked:
                continue
            if _passed(self.metric_value(spec, steps_total), spec):
                already_unlocked.add(spec['id'])
                new.append({'id': spec['id'], 'title': spec['title'],
                            'desc': spec['desc'], 'tier': spec['tier']})
        return new


def compute_final_achievements(specs: list, data: list) -> list:
    """게임 종료 후 에피소드 데이터 전체로 일괄 판정."""
    if not data:
        return []
    score = episode_score(data)
    ll    = life_losses(data)
    steps = len(data)

    def value(spec: dict):
        m = spec['metric']
        if m == 'score':
            return score
        if m == 'combo':
            return combo_max(data, window=spec['window'], min_r=0, max_r=1e9)
        if m == 'streak':
            return streak_max(data, gap=spec['gap'], min_r=0, max_r=1e9)
        if m == 'survive':
            return steps
        if m == 'no_death':
            return steps if ll == 0 else -1
        if m == 'reward_count':
            if 'reward_exact' in spec:
                return sum(1 for d in data if d.get('reward', 0) == spec['reward_exact'])
            return sum(1 for d in data if d.get('reward', 0) >= spec['reward_min'])
        raise ValueError(f'알 수 없는 metric: {m}')

    return [{'id': s['id'], 'title': s['title'], 'desc': s['desc'], 'tier': s['tier']}
            for s in specs if _passed(value(s), s)]
