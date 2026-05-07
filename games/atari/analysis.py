"""
analysis.py (games.atari.analysis)

Q-value 기반 분석 mixin.

- _select_top5      : worst-N 선정 (스텝 간격 보장)
- _analysis_snapshot: 현재 analysis_results 의 요약 (저장 / 로드용)
- _basic_analyze    : 기본 점수/스텝 통계 (모델 없을 때도 동작)
- _run_analysis     : 백그라운드에서 모든 valid entry 의 Q-value 분석

이전 위치: atari_base.py 의 동명 메서드들.
`AtariGame` 의 다중 상속 부모로 사용되며, 단독 인스턴스화하지 않는다.
"""
from collections import Counter

import numpy as np


class _AnalysisMixin:

    def _select_top5(self, analyses: list, k: int = 5, min_gap: int = 3) -> list:
        if not analyses:
            return []
        total = len(analyses)
        seg_size = max(1, total // k)
        selected = []
        for seg_idx in range(k):
            start = seg_idx * seg_size
            end = total if seg_idx == k - 1 else (seg_idx + 1) * seg_size
            candidates = sorted(analyses[start:end], key=lambda a: a['loss'], reverse=True)
            for c in candidates:
                if all(abs(c['step'] - s['step']) >= min_gap for s in selected):
                    selected.append(c)
                    break
        return sorted(selected, key=lambda a: a['loss'], reverse=True)

    def _analysis_snapshot(self) -> 'dict | None':
        if not self.analysis_results:
            return None
        losses = [a['loss'] for a in self.analysis_results]
        p_cnt  = Counter(a['action_name']      for a in self.analysis_results)
        ai_cnt = Counter(a['best_action_name'] for a in self.analysis_results)
        return {
            'avg_loss':      round(float(np.mean(losses)), 3),
            'worst':         max(self.analysis_results, key=lambda a: a['loss']),
            'worst_10':      self._select_top5(self.analysis_results),
            'total_steps':   len(self.analysis_results),
            'agree_rate':    round(sum(1 for a in self.analysis_results if a['is_best']) / len(self.analysis_results) * 100, 1),
            'player_actions': dict(p_cnt),
            'ai_actions':    dict(ai_cnt),
        }

    def _basic_analyze(self) -> dict:
        acts   = [d['action'] for d in self.episode_data if d['action'] is not None]
        total  = len(acts) or 1
        reward = sum(d['reward'] for d in self.episode_data)
        return {'total_reward': int(reward), 'total_steps': total}

    def _run_analysis(self):
        valid = [d for d in self.episode_data if d['action'] is not None]
        if not valid:
            return
        sid   = self.session_id
        total = len(valid)
        self.valid_entries = valid
        self.socketio.emit(f'{self.prefix}analysis_start', {'total': total, 'session_id': sid})

        analyses = []
        for i, entry in enumerate(valid):
            if sid != self.session_id:
                return   # 새 게임 시작됨
            if i % 50 == 0:
                self.socketio.emit(f'{self.prefix}analysis_progress', {
                    'current': i, 'total': total,
                    'pct': round(i / total * 100), 'session_id': sid,
                })
            q_vals   = self._get_q_values(entry['pre_stacked_state'])
            action   = entry['action']
            best_act = int(np.argmax(q_vals))
            player_q = float(q_vals[action])
            best_q   = float(q_vals[best_act])
            analyses.append({
                'step':             entry['step'],
                'action':           action,
                'entry_index':      i,
                'action_name':      self.action_names.get(action,   str(action)),
                'best_action':      best_act,
                'best_action_name': self.action_names.get(best_act, str(best_act)),
                'q_values':         [round(float(v), 3) for v in q_vals],
                'player_q':         round(player_q, 3),
                'best_q':           round(best_q, 3),
                'loss':             round(best_q - player_q, 3),
                'reward':           entry['reward'],
                'is_best':          (action == best_act),
            })

        losses   = [a['loss'] for a in analyses]
        avg_loss = round(float(np.mean(losses)), 3)
        worst    = max(analyses, key=lambda a: a['loss'])
        worst_10 = self._select_top5(analyses)
        agree    = round(sum(1 for a in analyses if a['is_best']) / len(analyses) * 100, 1)
        self.analysis_results = analyses

        p_cnt  = Counter(a['action_name']      for a in analyses)
        ai_cnt = Counter(a['best_action_name'] for a in analyses)

        self.socketio.emit(f'{self.prefix}analysis_done', {
            'avg_loss': avg_loss, 'worst': worst, 'worst_10': worst_10,
            'total_steps': total, 'agree_rate': agree,
            'player_actions': dict(p_cnt), 'ai_actions': dict(ai_cnt),
            'session_id': sid,
        })
        print(f'✅ [{self.game_id}] 분석 완료 avg_loss={avg_loss:.3f} agree={agree}%')