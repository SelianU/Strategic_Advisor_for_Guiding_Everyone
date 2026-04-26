"""
feedback_analyzer.py  —  AI 피드백 분석기

human_play_recorder.py 로 기록된 플레이 데이터를 분석합니다.

분석 내용:
  ① Q값 비교
     - 사람이 선택한 행동의 Q값
     - AI 최선 행동과 그 Q값
     - Q 손실(opportunity cost) = max Q - 사람 Q
     - 손실이 큰 순서로 "놓친 순간" 랭킹

  ② Advantage stream Grad-CAM
     - 각 분석 프레임에서 AI가 집중한 영역 시각화

  ③ 리포트 출력
     - 개별 프레임 PNG
     - 요약 HTML 리포트 (브라우저에서 열람 가능)

사용법:
    python feedback_analyzer.py --data ./human_data/play_001.pkl
    python feedback_analyzer.py --data ./human_data/play_001.pkl \\
        --model ./checkpoints_v3/best_model.pth \\
        --top 10 --out ./feedback_report
"""

import os
import sys
import pickle
import argparse
import base64
from io import BytesIO

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import gymnasium as gym
import ale_py  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D3QN_v3 import Config, DuelingDQN, GradRescale

# =============================================================================
# 설정
# =============================================================================
DEFAULT_MODEL   = "./checkpoints_v3/best_model.pth"
DEFAULT_OUT_DIR = "./feedback_report"
DEFAULT_TOP_N   = 8    # 가장 손실 큰 순간 N개 분석

ACTION_NAMES = {
    0: "NOOP", 1: "FIRE", 2: "RIGHT",
    3: "LEFT", 4: "RIGHT+FIRE", 5: "LEFT+FIRE",
}
ACTION_COLORS = {
    0: "#888888", 1: "#FFD700", 2: "#4FC3F7",
    3: "#81C784", 4: "#FF8A65", 5: "#CE93D8",
}


# =============================================================================
# 모델 & Grad-CAM 로드
# =============================================================================
def load_model(model_path, device):
    config  = Config()
    env_tmp = gym.make(config.ENV_NAME, frameskip=1, repeat_action_probability=0.0)
    n_actions = env_tmp.action_space.n
    env_tmp.close()

    net = DuelingDQN((4, 84, 84), n_actions).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd   = ckpt.get('q_network', ckpt)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    net.load_state_dict(cleaned)
    net.eval()
    print(f"모델 로드: {model_path}  (frame={ckpt.get('frame',0):,})")
    return net, n_actions


class AdvantageGradCAM:
    def __init__(self, net, device):
        self.net    = net
        self.device = device
        self._grads = None
        self._feats = None
        self._hooks = []

        last_conv = None
        for m in self.net.features.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m

        self._hooks.append(last_conv.register_forward_hook(
            lambda mo, inp, out: setattr(self, '_feats', out.detach())))
        self._hooks.append(last_conv.register_full_backward_hook(
            lambda mo, gi, go: setattr(self, '_grads', go[0].detach())))

    def remove(self):
        for h in self._hooks: h.remove()

    def __call__(self, state_uint8):
        self.net.zero_grad()
        s = torch.from_numpy(state_uint8.astype(np.float32)).unsqueeze(0).to(self.device)

        x        = s.contiguous().float().mul_(1.0 / 255.0)
        features = self.net.features(x).flatten(1)
        features = self.net.grad_rescale(features)

        value     = self.net.value_stream(features)
        advantage = self.net.advantage_stream(features)
        q_values  = (value + advantage - advantage.mean(dim=1, keepdim=True))[0]

        ai_action = q_values.argmax().item()
        advantage[0, ai_action].backward()

        grads   = self._grads[0]
        feats   = self._feats[0]
        weights = grads.mean(dim=(1,2), keepdim=True)
        cam     = F.relu((weights * feats).sum(dim=0)).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        heatmap = cv2.resize(cam, (84, 84), interpolation=cv2.INTER_LINEAR)

        return heatmap, ai_action, q_values.detach().cpu().numpy()


# =============================================================================
# 핵심 분석: Q값 비교
# =============================================================================
def analyze_frames(frames, net, grad_cam, device):
    """
    모든 프레임에 대해 Q값 계산 후
    opportunity_cost(AI 최선 Q - 사람 Q) 기준으로 정렬
    """
    results = []

    for fr in frames:
        state  = fr["state"]        # (4,84,84) uint8
        human_action = fr["action"]

        with torch.no_grad():
            s  = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
            s  = s.float().mul_(1.0 / 255.0)
            ft = net.features(s).flatten(1)
            ft = net.grad_rescale(ft)
            v  = net.value_stream(ft)
            a  = net.advantage_stream(ft)
            q  = (v + a - a.mean(dim=1, keepdim=True))[0].cpu().numpy()

        ai_action        = int(q.argmax())
        human_q          = float(q[human_action])
        ai_q             = float(q[ai_action])
        opportunity_cost = ai_q - human_q   # 클수록 사람이 손해 본 순간

        results.append({
            **fr,
            "q_values":         q,
            "ai_action":        ai_action,
            "human_q":          human_q,
            "ai_q":             ai_q,
            "opportunity_cost": opportunity_cost,
        })

    return results


# =============================================================================
# 시각화: 개별 피드백 프레임
# =============================================================================
def render_feedback_frame(result, grad_cam, out_path, rank):
    state       = result["state"]
    raw_frame   = result["raw_frame"]
    human_action = result["action"]
    ai_action   = result["ai_action"]
    q_values    = result["q_values"]
    opp_cost    = result["opportunity_cost"]
    score       = result["score"]
    step        = result["step"]

    # Grad-CAM
    heatmap, _, _ = grad_cam(state)
    H, W = raw_frame.shape[:2]
    heatmap_full = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

    fig = plt.figure(figsize=(16, 6), facecolor='#0E0E0E')
    gs  = gridspec.GridSpec(1, 4, figure=fig,
                             left=0.03, right=0.78,
                             top=0.83, bottom=0.08,
                             wspace=0.06)
    gs2 = gridspec.GridSpec(1, 1, figure=fig,
                             left=0.80, right=0.98,
                             top=0.83, bottom=0.08)

    # ── (A) 원본 프레임 ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(raw_frame)
    ax1.set_title("사람 플레이", color='white', fontsize=10, pad=4)
    # 사람 행동 표시
    ax1.text(0.5, -0.08,
             f"▶ {ACTION_NAMES[human_action]}",
             transform=ax1.transAxes, ha='center',
             color=ACTION_COLORS[human_action], fontsize=10, fontweight='bold')
    ax1.axis('off')

    # ── (B) AI Grad-CAM 오버레이 ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(raw_frame)
    ax2.imshow(heatmap_full, cmap='jet', alpha=0.55, vmin=0, vmax=1,
               extent=[0, W, H, 0])
    ax2.set_title("AI가 본 것 (Grad-CAM)", color='white', fontsize=10, pad=4)
    ax2.text(0.5, -0.08,
             f"▶ {ACTION_NAMES[ai_action]}",
             transform=ax2.transAxes, ha='center',
             color=ACTION_COLORS[ai_action], fontsize=10, fontweight='bold')
    ax2.axis('off')

    # ── (C) 전처리 프레임 ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(state[-1], cmap='gray', vmin=0, vmax=255)
    ax3.set_title("AI 입력 (전처리)", color='white', fontsize=10, pad=4)
    ax3.axis('off')

    # ── (D) Grad-CAM only ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.imshow(state[-1], cmap='gray', vmin=0, vmax=255)
    ax4.imshow(heatmap, cmap='jet', alpha=0.65, vmin=0, vmax=1)
    ax4.set_title("집중 영역 히트맵", color='white', fontsize=10, pad=4)
    ax4.axis('off')

    # ── (E) Q값 바 차트 ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs2[0])
    ax5.set_facecolor('#1A1A1A')

    actions = list(range(len(q_values)))
    colors  = []
    for i in actions:
        if i == ai_action:
            colors.append('#00E676')    # AI 최선 → 초록
        elif i == human_action:
            colors.append('#FF5252')    # 사람 선택 → 빨강
        else:
            colors.append('#555555')

    bars = ax5.barh(actions, q_values, color=colors, height=0.6)

    # 레이블
    for i, (bar, q) in enumerate(zip(bars, q_values)):
        label = ACTION_NAMES[i]
        if i == ai_action:
            label = f"★ {label}"
        elif i == human_action:
            label = f"✗ {label}"
        ax5.text(q + 0.05, i, f"{q:.2f}", va='center',
                 color='white', fontsize=7.5)
        ax5.text(min(q_values) - 0.2, i, label, va='center', ha='right',
                 color=colors[i], fontsize=8)

    ax5.set_xlim(min(q_values) - 1.5, max(q_values) + 1.5)
    ax5.set_yticks([])
    ax5.set_title("Q값 비교", color='white', fontsize=10, pad=4)
    ax5.tick_params(colors='#888', labelsize=7)
    ax5.spines['bottom'].set_color('#444')
    ax5.spines['top'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.xaxis.label.set_color('#888')

    # 범례
    ax5.text(0.02, -0.10, "★ AI 최선  ✗ 사람 선택",
             transform=ax5.transAxes, color='#AAAAAA', fontsize=7.5)

    # ── 타이틀 ───────────────────────────────────────────────────────────────
    same = human_action == ai_action
    if same:
        verdict = "✓ 최선의 행동!"
        v_color = "#00E676"
    else:
        verdict = f"△ 손실: +{opp_cost:.2f}Q  →  {ACTION_NAMES[ai_action]} 권장"
        v_color = "#FF8A65"

    fig.suptitle(
        f"[#{rank}]  Step {step}  |  Score {int(score)}  |  {verdict}",
        color=v_color, fontsize=12, fontweight='bold', y=0.97
    )

    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0E0E0E')
    plt.close(fig)


# =============================================================================
# HTML 리포트 생성
# =============================================================================
def make_html_report(top_results, all_results, out_dir, episode_info):
    """
    브라우저에서 열람 가능한 HTML 피드백 리포트 생성
    """

    def img_to_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # 전체 Q 손실 히스토그램
    opp_costs = [r["opportunity_cost"] for r in all_results]
    fig, ax = plt.subplots(figsize=(8, 3), facecolor='#111')
    ax.set_facecolor('#1A1A1A')
    ax.hist(opp_costs, bins=30, color='#4C9BE8', alpha=0.8, edgecolor='#222')
    ax.axvline(0, color='#00E676', lw=1.5, linestyle='--', label='Q 손실 = 0 (최선)')
    ax.set_xlabel("Opportunity Cost (AI Q - 사람 Q)", color='#AAA', fontsize=9)
    ax.set_ylabel("빈도", color='#AAA', fontsize=9)
    ax.set_title("전체 플레이 Q값 손실 분포", color='white', fontsize=10)
    ax.tick_params(colors='#888')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax.legend(fontsize=8, facecolor='#222', labelcolor='white')
    hist_path = os.path.join(out_dir, "_hist.png")
    plt.savefig(hist_path, dpi=100, bbox_inches='tight', facecolor='#111')
    plt.close(fig)

    # 점수별 행동 분포
    human_actions  = [r["action"] for r in all_results]
    action_counts  = {k: human_actions.count(k) for k in range(6)}

    # 통계
    total_steps    = len(all_results)
    optimal_steps  = sum(1 for r in all_results if r["opportunity_cost"] < 0.5)
    avg_cost       = np.mean(opp_costs)
    worst_moments  = sorted(all_results, key=lambda x: x["opportunity_cost"], reverse=True)[:3]

    # 프레임 이미지 b64 인라인
    frame_imgs_html = ""
    for rank, result in enumerate(top_results, 1):
        img_path = os.path.join(out_dir, f"rank_{rank:02d}_step_{result['step']:05d}.png")
        if os.path.exists(img_path):
            b64 = img_to_b64(img_path)
            opp  = result["opportunity_cost"]
            same = result["action"] == result["ai_action"]
            color = "#00E676" if same else "#FF8A65"
            label = "✓ 최선" if same else f"△ 손실 +{opp:.2f}Q"
            frame_imgs_html += f"""
            <div class="frame-card">
              <div class="frame-label" style="color:{color}">
                #{rank} Step {result['step']} | {label}
              </div>
              <img src="data:image/png;base64,{b64}" style="width:100%">
              <div class="action-info">
                사람: <span style="color:#FF8A65">{ACTION_NAMES[result['action']]}</span>
                &nbsp;→&nbsp;
                AI 권장: <span style="color:#00E676">{ACTION_NAMES[result['ai_action']]}</span>
                &nbsp;|&nbsp; Q차이: <span style="color:#FFD700">{opp:+.2f}</span>
              </div>
            </div>
            """

    hist_b64 = img_to_b64(hist_path)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>D3QN 피드백 리포트</title>
<style>
  body {{ background:#0D0D0D; color:#DDD; font-family:'Segoe UI',sans-serif; margin:0; padding:20px; }}
  h1   {{ color:#4C9BE8; border-bottom:2px solid #222; padding-bottom:10px; }}
  h2   {{ color:#81C784; margin-top:30px; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:20px 0; }}
  .stat-card {{ background:#1A1A1A; border-radius:8px; padding:16px; text-align:center; border:1px solid #333; }}
  .stat-val  {{ font-size:2em; font-weight:bold; color:#4C9BE8; }}
  .stat-lbl  {{ font-size:0.85em; color:#888; margin-top:4px; }}
  .frame-card {{ background:#151515; border-radius:8px; margin:16px 0; padding:12px; border:1px solid #2A2A2A; }}
  .frame-label {{ font-size:1em; font-weight:bold; margin-bottom:8px; }}
  .action-info {{ margin-top:8px; font-size:0.9em; color:#AAA; }}
  table {{ width:100%; border-collapse:collapse; margin:10px 0; }}
  th    {{ background:#1A1A1A; padding:8px; color:#888; font-size:0.85em; border-bottom:1px solid #333; }}
  td    {{ padding:8px; font-size:0.85em; border-bottom:1px solid #1A1A1A; }}
  tr:hover td {{ background:#1A1A1A; }}
</style>
</head>
<body>
<h1>🎮 D3QN 피드백 리포트</h1>
<p style="color:#666">에피소드 수: {episode_info['n_episodes']} &nbsp;|&nbsp;
   총 점수: {episode_info['total_score']} &nbsp;|&nbsp;
   총 스텝: {total_steps:,}</p>

<h2>📊 플레이 요약</h2>
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-val">{total_steps:,}</div>
    <div class="stat-lbl">총 스텝</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{optimal_steps/total_steps*100:.1f}%</div>
    <div class="stat-lbl">최선에 가까운 행동 비율<br><small>(Q 손실 &lt; 0.5)</small></div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:#FF8A65">{avg_cost:.2f}</div>
    <div class="stat-lbl">평균 Q 손실</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color:#FF5252">{worst_moments[0]['opportunity_cost']:.2f}</div>
    <div class="stat-lbl">최대 Q 손실 순간</div>
  </div>
</div>

<h2>📈 Q값 손실 분포</h2>
<img src="data:image/png;base64,{hist_b64}" style="max-width:700px;border-radius:8px">
<p style="color:#666;font-size:0.85em">
  Q 손실이 0에 가까울수록 AI와 같은 선택. 양수가 클수록 손해 본 순간.
</p>

<h2>🎯 행동별 선택 빈도</h2>
<table>
  <tr><th>행동</th><th>횟수</th><th>비율</th></tr>
  {''.join(f"<tr><td>{ACTION_NAMES[a]}</td><td>{action_counts[a]}</td><td>{action_counts[a]/total_steps*100:.1f}%</td></tr>" for a in range(6))}
</table>

<h2>🔴 가장 손해 본 순간 TOP {len(top_results)}</h2>
<p style="color:#666;font-size:0.85em">
  AI가 더 유리한 행동을 선택했을 순간들. Q값 차이가 클수록 개선 여지가 큰 순간입니다.
</p>
{frame_imgs_html}

<hr style="border-color:#222;margin-top:40px">
<p style="color:#444;font-size:0.8em">
  Generated by D3QN Feedback System &nbsp;|&nbsp;
  Model: best_model.pth (1034 pts eval) &nbsp;|&nbsp;
  ALE/SpaceInvaders-v5
</p>
</body>
</html>"""

    html_path = os.path.join(out_dir, "feedback_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 히스토그램 임시 파일 정리
    os.remove(hist_path)

    return html_path


# =============================================================================
# 메인
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="AI 피드백 분석기")
    parser.add_argument("--data",  required=True,          help="pkl 데이터 경로")
    parser.add_argument("--model", default=DEFAULT_MODEL,  help="모델 경로")
    parser.add_argument("--top",   default=DEFAULT_TOP_N,  type=int, help="분석할 top N 순간")
    parser.add_argument("--out",   default=DEFAULT_OUT_DIR, help="출력 폴더")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터 로드
    with open(args.data, "rb") as f:
        episodes = pickle.load(f)
    print(f"데이터 로드: {len(episodes)}개 에피소드")

    # 모델 로드
    net, n_actions = load_model(args.model, device)
    grad_cam = AdvantageGradCAM(net, device)

    # 전체 프레임 수집
    all_frames = []
    for ep in episodes:
        all_frames.extend(ep["frames"])
    print(f"총 분석 프레임: {len(all_frames):,}")

    # Q값 분석
    print("Q값 분석 중...")
    results = analyze_frames(all_frames, net, grad_cam, device)

    # opportunity cost 기준 정렬 → Top N
    sorted_results = sorted(results, key=lambda x: x["opportunity_cost"], reverse=True)
    top_results    = sorted_results[:args.top]

    # ── 터미널 요약 출력 ─────────────────────────────────────────────────────
    opp_costs     = [r["opportunity_cost"] for r in results]
    optimal_steps = sum(1 for c in opp_costs if c < 0.5)
    total_steps   = len(results)
    total_score   = sum(ep["total_score"] for ep in episodes)

    print()
    print("=" * 60)
    print("  [분석 결과 요약]")
    print("=" * 60)
    print(f"  에피소드 수  : {len(episodes)}개")
    print(f"  총 점수      : {int(total_score)}")
    print(f"  총 스텝      : {total_steps:,}")
    print(f"  최선 행동 비율: {optimal_steps/total_steps*100:.1f}%  "
          f"({optimal_steps:,}/{total_steps:,} 스텝, Q 손실 < 0.5)")
    print(f"  평균 Q 손실  : {np.mean(opp_costs):.3f}")
    print(f"  최대 Q 손실  : {max(opp_costs):.3f}  (step {sorted_results[0]['step']})")
    print()

    # 행동별 선택 빈도
    human_actions = [r["action"] for r in results]
    print("  [행동별 선택 빈도]")
    for a in range(6):
        cnt   = human_actions.count(a)
        bar   = "█" * int(cnt / total_steps * 40)
        print(f"    {ACTION_NAMES[a]:12s} {cnt:5d}회 ({cnt/total_steps*100:4.1f}%)  {bar}")
    print()

    # Top N 손실 순간 터미널 출력
    print(f"  [가장 손해 본 순간 TOP {args.top}]")
    print(f"  {'순위':>4}  {'Step':>6}  {'Score':>6}  {'사람 행동':>12}  {'AI 권장':>12}  {'Q 손실':>8}")
    print("  " + "-" * 58)
    for rank, r in enumerate(top_results, 1):
        same   = r["action"] == r["ai_action"]
        marker = "  OK" if same else " !!"
        print(f"  {rank:>4}  {r['step']:>6}  {int(r['score']):>6}  "
              f"{ACTION_NAMES[r['action']]:>12}  {ACTION_NAMES[r['ai_action']]:>12}  "
              f"{r['opportunity_cost']:>+8.3f}{marker}")
    print()

    # 개별 프레임 이미지 생성
    print(f"  피드백 이미지 생성 중 (Top {args.top})...")
    for rank, result in enumerate(top_results, 1):
        img_path = os.path.join(args.out, f"rank_{rank:02d}_step_{result['step']:05d}.png")
        render_feedback_frame(result, grad_cam, img_path, rank)
        print(f"    [{rank}/{args.top}] 저장 → {os.path.basename(img_path)}")

    # 에피소드 요약 정보
    episode_info = {
        "n_episodes":  len(episodes),
        "total_score": total_score,
    }

    # HTML 리포트 생성
    print()
    print("  HTML 리포트 생성 중...")
    html_path = make_html_report(top_results, results, args.out, episode_info)

    grad_cam.remove()

    print()
    print("=" * 60)
    print("  완료!")
    print(f"  이미지  : {args.out}/rank_*.png  ({args.top}개)")
    print(f"  리포트  : {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()