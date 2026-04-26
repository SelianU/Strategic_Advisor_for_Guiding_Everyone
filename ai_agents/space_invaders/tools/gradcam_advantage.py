"""
gradcam_advantage.py  —  D3QN Advantage Stream Grad-CAM 시각화

Grad-CAM을 Advantage stream에 적용:
  - 선택된 행동의 Advantage A(s, a*)에 대해
  - 마지막 Conv layer(features[-1]) feature map의 기여도를 계산
  - 원본 게임 화면에 heatmap으로 오버레이

사용법:
    python gradcam_advantage.py
    python gradcam_advantage.py --model ./checkpoints_v3/best_model.pth --frames 8 --out ./gradcam_out

출력:
    gradcam_out/gradcam_grid.png   ← 논문용 그리드 이미지
    gradcam_out/frame_XXX.png      ← 개별 프레임
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import gymnasium as gym
import ale_py  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from D3QN_v3 import Config, DuelingDQN, GradRescale, make_env

# =============================================================================
# 설정
# =============================================================================
DEFAULT_MODEL   = "./checkpoints_v3/best_model.pth"
DEFAULT_OUT_DIR = "./gradcam_out"
DEFAULT_FRAMES  = 8     # 캡처할 프레임 수
EVAL_EPSILON    = 0.001
CAPTURE_EVERY   = 60    # 몇 스텝마다 캡처할지 (너무 촘촘하면 비슷한 장면만 나옴)

ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHT+FIRE",
    5: "LEFT+FIRE",
}

# 논문용 커스텀 컬러맵: 투명 → 파랑 → 초록 → 노랑 → 빨강
GRADCAM_CMAP = LinearSegmentedColormap.from_list(
    'gradcam',
    [(0.0, (0.0, 0.0, 0.8, 0.0)),    # 투명
     (0.3, (0.0, 0.6, 1.0, 0.4)),    # 하늘색
     (0.6, (0.2, 1.0, 0.2, 0.6)),    # 초록
     (0.8, (1.0, 1.0, 0.0, 0.75)),   # 노랑
     (1.0, (1.0, 0.1, 0.0, 0.9))],   # 빨강
)


# =============================================================================
# 모델 로드
# =============================================================================
def load_model(model_path, device):
    config  = Config()
    env_tmp = make_env(config)
    obs, _  = env_tmp.reset()
    input_shape = obs.shape
    n_actions   = env_tmp.action_space.n
    env_tmp.close()

    net = DuelingDQN(input_shape, n_actions).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd   = ckpt.get('q_network', ckpt)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    net.load_state_dict(cleaned)
    net.eval()
    print(f"모델 로드: {model_path}  (frame={ckpt.get('frame',0):,})")
    return net, input_shape, n_actions


# =============================================================================
# Grad-CAM (Advantage stream)
# =============================================================================
class AdvantageGradCAM:
    """
    Grad-CAM for the Advantage stream of DuelingDQN.

    타겟: A(s, a*)  — 에이전트가 선택한 행동의 Advantage 값
    소스: self.net.features[-1] (마지막 Conv, 64ch 7×7)

    절차:
        1. forward pass → A(s, a*)
        2. backward → 마지막 Conv의 gradient 수집
        3. GAP(gradient) → 채널별 가중치 α_k
        4. heatmap = ReLU( Σ_k α_k * A_k )  [feature map]
        5. 84×84 → 원본 해상도로 업샘플링
    """

    def __init__(self, net: DuelingDQN, device):
        self.net    = net
        self.device = device
        self._grads = None
        self._feats = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # 마지막 Conv layer (features의 마지막 Conv2d)
        last_conv = None
        for m in self.net.features.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m

        def fwd_hook(module, inp, out):
            self._feats = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        self._hooks.append(last_conv.register_forward_hook(fwd_hook))
        self._hooks.append(last_conv.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, state_uint8: np.ndarray):
        """
        state_uint8: (4, 84, 84) uint8
        반환: (heatmap_84x84, chosen_action, advantage_values, q_values)
        """
        self.net.zero_grad()

        s = torch.from_numpy(state_uint8.astype(np.float32)).unsqueeze(0).to(self.device)

        # ── forward (advantage stream만 분리해서 grad 계산) ──────────────────
        x        = s.contiguous().float().mul_(1.0 / 255.0)
        features = self.net.features(x).flatten(1)
        features = self.net.grad_rescale(features)

        value     = self.net.value_stream(features)
        advantage = self.net.advantage_stream(features)
        q_values  = value + advantage - advantage.mean(dim=1, keepdim=True)

        chosen_action = q_values.argmax(dim=1).item()

        # ── backward: 선택된 행동의 Advantage에 대해 ────────────────────────
        target = advantage[0, chosen_action]
        target.backward()

        # ── Grad-CAM 계산 ────────────────────────────────────────────────────
        grads = self._grads[0]    # (64, 7, 7)
        feats = self._feats[0]    # (64, 7, 7)

        weights   = grads.mean(dim=(1, 2), keepdim=True)  # GAP → (64,1,1)
        cam       = F.relu((weights * feats).sum(dim=0))  # (7, 7)
        cam_np    = cam.cpu().numpy()

        # 84×84로 업샘플
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        heatmap = cv2.resize(cam_np, (84, 84), interpolation=cv2.INTER_LINEAR)

        return (heatmap,
                chosen_action,
                advantage[0].detach().cpu().numpy(),
                q_values[0].detach().cpu().numpy())


# =============================================================================
# 시각화: 단일 프레임
# =============================================================================
def visualize_frame(raw_rgb, state_uint8, heatmap,
                    action, advantage, q_values,
                    frame_idx, score, out_path):
    """
    raw_rgb   : (H, W, 3) 원본 게임 화면
    state_uint8: (4, 84, 84)
    heatmap   : (84, 84) 0~1
    """
    H, W = raw_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

    fig = plt.figure(figsize=(14, 5), facecolor='#0D0D0D')
    gs  = gridspec.GridSpec(1, 4, figure=fig,
                             left=0.03, right=0.97,
                             top=0.82, bottom=0.10,
                             wspace=0.08)

    # ── (A) 원본 게임 화면 ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(raw_rgb)
    ax1.set_title("Game Frame", color='white', fontsize=10, pad=4)
    ax1.axis('off')

    # ── (B) 마지막 스택 프레임 (흑백) ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(state_uint8[-1], cmap='gray', vmin=0, vmax=255)
    ax2.set_title("Processed Frame (t)", color='white', fontsize=10, pad=4)
    ax2.axis('off')

    # ── (C) Grad-CAM 히트맵 단독 ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(state_uint8[-1], cmap='gray', vmin=0, vmax=255)
    ax3.imshow(heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    ax3.set_title("Grad-CAM\n(Advantage Stream)", color='white', fontsize=10, pad=4)
    ax3.axis('off')

    # ── (D) 원본에 오버레이 ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.imshow(raw_rgb)
    ax4.imshow(heatmap_resized, cmap='jet', alpha=0.55, vmin=0, vmax=1,
               extent=[0, W, H, 0])
    ax4.set_title("Overlay on Game", color='white', fontsize=10, pad=4)
    ax4.axis('off')

    # ── Advantage bar chart (하단 텍스트) ─────────────────────────────────────
    action_name = ACTION_NAMES.get(action, str(action))
    adv_str = "  ".join(
        f"{'→' if i == action else ' '}{ACTION_NAMES.get(i,i)}:{advantage[i]:+.2f}"
        for i in range(len(advantage))
    )
    fig.text(0.5, 0.04, adv_str,
             ha='center', va='bottom', fontsize=7.5,
             color='#CCCCCC', fontfamily='monospace')

    fig.suptitle(
        f"Frame #{frame_idx}  |  Score: {int(score)}  |  "
        f"Action: {action_name}  |  max Q: {q_values[action]:.2f}",
        color='white', fontsize=11, y=0.95, fontweight='bold'
    )

    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0D0D0D')
    plt.close(fig)


# =============================================================================
# 그리드 요약 이미지 (논문용)
# =============================================================================
def make_grid(frame_data, out_path):
    """
    frame_data: list of (raw_rgb, heatmap, action, score, frame_idx)
    2행 N/2열 그리드로 논문용 요약 이미지 생성
    """
    n   = len(frame_data)
    cols = (n + 1) // 2
    rows = 2

    fig, axes = plt.subplots(rows, cols * 2,
                              figsize=(cols * 5, rows * 3.2),
                              facecolor='#0D0D0D')
    axes = axes.flatten()

    for idx, (raw_rgb, heatmap, action, score, frame_idx) in enumerate(frame_data):
        H, W = raw_rgb.shape[:2]
        heatmap_r = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

        # 원본
        ax_orig = axes[idx * 2]
        ax_orig.imshow(raw_rgb)
        ax_orig.set_title(f"F#{frame_idx}  Score:{int(score)}",
                           color='white', fontsize=7, pad=2)
        ax_orig.axis('off')

        # Grad-CAM overlay
        ax_cam = axes[idx * 2 + 1]
        ax_cam.imshow(raw_rgb)
        ax_cam.imshow(heatmap_r, cmap='jet', alpha=0.55, vmin=0, vmax=1,
                      extent=[0, W, H, 0])
        action_name = ACTION_NAMES.get(action, str(action))
        ax_cam.set_title(f"→ {action_name}", color='#FFD700', fontsize=7, pad=2)
        ax_cam.axis('off')

    # 남는 축 숨기기
    for i in range(len(frame_data) * 2, len(axes)):
        axes[i].axis('off')

    fig.suptitle("D3QN Grad-CAM  |  Advantage Stream  |  ALE/SpaceInvaders-v5",
                 color='white', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0D0D0D', edgecolor='none')
    plt.close(fig)
    print(f"그리드 저장: {out_path}")


# =============================================================================
# 메인 실행
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--frames",  default=DEFAULT_FRAMES,  type=int,
                        help="캡처할 프레임 수")
    parser.add_argument("--every",   default=CAPTURE_EVERY,   type=int,
                        help="몇 스텝마다 캡처")
    parser.add_argument("--out",     default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    net, input_shape, n_actions = load_model(args.model, device)
    grad_cam = AdvantageGradCAM(net, device)

    # 환경 생성 (rgb_array 모드)
    config = Config()
    config.REWARD_CLIP           = False
    config.TERMINAL_ON_LIFE_LOSS = False

    env = gym.make(
        config.ENV_NAME,
        frameskip=1,  # AtariWrapper가 내부에서 FRAME_SKIP 처리
        repeat_action_probability=0.0,
        render_mode="rgb_array",
    )
    from D3QN_v3 import AtariWrapper
    env = AtariWrapper(env, config)

    state, _ = env.reset()
    score     = 0.0
    step      = 0
    captured  = 0
    frame_data = []

    print(f"\n플레이 시작... {args.frames}프레임 캡처 목표 (매 {args.every}스텝)")

    while captured < args.frames:
        # Grad-CAM 계산
        heatmap, action, advantage, q_values = grad_cam(state)

        # 환경 스텝
        next_state, reward, terminated, truncated, _ = env.step(action)
        score += reward
        step  += 1

        # 지정 주기마다 캡처
        if step % args.every == 0:
            raw_rgb = env.render()
            out_path = os.path.join(args.out, f"frame_{step:05d}.png")
            visualize_frame(raw_rgb, state, heatmap,
                            action, advantage, q_values,
                            step, score, out_path)
            frame_data.append((raw_rgb, heatmap, action, score, step))
            captured += 1
            print(f"  [{captured}/{args.frames}] step={step}  score={int(score)}  "
                  f"action={ACTION_NAMES.get(action, action)}")

        state = next_state
        if terminated or truncated:
            print(f"  에피소드 종료: score={int(score)}, step={step}")
            state, _ = env.reset()
            score = 0.0

    env.close()
    grad_cam.remove_hooks()

    # 그리드 이미지 생성
    grid_path = os.path.join(args.out, "gradcam_grid.png")
    make_grid(frame_data, grid_path)

    print(f"\n완료! 개별 프레임: {args.out}/frame_*.png")
    print(f"논문용 그리드:   {grid_path}")


if __name__ == "__main__":
    main()