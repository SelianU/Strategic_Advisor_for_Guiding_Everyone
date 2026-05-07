"""
gradcam.py (games.atari.gradcam)

DuelingDQN 의 Advantage stream 에 Grad-CAM 을 적용한다.

원본 atari_base.py 의 AtariGradCAM 클래스 + encode_frame_with_gradcam 헬퍼를
그대로 옮긴 모듈. 게임 본체 로직(base.py)과 분리해 둔다.
"""
import base64

import cv2
import numpy as np


class AtariGradCAM:
    """
    DuelingDQN의 Advantage stream에 Grad-CAM을 적용합니다.

    지원 조건: net에 features / grad_rescale / value_stream / advantage_stream 속성 존재.
    torch.compile로 래핑된 경우 _orig_mod를 통해 내부 모델에 접근합니다.

    사용법:
        gcam = AtariGradCAM(net, device)
        heatmap, action, q_values = gcam(stacked_uint8)  # (4,84,84) uint8
        gcam.remove()   # 훅 해제 (반드시 호출)

    반환:
        heatmap  : (84,84) float32 [0,1]
        action   : int (argmax Q-value)
        q_values : np.ndarray (n_actions,)
    """

    def __init__(self, net, device: str):
        import torch.nn as nn
        self.net    = net
        self.device = device
        self._grads = None
        self._feats = None
        self._hooks: list = []
        self._enabled = False

        # torch.compile 래핑 벗기기
        inner = getattr(net, '_orig_mod', net)
        if not all(hasattr(inner, a) for a in ('features', 'grad_rescale',
                                                'value_stream', 'advantage_stream')):
            return  # 지원 불가 모델 → _enabled=False

        self._inner = inner
        last_conv = None
        for m in inner.features.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m

        if last_conv is None:
            return

        def fwd(module, inp, out):
            self._feats = out.detach()

        def bwd(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        self._hooks.append(last_conv.register_forward_hook(fwd))
        self._hooks.append(last_conv.register_full_backward_hook(bwd))
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, state_uint8: np.ndarray):
        """
        state_uint8: (4, 84, 84) uint8
        returns: (heatmap float32 84×84, chosen_action int, q_values ndarray)
        """
        import torch
        import torch.nn.functional as F

        if not self._enabled:
            with torch.no_grad():
                s = torch.from_numpy(state_uint8.astype(np.uint8)).unsqueeze(0).to(self.device)
                q = self.net(s).squeeze(0).cpu().numpy()
            return np.zeros((84, 84), dtype=np.float32), int(np.argmax(q)), q

        inner = self._inner
        inner.zero_grad()

        s = torch.from_numpy(state_uint8.astype(np.float32)).unsqueeze(0).to(self.device)

        # forward — advantage stream 기준
        x        = s.contiguous().float().mul_(1.0 / 255.0)
        features = inner.features(x).flatten(1)
        features = inner.grad_rescale(features)
        value     = inner.value_stream(features)
        advantage = inner.advantage_stream(features)
        q_values  = (value + advantage - advantage.mean(dim=1, keepdim=True))[0]

        chosen = int(q_values.argmax().item())

        # backward on chosen action's advantage
        advantage[0, chosen].backward()

        if self._grads is None or self._feats is None:
            return np.zeros((84, 84), dtype=np.float32), chosen, q_values.detach().cpu().numpy()

        grads   = self._grads[0]   # (C, H, W)
        feats   = self._feats[0]   # (C, H, W)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam     = F.relu((weights * feats).sum(dim=0)).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        heatmap = cv2.resize(cam.astype(np.float32), (84, 84),
                             interpolation=cv2.INTER_LINEAR)

        return heatmap, chosen, q_values.detach().cpu().numpy()


def encode_frame_with_gradcam(
    frame_rgb: np.ndarray,
    heatmap_84: np.ndarray,
    alpha: float = 0.50,
) -> str:
    """
    frame_rgb  : (H, W, 3) uint8 RGB 원본 프레임
    heatmap_84 : (84, 84) float32 [0,1] Grad-CAM heatmap
    alpha      : heatmap 투명도 (기본 0.50)
    반환       : base64 JPEG 문자열
    """
    H, W = frame_rgb.shape[:2]
    hm_scaled = (heatmap_84 * 255).astype(np.uint8)
    hm_color  = cv2.applyColorMap(
        cv2.resize(hm_scaled, (W, H), interpolation=cv2.INTER_LINEAR),
        cv2.COLORMAP_JET,
    )  # BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay   = cv2.addWeighted(frame_bgr, 1.0 - alpha, hm_color, alpha, 0)
    _, buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')