"""
teacher_agent/tools/network.py
────────────────────────────────
Teacher Agent 네트워크.

TeacherActor  : (obs, lives) → intervention_prob ∈ [0, 1]
TeacherCritic : (obs, lives, action) → Q-value (scalar)

State 설계:
  - obs   : 4×84×84 uint8 프레임 스택 → CNN 인코딩
  - lives : 스칼라 정수 → 정규화(/ MAX_LIVES) 후 CNN feature 뒤에 concat

Action 설계:
  - 단일 연속값 ∈ [0, 1]  (Gaussian → sigmoid squash)
  - prob >= THRESHOLD(0.65) → 개입
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def layer_init(layer: nn.Module, bias_const: float = 0.0) -> nn.Module:
    """Kaiming normal 초기화."""
    nn.init.kaiming_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _cnn_encoder(in_channels: int) -> nn.Sequential:
    """Nature DQN CNN 인코더 (공통)."""
    return nn.Sequential(
        layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)), nn.ReLU(),
        layer_init(nn.Conv2d(32,          64, kernel_size=4, stride=2)), nn.ReLU(),
        layer_init(nn.Conv2d(64,          64, kernel_size=3, stride=1)), nn.ReLU(),
        nn.Flatten(),
    )


# ── Teacher Actor ──────────────────────────────────────────────────────────────

class TeacherActor(nn.Module):
    """
    Teacher Actor (Stochastic, sigmoid-squashed Gaussian).

    입력:
        obs   : (B, STACK, H, W) uint8
        lives : (B,) int  — 현재 남은 목숨

    출력 (get_action):
        prob      : (B,) float in [0, 1]   — 개입 확률
        log_prob  : (B,) float             — log π(a|s), change-of-variables 보정
        intervene : (B,) bool              — prob >= threshold
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX =  2

    def __init__(self, input_shape: tuple, max_lives: int = 5):
        super().__init__()
        self.max_lives = max_lives

        # CNN 인코더
        self.conv = _cnn_encoder(input_shape[0])
        with torch.inference_mode():
            flat = self.conv(torch.zeros(1, *input_shape)).shape[1]

        # flat + 1(lives) → 512
        self.fc1        = layer_init(nn.Linear(flat + 1, 512))
        self.fc_mean    = layer_init(nn.Linear(512, 1))
        self.fc_log_std = layer_init(nn.Linear(512, 1))

    def _encode(self, obs: torch.Tensor, lives: torch.Tensor) -> torch.Tensor:
        """CNN 인코딩 후 lives concat → hidden."""
        cnn_feat   = self.conv(obs.float() / 255.0)                    # (B, flat)
        lives_feat = (lives.float() / self.max_lives).unsqueeze(1)     # (B, 1)
        return F.relu(self.fc1(torch.cat([cnn_feat, lives_feat], dim=1)))  # (B, 512)

    def forward(self, obs: torch.Tensor, lives: torch.Tensor):
        """mean, log_std 반환 (샘플링 전 중간 출력)."""
        h       = self._encode(obs, lives)
        mean    = self.fc_mean(h)                                       # (B, 1)
        log_std = self.fc_log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def get_action(self, obs: torch.Tensor, lives: torch.Tensor, threshold: float = 0.65):
        """
        개입 확률 샘플링 (reparameterization trick + sigmoid squash).

        Returns:
            prob      (B,): 개입 확률 ∈ [0, 1]
            log_prob  (B,): log π(a|s)
            intervene (B,): prob >= threshold 여부
        """
        mean, log_std = self.forward(obs, lives)
        std    = log_std.exp()
        dist   = Normal(mean, std)
        z      = dist.rsample()                     # reparameterization: z ~ N(μ, σ²)
        prob   = torch.sigmoid(z)                   # (B, 1) ∈ [0, 1]

        # log π(a|s) = log N(z; μ, σ²) - log|dσ(z)/dz|
        #            = log N(z; μ, σ²) - log(prob * (1 - prob))
        log_prob = (
            dist.log_prob(z) - torch.log(prob * (1.0 - prob) + 1e-6)
        ).squeeze(-1)                               # (B,)

        return (
            prob.squeeze(-1),                       # (B,)
            log_prob,                               # (B,)
            prob.squeeze(-1) >= threshold,          # (B,) bool
        )


# ── Teacher Critic ─────────────────────────────────────────────────────────────

class TeacherCritic(nn.Module):
    """
    Teacher Soft Q-Network.

    입력:
        obs    : (B, STACK, H, W) uint8
        lives  : (B,) int
        action : (B,) float in [0, 1]  — 개입 확률

    출력:
        Q-value : (B, 1) scalar
    """

    def __init__(self, input_shape: tuple, max_lives: int = 5):
        super().__init__()
        self.max_lives = max_lives

        self.conv = _cnn_encoder(input_shape[0])
        with torch.inference_mode():
            flat = self.conv(torch.zeros(1, *input_shape)).shape[1]

        # flat + 1(lives) + 1(action) → 512 → 1
        self.fc1  = layer_init(nn.Linear(flat + 2, 512))
        self.fc_q = layer_init(nn.Linear(512, 1))

    def forward(
        self,
        obs:    torch.Tensor,
        lives:  torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        cnn_feat   = self.conv(obs.float() / 255.0)                    # (B, flat)
        lives_feat = (lives.float() / self.max_lives).unsqueeze(1)     # (B, 1)
        act_feat   = action.unsqueeze(1) if action.dim() == 1 else action  # (B, 1)
        x = torch.cat([cnn_feat, lives_feat, act_feat], dim=1)         # (B, flat+2)
        x = F.relu(self.fc1(x))
        return self.fc_q(x)                                            # (B, 1)
