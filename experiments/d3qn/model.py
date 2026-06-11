import torch
import torch.nn as nn


class GradRescale(nn.Module):
    """
    Wang et al. 2016 §4: flatten 직후 두 스트림 입력 전 gradient를 1/√2 로 rescaling.
    forward는 identity; backward hook에서 grad_input을 스케일링.
    """
    SCALE = 1.0 / (2.0 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def backward_hook(self, module, grad_input, grad_output):
        return tuple(g * self.SCALE if g is not None else None for g in grad_input)


class DuelingDQN(nn.Module):
    """Dueling Network Architecture [Wang et al. 2016]. CNN + Value/Advantage streams."""

    def __init__(self, input_shape: tuple, n_actions: int):
        super().__init__()
        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c,  32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        with torch.no_grad():
            flat = self.features(torch.zeros(1, *input_shape)).flatten().shape[0]

        self.grad_rescale = GradRescale()
        self.grad_rescale.register_full_backward_hook(self.grad_rescale.backward_hook)

        self.value_stream = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x         = x.contiguous().float().mul_(1.0 / 255.0)
        features  = self.features(x).flatten(1)
        features  = self.grad_rescale(features)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
