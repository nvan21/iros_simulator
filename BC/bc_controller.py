from double_pendulum.controller.abstract_controller import AbstractController
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BCController(AbstractController):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim=4, action_dim=2, max_action=5.0).to(self.device)
        self.actor.load_state_dict(
            torch.load("BC-double-pendulum-sac-v2-22a2afd8_100_1000.pt")["actor"]
        )
        self.obs_mean = np.array(
            [3.10138527e00, -1.25685267e-01, 3.17212890e-01, -2.20910353e-03]
        )
        self.obs_std = np.array([0.43561072, 0.38527233, 1.89671584, 2.06232309])
        self.obs_mean_torch = torch.tensor(
            self.obs_mean, device=self.device, dtype=torch.float32
        )
        self.obs_std_torch = torch.tensor(
            self.obs_std, device=self.device, dtype=torch.float32
        )

    def get_control_output_(self, x, t=None):
        return self.actor.act((x - self.obs_mean) / self.obs_std, device=self.device)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    def forward_with_noise(self, state: torch.Tensor) -> torch.Tensor:
        action = self.net(state)
        noise = torch.randn_like(action) * 0.2
        return self.max_action * action + noise

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()
