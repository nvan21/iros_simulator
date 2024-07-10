import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        units: list,
        device: torch.device,
        activation_fn,
    ):
        super(Critic, self).__init__()

        layers = [obs_dim] + units + [1]
        modules = []

        for in_dim, out_dim in zip(layers, layers[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(activation_fn)
            modules.append(nn.LayerNorm(out_dim))

        # Get rid of the last two items in the list because the last layer only needs the linear connection
        self.critic = nn.Sequential(*modules[:-2]).to(device)

        # The learning rate will be updated every backwards pass, so there's no need to set it here
        self.optimizer = torch.optim.Adam(self.parameters())

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def backward(self, loss: torch.Tensor, learning_rate: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
