import torch
import torch.nn as nn
from torch.distributions import Normal


class StochasticActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        units: list,
        device: torch.device,
        activation_fn,
    ):
        super(StochasticActor, self).__init__()

        layers = [obs_dim] + units + [act_dim]
        modules = []

        for in_dim, out_dim in zip(layers, layers[1:]):
            modules.append(nn.Linear(in_dim, out_dim, dtype=torch.float32))
            modules.append(activation_fn)
            modules.append(nn.LayerNorm(out_dim, dtype=torch.float32))

        self.mu_network = nn.Sequential(*modules[:-2]).to(device)
        self.logstd_layer = nn.Parameter(torch.ones(act_dim, device=device) * -1).to(
            device
        )

        # The learning rate will be updated every backwards pass, so there's no need to set it here
        self.optimizer = torch.optim.Adam(self.parameters())

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Get the predicted mean and standard deviation of the state
        mu = self.mu_network(obs)
        std = self.logstd_layer.exp()

        # Sample actions from the corresponding normal distribution, and then apply tanh squashing function
        dist = Normal(mu, std)
        action = dist.rsample()
        action = torch.tanh(action)

        return action

    def backward(self, loss: torch.Tensor, learning_rate: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        with torch.autograd.set_detect_anomaly(True):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
