from double_pendulum.controller.abstract_controller import AbstractController
import iql_utils
from iql_utils import TrainConfig as TrainConfig
import torch
import numpy as np


class IQLController(AbstractController):
    def __init__(self, max_action: float = 4.99):
        super().__init__()

        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        self.model = iql_utils.GaussianPolicy(
            4,
            2,
            max_action=float(4.99),
        )
        iql_all_models = torch.load(
            "/work/flemingc/nvan21/projects/iros_simulator/IQL/iql_double-pendulum-sac-v2_500000_final.pt",
            map_location=self.device,
        )
        self.model.load_state_dict(iql_all_models["state_dict"]["actor"])
        self.state_mean = iql_all_models["state_mean"]
        self.state_std = iql_all_models["state_std"]
        self.model.to(self.device)
        self.model.eval()

    def get_control_output_(self, x, t=None):
        norm_obs = iql_utils.normalize_states(x, self.state_mean, self.state_std)
        action = self.model.act(norm_obs, device=self.device)
        print(x)
        return action
