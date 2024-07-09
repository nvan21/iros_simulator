import torch

from typing import TYPE_CHECKING

# Imports just for type checking
if TYPE_CHECKING:
    from plant import DoublePendulumPlant


class Simulator:
    def __init__(self, plant: "DoublePendulumPlant", num_envs: int) -> None:
        self.plant = plant
        self.num_envs = num_envs

        self.x = torch.zeros((num_envs, 4))  # position, velocity
        self.t = 0.0  # time

    def set_measurement_parameters(
        self, meas_noise_sigmas, delay, delay_mode, C=torch.eye(4), D=torch.zeros(4, 2)
    ) -> None:
        self.meas_C = C
        self.meas_D = D
        self.meas_noise_sigmas = meas_noise_sigmas
        self.delay = delay
        self.delay_mode = delay_mode

    def set_motor_parameters(
        self, u_noise_sigmas=[0.0, 0.0], u_responsiveness=1.0
    ) -> None:
        self.u_noise_sigmas = u_noise_sigmas
        self.u_responsiveness = u_responsiveness

    def runge_integrator(
        self, x: torch.Tensor, u: torch.Tensor, dt: float, t: float
    ) -> torch.Tensor:
        k1 = self.plant.rhs(t, x, u)
        k2 = self.plant.rhs(t + 0.5 * dt, x + 0.5 * dt * k1, u)
        k3 = self.plant.rhs(t + 0.5 * dt, x + 0.5 * dt * k2, u)
        k4 = self.plant.rhs(t + dt, x + dt * k3, u)

        return (k1 + 2.0 * (k2 + k3) + k4) / 6.0

    def get_measurement(self) -> torch.Tensor:
        pass

    def get_control_u(self) -> torch.Tensor:
        pass

    def get_real_applied_u(self) -> torch.Tensor:
        pass

    def controller_step(self) -> bool:
        pass

    def simulate(self) -> None:
        pass
