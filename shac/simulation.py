import torch

from typing import TYPE_CHECKING, Tuple
import time

# Imports just for type checking
if TYPE_CHECKING:
    from plant import DoublePendulumPlant


class Simulator:
    def __init__(
        self, plant: "DoublePendulumPlant", num_envs: int, device: torch.device
    ) -> None:
        self.plant = plant
        self.num_envs = num_envs
        self.device = device

        self.x = torch.zeros(
            (num_envs, 4), dtype=torch.float32, device=device
        )  # position, velocity
        self.t = 0.0  # time

        self.reset()

    def set_measurement_parameters(
        self,
        C=torch.eye(4),
        D=torch.zeros(4, 2),
        meas_noise_sigmas=torch.zeros(4),
        delay=0.0,
        delay_mode="None",
    ) -> None:
        self.meas_C = C.to(self.device)
        self.meas_D = D.to(self.device)
        self.meas_noise_sigmas = meas_noise_sigmas.to(self.device)
        self.delay = delay
        self.delay_mode = delay_mode

    def set_motor_parameters(
        self, u_noise_sigmas=torch.zeros(2), u_responsiveness=1.0
    ) -> None:
        self.u_noise_sigmas = u_noise_sigmas.to(self.device)
        self.u_responsiveness = u_responsiveness

    def set_process_noise(
        self, process_noise_sigmas=torch.zeros(4, dtype=torch.float32)
    ):
        self.process_noise_sigmas = process_noise_sigmas.to(self.device)

    def set_state(self, t: float, x: torch.Tensor) -> None:
        self.t = t
        self.x = x.clone()

    def reset(self):
        self.set_measurement_parameters()
        self.set_motor_parameters()
        self.set_process_noise()

        self.t_values = []
        self.x_values = []
        self.u_values = []

        self.meas_x_values = []
        self.con_u_values = []

    def record_data(self, t: float, x: torch.Tensor, u: torch.Tensor = None) -> None:
        self.t_values.append(t)
        self.x_values.append(x)
        if u is not None:
            self.u_values.append(u)

    def runge_integrator(
        self, x: torch.Tensor, u: torch.Tensor, dt: float, t: float
    ) -> torch.Tensor:
        k1 = self.plant.rhs(t, x, u)
        k2 = self.plant.rhs(t + 0.5 * dt, x + 0.5 * dt * k1, u)
        k3 = self.plant.rhs(t + 0.5 * dt, x + 0.5 * dt * k2, u)
        k4 = self.plant.rhs(t + dt, x + dt * k3, u)

        return (k1 + 2.0 * (k2 + k3) + k4) / 6.0

    def get_measurement(self, dt: float) -> torch.Tensor:
        x_meas = self.x.clone()

        # If the delay is larger than the timestep, then there will be measurement delay
        n_delay = int(self.delay / dt) + 1
        if n_delay > 1:
            len_x = len(self.x_values)

            # Get the last internal state possible according to the measurement delay - Cx(t-delay)
            if self.delay_mode == "posvel":
                x_meas = torch.clone(self.x_values[max(-n_delay, -len_x)])
            elif self.delay_mode == "vel":
                x_meas[:, 2:] = self.x_values[max(-n_delay, -len_x)][:, 2:]

        # Get the last torque value - Du(t-delay)
        if len(self.u_values) > n_delay:
            u = self.u_values[-n_delay]
        else:
            u = torch.zeros(
                (self.num_envs, self.plant.n_actuators),
                dtype=x_meas.dtype,
                device=x_meas.device,
            )

        Cx = (self.meas_C @ x_meas.T).T
        Du = (self.meas_D @ u.T).T
        noise = torch.randn_like(self.x) * self.meas_noise_sigmas
        x_meas = Cx + Du + noise

        self.meas_x_values.append(x_meas.clone())

        return x_meas

    def get_control_u(
        self, controller, x: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        realtime = True
        if controller is not None:
            t0 = time.time()
            u = controller.get_control_output(x=x, t=t)
            if time.time() - t0 > dt:
                realtime = False
        else:
            u = torch.zeros((self.num_envs, 1), device=self.device)

        self.con_u_values.append(u.clone())

        return u, realtime

    def get_real_applied_u(self, u: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        nu = u.clone()

        # motor responsiveness
        if len(self.u_values) > 0:
            last_u = self.u_values[-1]
        else:
            last_u = torch.zeros((self.num_envs, 1), device=self.device)
        nu = last_u + self.u_responsiveness * (nu - last_u)

        # change torque shape from (num_envs, 1) to (num_envs, 2)
        if self.plant.torque_limit[0] == 0 and nu.shape[1] == 1:
            nu = torch.cat([torch.zeros_like(nu), nu], dim=1)
        elif self.plant.torque_limit[1] == 0 and nu.shape[1] == 1:
            nu = torch.cat([nu, torch.zeros_like(nu)], dim=1)

        # torque noise
        noise = torch.randn_like(u) * self.u_noise_sigmas
        nu = nu + noise

        # clip torques
        nu[:, 0] = torch.clip(
            nu[:, 0], -self.plant.torque_limit[0], self.plant.torque_limit[0]
        )
        nu[:, 1] = torch.clip(
            nu[:, 1], -self.plant.torque_limit[1], self.plant.torque_limit[1]
        )

        # TODO: add perturbance for joints

        return nu

    def step(self, u: torch.Tensor, dt: float) -> None:
        # Add new simulator step
        self.x = torch.add(self.x, dt * self.runge_integrator(self.x, u, dt, self.t))

        # Process noise
        noise = torch.randn_like(self.x) * self.process_noise_sigmas
        self.x = torch.add(self.x, noise)

        self.t += dt
        self.record_data(self.t, self.x.clone(), u.clone())

    def controller_step(self, dt: float, controller=None) -> bool:
        x_meas = self.get_measurement(dt)
        u, realtime = self.get_control_u(controller, x_meas, self.t, dt)
        nu = self.get_real_applied_u(u, self.t, dt)

        self.step(nu, dt)

        return realtime

    def simulate(
        self, t0: float, x0: torch.Tensor, tf: float, dt: float, controller=None
    ) -> Tuple[list, list, list]:
        self.set_state(t0, x0)
        self.reset()
        self.record_data(t0, x0.clone(), None)

        total_steps = 0
        while self.t < tf:
            _ = self.controller_step(dt, controller)
            total_steps += self.num_envs

        return self.x_values, self.t_values, self.u_values
