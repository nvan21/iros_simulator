import torch
import numpy as np
import os

from typing import TYPE_CHECKING, Tuple
import time

# Imports just for type checking
if TYPE_CHECKING:
    from plant import DoublePendulumPlant


class Simulator:
    def __init__(
        self,
        plant: "DoublePendulumPlant",
        num_envs: int,
        dt: float,
        device: torch.device,
    ) -> None:
        self.plant = plant
        self.num_envs = num_envs
        self.device = device

        self.x = torch.zeros(
            (num_envs, 4), dtype=torch.float32, device=device
        )  # position, velocity
        self.t = 0.0  # time
        self.dt = dt

        self.step_count = 0
        self.max_time = 10.0
        self.max_timesteps = int(self.max_time / self.dt)
        self.max_velocity = 20.0

        self.dones = torch.zeros(num_envs, device=device)
        self.truncateds = torch.zeros(num_envs, device=device)

        self.observation_space = np.ndarray((4,))
        self.action_space = np.ndarray((1,))

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

    def set_reward_parameters(
        self,
        lqr_load_path: str,
        control_line: float = 0.4,
        r_line: float = 1e3,
        r_vel: float = 1e4,
        r_roa: float = 1e5,
        v_thresh: float = 8.0,
        goal=None,
        Q=None,
        R=None,
    ):
        # Import LQR parameters
        self.rho = torch.tensor(
            np.loadtxt(os.path.join(lqr_load_path, "rho")),
            dtype=torch.float32,
            device=self.device,
        )
        self.vol = torch.tensor(
            np.loadtxt(os.path.join(lqr_load_path, "vol")),
            dtype=torch.float32,
            device=self.device,
        )
        self.S = (
            torch.tensor(
                np.loadtxt(os.path.join(lqr_load_path, "Smatrix")),
                dtype=torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1)
        )
        self.control_line = control_line
        self.r_line = r_line
        self.r_vel = r_vel
        self.r_roa = r_roa
        self.v_thresh = v_thresh
        if goal is None:
            self.goal = torch.zeros(
                (self.num_envs, 4), dtype=torch.float32, device=self.device
            )
            self.goal[:, 0] = torch.pi

        if Q is None:
            self.Q = torch.zeros(
                (self.num_envs, 4, 4), dtype=torch.float32, device=self.device
            )
            self.Q[:, 0, 0] = 100.0
            self.Q[:, 1, 1] = 100.0
            self.Q[:, 2, 2] = 1.0
            self.Q[:, 3, 3] = 1.0

        if R is None:
            self.R = torch.zeros(
                (self.num_envs, 2, 2), dtype=torch.float32, device=self.device
            )
            self.R[:, 0, 0] = 0.01
            self.R[:, 1, 1] = 0.01

    def set_state(self, t: float, x: torch.Tensor) -> None:
        self.t = t
        self.x = x.clone()

    @torch.no_grad()
    def reset_random_state(self) -> None:
        self.t = 0.0
        noise = torch.randn((self.num_envs, 4), device=self.device) * 0.01
        noise[:, 2:] = noise[:, 2:] - 0.05
        self.x = (
            torch.tensor((0.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
            + noise
        )
        # self.x = torch.zeros(
        #     (self.num_envs, 4), dtype=torch.float32, device=self.device
        # )
        # self.x[:, 0] = torch.pi

    def reset(self) -> torch.Tensor:
        self.set_measurement_parameters()
        self.set_motor_parameters()
        self.set_process_noise()
        self.reset_random_state()
        self.reset_data_recorder()

        return self.x.clone()

    def reset_data_recorder(self):
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

    # @torch.no_grad()
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

            # Fix shape to be (num_envs, 2) instead of (num_envs, 1)
            # if self.plant.torque_limit[0] == 0:
            #     u = torch.cat([torch.zeros_like(u), u], dim=1)
            # elif self.plant.torque_limit[1] == 0 and u.shape[1] == 1:
            #     u = torch.cat([u, torch.zeros_like(u)], dim=1)
        else:
            u = torch.zeros((self.num_envs, 2), device=self.device)

        self.con_u_values.append(u.clone())

        return u, realtime

    # @torch.no_grad()
    def get_real_applied_u(self, u: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        nu = u.clone()

        # motor responsiveness
        if len(self.u_values) > 0:
            last_u = self.u_values[-1]
        else:
            last_u = torch.zeros((self.num_envs, 2), device=self.device)
        nu = last_u + self.u_responsiveness * (nu - last_u)

        # torque noise
        # noise = torch.randn_like(nu) * self.u_noise_sigmas
        # nu = nu + noise

        # clip torques
        with torch.no_grad():
            nu0 = torch.clamp(
                nu[:, 0].clone(),
                -self.plant.torque_limit[0],
                self.plant.torque_limit[0],
            )
            nu1 = torch.clamp(
                nu[:, 1].clone(),
                -self.plant.torque_limit[1],
                self.plant.torque_limit[1],
            )

        # TODO: add perturbance for joints

        clamped_nu = torch.stack((nu0, nu1), dim=1)
        # print(
        #     torch.clamp(
        #         nu,
        #         torch.tensor(
        #             [-self.plant.torque_limit[0], -self.plant.torque_limit[1]],
        #             device=self.device,
        #         ),
        #         torch.tensor(
        #             [self.plant.torque_limit[0], self.plant.torque_limit[1]],
        #             device=self.device,
        #         ),
        #     )
        # )
        # print(nu0.unsqueeze(0).shape)
        return clamped_nu

    def get_reward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        There is a base quadratic reward which depends on how close the current state
        is to the goal

        There are three different checks for bonus reward:
        1) If the second pendulum (the one at the end of the configuration) is
        above the control line, then give bonus reward
        2) If the state is in the region of attraction, then give a bonus reward
        3) If the first condition is true and either one of the pendulum velocities
        is above the threshold, then penalize for high velocity
        """
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        if len(u.size()) == 1:
            u = u.unsqueeze(0)

        x[:, 0] = x[:, 0] * torch.pi + torch.pi  # [0, 2pi]
        x[:, 1] = (x[:, 1] * torch.pi + torch.pi + torch.pi) % (
            2 * torch.pi
        ) - torch.pi  # [-pi, pi]
        x[:, 2] = x[:, 2] * self.max_velocity
        x[:, 3] = x[:, 3] * self.max_velocity

        x = self._normalize_angles(x)
        coordinates = self.plant.forward_kinematics(x)
        ee2_pos_y = coordinates[:, 3]
        x_diff = x - self.goal

        # Calculate quadratic reward
        cost_state = (
            torch.bmm(x_diff.unsqueeze(1), self.Q).bmm(x_diff.unsqueeze(2)).squeeze()
        )
        cost_control = torch.bmm(u.unsqueeze(1), self.R).bmm(u.unsqueeze(2)).squeeze()
        reward = -(cost_state + cost_control)

        # Bonus criterion 1: control line
        control_line_mask = ee2_pos_y >= self.control_line
        control_line_reward = control_line_mask * self.r_line

        # Bonus criterion 2: region of attraction check
        rad = torch.bmm(x_diff.unsqueeze(1), self.S).bmm(x_diff.unsqueeze(2)).squeeze()
        roa_reward = (rad < self.rho) * self.r_roa

        # Bonus criterion 3: velocity check
        v1 = torch.abs(x[:, 2]) > self.v_thresh
        v2 = torch.abs(x[:, 3]) > self.v_thresh
        velocity_reward = (control_line_mask & (v1 | v2)) * self.r_vel

        # Sum up all bonus rewards
        reward = reward + control_line_reward + roa_reward - velocity_reward

        return reward  # , rad

    def step(
        self, controller=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.step_count += 1
        info = {}

        # Reset dones and truncateds
        self.dones = torch.zeros_like(self.dones)
        self.truncateds = torch.zeros_like(self.truncateds)

        # Get measured state and query controller for action
        # x_meas = self.get_measurement(self.dt)
        xn = self._normalize_states(self.x.clone())
        u, realtime = self.get_control_u(controller, xn, self.t, self.dt)
        nu = self.get_real_applied_u(u, self.t, self.dt)

        # Add new simulator step
        self.x = torch.add(
            self.x.clone(),
            self.dt * self.runge_integrator(self.x.clone(), nu, self.dt, self.t),
        )

        # Process noise
        # with torch.no_grad():
        #     noise = torch.randn_like(self.x) * self.process_noise_sigmas
        #     self.x = torch.add(self.x, noise)

        # Calculate reward
        # Not having self.x.clone() causes nan errors
        xn = self._normalize_states(self.x.clone())
        reward = self.get_reward(xn, nu)

        # Increment time and record data
        self.t += self.dt
        self.record_data(self.t, self.x.clone(), nu.clone())

        # Get information for Gym API return
        if self.step_count == self.max_timesteps:
            info["final_observation"] = self.x.clone()
            self.truncateds[:] = 1.0
            self.reset_random_state()  # resets self.x to random starting states
            self.reset_data_recorder()
            self.step_count = 0
            self.t = 0.0

        # new_states = self.x.clone()
        new_states = self._normalize_states(self.x.clone())

        return (new_states, reward, self.dones, self.truncateds, info)

    @torch.no_grad()
    def clear_grad(self):
        x = self.x.clone()
        self.x = x.clone()

    @torch.no_grad()
    def _normalize_angles(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes in the state observation and normalizes the angles to [-pi, pi]
        """
        y = x.clone()
        y[:, :2] = y[:, :2] % (2 * torch.pi)
        y[:, :2] = torch.where(y[:, :2] > torch.pi, y[:, :2] - 2 * torch.pi, y[:, :2])

        return y

    # @torch.no_grad()
    def _normalize_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes in the state observation and normalizes it to [-1, 1]
        """
        if len(x.size()) == 1:
            x = x.unsqueeze(0)

        x[:, 0] = (x[:, 0] % (2 * torch.pi) - torch.pi) / torch.pi
        x[:, 1] = (x[:, 1] % (2 * torch.pi) - torch.pi) / torch.pi
        x[:, 2] = (
            torch.clip(x[:, 2], -self.max_velocity, self.max_velocity)
            / self.max_velocity
        )
        x[:, 3] = (
            torch.clip(x[:, 3], -self.max_velocity, self.max_velocity)
            / self.max_velocity
        )

        return x
