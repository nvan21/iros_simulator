import torch

from typing import TYPE_CHECKING

# Classes for type hinting
if TYPE_CHECKING:
    from double_pendulum.model.model_parameters import model_parameters


class DoublePendulumPlant:
    def __init__(
        self, model_params: "model_parameters", num_envs: float, device: torch.device
    ) -> None:
        # Get the parameters from the predefined model parameters
        self.m = model_params.m
        self.l = model_params.l
        self.com = model_params.r
        self.b = model_params.b
        self.coulomb_fric = model_params.cf
        self.g = model_params.g
        self.I = model_params.I
        self.Ir = model_params.Ir
        self.gr = model_params.gr
        self.torque_limit = model_params.tl

        # Set other parameters
        self.num_envs = num_envs
        self.dof = 2
        self.n_actuators = 2
        self.base = [0, 0]
        self.n_links = 2

        if self.torque_limit[0] == 0:
            self.D = [[0, 0], [0, 1]]
        elif self.torque_limit[1] == 0:
            self.D = [[1, 0], [0, 0]]
        else:
            self.D = [[1, 0], [0, 1]]

    def forward_kinematics(self, pos: torch.Tensor) -> torch.Tensor:
        pass

    def forward_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        vel = x[:, self.dof :]

        M = self.mass_tensor(x)
        C = self.coriolis_tensor(x)
        G = self.gravity_tensor(x)
        F = self.coulomb_tensor(x)

        Minv = torch.inverse(M)
        D = torch.tensor(self.D, dtype=x.dtype, device=x.device).repeat(
            x.size(dim=0), 1, 1
        )
        force = (
            G.unsqueeze(-1)
            + torch.bmm(D, u.unsqueeze(-1))
            - torch.bmm(C, vel.unsqueeze(-1))
        )
        friction = F.unsqueeze(-1)
        accn = torch.bmm(Minv, (force - friction)).squeeze(-1)

        return accn

    def rhs(self, t: float, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Forward dynamics
        accn = self.forward_dynamics(x, u)

        # Next state
        v1 = x[:, 2]
        v2 = x[:, 3]
        a1 = accn[:, 0]
        a2 = accn[:, 1]

        # Transpose the new matrix so that the shape stays the same
        new_x = torch.stack((v1, v2, a1, a2)).T

        return new_x

    def mass_tensor(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[:, : self.dof]

        M00 = (
            self.I[0]
            + self.I[1]
            + self.m[1] * self.l[0] ** 2.0
            + 2.0 * self.l[0] * self.m[1] * self.com[1] * torch.cos(pos[:, 1])
            + self.gr**2.0 * self.Ir
            + self.Ir
        )
        M01 = (
            self.I[1]
            + self.l[0] * self.m[1] * self.com[1] * torch.cos(pos[:, 1])
            - self.gr * self.Ir
        )
        M10 = (
            self.I[1]
            + self.l[0] * self.m[1] * self.com[1] * torch.cos(pos[:, 1])
            - self.gr * self.Ir
        )
        M11 = self.I[1] + self.gr**2 * self.Ir

        M = torch.zeros((x.size(dim=0), 2, 2), device=x.device, dtype=x.dtype)
        M[:, 0, 0] = M00
        M[:, 0, 1] = M01
        M[:, 1, 0] = M10
        M[:, 1, 1] = M11

        return M

    def coriolis_tensor(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[:, : self.dof]
        vel = x[:, self.dof :]

        C00 = (
            -2 * vel[:, 1] * self.l[0] * self.m[1] * self.com[1] * torch.sin(pos[:, 1])
        )
        C01 = -vel[:, 1] * self.l[0] * self.m[1] * self.com[1] * torch.sin(pos[:, 1])
        C10 = vel[:, 0] * self.l[0] * self.m[1] * self.com[1] * torch.sin(pos[:, 1])

        C = torch.zeros((x.size(dim=0), 2, 2), device=x.device, dtype=x.dtype)
        C[:, 0, 0] = C00
        C[:, 0, 1] = C01
        C[:, 1, 0] = C10

        return C

    def gravity_tensor(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[:, : self.dof]

        G0 = -self.g * self.m[0] * self.com[0] * torch.sin(pos[:, 0]) - self.g * self.m[
            1
        ] * (
            self.l[0] * torch.sin(pos[:, 0])
            + self.com[1] * torch.sin(pos[:, 0] + pos[:, 1])
        )
        G1 = -self.g * self.m[1] * self.com[1] * torch.sin(pos[:, 0] + pos[:, 1])

        G = torch.zeros((x.size(dim=0), 2), device=x.device, dtype=x.dtype)
        G[:, 0] = G0
        G[:, 1] = G1

        return G

    def coulomb_tensor(self, x: torch.Tensor) -> torch.Tensor:
        vel = x[:, self.dof :]

        F = torch.einsum(
            "ji,i->ji", vel, torch.tensor(self.b, device=x.device, dtype=x.dtype)
        ) + torch.einsum(
            "ji,i->ji",
            torch.arctan(100 * vel),
            torch.tensor(self.coulomb_fric, device=x.device, dtype=x.dtype),
        )

        return F

    def kinetic_energy(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def total_energy(self, x: torch.Tensor) -> torch.Tensor:
        pass
