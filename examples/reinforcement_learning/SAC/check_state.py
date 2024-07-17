import os
import numpy as np
import torch

load_path = "../../../data/controller_parameters/design_C.1/model_1.1/acrobot/lqr/"
# load_path = "lqr_data/pendubot/lqr/roa"

rho = np.loadtxt(os.path.join(load_path, "rho"))
vol = np.loadtxt(os.path.join(load_path, "vol"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))

print("rho: ", rho)
print("volume: ", vol)
print("S", S)


def check_if_state_in_roa(S, rho, x):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < rho, rad


def check_if_state_in_roa_torch(S, rho, x: torch.Tensor):
    xdiff = x - torch.tensor([torch.pi, 0.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(
        0
    )
    rad = torch.bmm(xdiff.unsqueeze(1), S).bmm(xdiff.unsqueeze(2)).squeeze()
    return rad < rho, rad


x1 = [0, 0, 0, 0]
x2 = [np.pi, 0, 0, 0]
x3 = [np.pi - 0.1, 0.1, 0, 0]
x4 = [np.pi, 0, 1.0, -1.0]
x5 = [np.pi, 0, 0.1, -0.1]

print(f"x={x1}: ", check_if_state_in_roa(S, rho, x1))
print(f"x={x2}: ", check_if_state_in_roa(S, rho, x2))
print(f"x={x3}: ", check_if_state_in_roa(S, rho, x3))
print(f"x={x4}: ", check_if_state_in_roa(S, rho, x4))
print(f"x={x5}: ", check_if_state_in_roa(S, rho, x5))

rho = torch.tensor(rho)
vol = torch.tensor(vol)
S = torch.tensor(S, dtype=torch.float32).unsqueeze(0)

x1 = torch.tensor(x1).unsqueeze(0)
x2 = torch.tensor(x2).unsqueeze(0)
x3 = torch.tensor(x3).unsqueeze(0)
x4 = torch.tensor(x4).unsqueeze(0)
x5 = torch.tensor(x5).unsqueeze(0)

print(f"x={x1}: ", check_if_state_in_roa_torch(S, rho, x1))
print(f"x={x2}: ", check_if_state_in_roa_torch(S, rho, x2))
print(f"x={x3}: ", check_if_state_in_roa_torch(S, rho, x3))
print(f"x={x4}: ", check_if_state_in_roa_torch(S, rho, x4))
print(f"x={x5}: ", check_if_state_in_roa_torch(S, rho, x5))
