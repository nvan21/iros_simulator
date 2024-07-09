import torch
import numpy as np
from plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters

device = torch.device("cuda")

# model parameters
design = "design_A.0"
model = "model_2.0"
robot = "acrobot"

if robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

plant = DoublePendulumPlant(model_params=mpar, num_envs=3, device=device)

state = torch.ones((3, 4), dtype=torch.float32, device=device)
torques = torch.ones((3, 1), dtype=torch.float32, device=device)
print(plant.forward_dynamics(state, torques))

from double_pendulum.model.plant import DoublePendulumPlant

plant = DoublePendulumPlant(model_pars=mpar)
state = np.array([1, 1, 1, 1])
tau = np.array([0, 1])
print(plant.rhs(t=0, state=state, tau=tau))
