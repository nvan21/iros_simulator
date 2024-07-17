import torch
import numpy as np
import os
import pickle
from plant import DoublePendulumPlant
from simulation import Simulator
from algo import ControllerSHAC
from hyperparameters import DoublePendulumConfig
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.utils.wrap_angles import wrap_angles_diff
from double_pendulum.model.plant import DoublePendulumPlant as DPP
from double_pendulum.simulation.simulation import Simulator as Sim


def check_if_state_in_roa(S, rho, x):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < rho, rad


device = torch.device("cuda")
params = DoublePendulumConfig()

# model parameters
design = "design_C.1"
model = "model_1.1"
robot = "acrobot"

if robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0

model_par_path = (
    "../../../../../data/system_identification/identified_parameters/"
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

lqr_par_path = f"../../../../../data/controller_parameters/{design}/{model}/{robot}/lqr"

num_envs = 1
plant = DoublePendulumPlant(model_params=mpar, num_envs=num_envs, device=device)
accurate_plant = DPP(model_pars=mpar)
sim = Simulator(plant=plant, num_envs=num_envs, device=device, dt=0.002)
accurate_sim = Sim(accurate_plant)
sim.set_reward_parameters(lqr_load_path=lqr_par_path)
controller = ControllerSHAC(params=params, envs=sim, train=True)
controller.load(filename="./weights/2024-07-16_14-53-42/best_policy.pt")


def sim_trajectory_comparison():
    """
    Compares state-action trajectory from the double_pendulum simulation.py
    and my simulation.py to make sure that they produce the same states given an action
    """
    with open("shac_trajectory_check.pkl", "rb") as f:
        variables = pickle.load(f)

    u_vals = variables["u"]
    x_vals = variables["x"]

    for i in range(320):
        accurate_sim.step(tau=u_vals[i].squeeze().cpu().detach().numpy(), dt=0.01)
        print(f"accurate sim: {accurate_sim.x}")
        print(f"my sim: {x_vals[i]} \n")


def reward_comparison():
    """
    Compares reward trajectory from train_sac.py and my simulation.py to make
    sure that they produce the same reward
    """

    with open("reward_debug_variables.pkl", "rb") as f:
        variables = pickle.load(f)

    rewards = variables["rew"]
    observations = variables["obs"]
    actions = variables["actions"]

    for reward, observation, action in zip(rewards, observations, actions):
        observation, action = (
            torch.tensor(observation, device=device),
            torch.tensor(action, device=device),
        )
        action = torch.cat((torch.tensor([0.0], device=device), action))
        my_reward = sim.get_reward(observation, action)

        print(f"real reward: {reward}")
        print(f"my reward: {my_reward.item()} \n")


def critic_test():
    """
    Takes the states and gives the estimated values to see if the critic
    is learning correctly
    """
    good_state = torch.tensor(
        (torch.pi, 0.0, 0.0, 0.0), dtype=torch.float32, device=device
    )
    bottom_state = torch.tensor(
        (0.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=device
    )
    bad_state = torch.tensor((0.0, 0.0, 10.0, 10.0), dtype=torch.float32, device=device)

    print(f"Good state value: {controller.critic(good_state)}")
    print(f"Bottom state value: {controller.critic(bottom_state)}")
    print(f"Bad state value: {controller.critic(bad_state)}")


def reward_test():
    state = sim._normalize_states(
        torch.tensor((torch.pi, 0.0, 0.0, 0.0), device=device)
    )
    action = torch.tensor((1.0, 0.0), device=device)

    print(sim.get_reward(state, action))


critic_test()
# directory = "/work/flemingc/nvan21/projects/iros_simulator/src/python/double_pendulum/controller/SHAC/weights"
# subdirs = [
#     d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
# ]
# print(subdirs)
