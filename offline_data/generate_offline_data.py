import os
import numpy as np
import h5py

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.wrap_angles import wrap_angles_diff
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)

"""
This testing script is purely for testing the behaviour of SAC controller in swing-up task.
"""

# TODO: The acrobot SAC policy expects scaling for whatever reason, so implement that when the time comes


# hyperparameters
robot = "pendubot"
# robot = "acrobot"

# model and reward parameter
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0

    design = "design_C.1"
    model = "model_1.0"
    scaling_state = False

    load_path = "../data/controller_parameters/design_C.1/model_1.1/pendubot/lqr/"
    model_path = "../data/policies/design_C.1/model_1.0/pendubot/SAC/best_model"
    warm_start_path = ""

    # define para for quadratic reward
    Q = np.zeros((4, 4))
    Q[0, 0] = 100.0
    Q[1, 1] = 100.0
    Q[2, 2] = 1.0
    Q[3, 3] = 1.0
    R = np.array([[0.01]])
    r_line = 1e3
    r_vel = 0
    r_lqr = 1e5

elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1

    design = "design_C.1"
    model = "model_1.0"
    scaling_state = True

    load_path = "../data/controller_parameters/design_C.1/model_1.1/acrobot/lqr/"
    model_path = "../data/policies/design_C.1/model_1.0/pendubot/SAC/sac_model"
    warm_start_path = ""

    # define para for quadratic reward
    Q = np.zeros((4, 4))
    Q[0, 0] = 100.0
    Q[1, 1] = 105.0
    Q[2, 2] = 1.0
    Q[3, 3] = 1.0
    R = np.array([[0.01]])
    r_line = 1e3
    r_vel = 1e4
    r_lqr = 1e5

# import model parameter
model_par_path = (
    "../data/system_identification/identified_parameters/"
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

# simulation parameters
dt = 0.002
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
    scaling=scaling_state,
)

max_velocity = 20.0

# import lqr parameters
rho = np.loadtxt(os.path.join(load_path, "rho"))
vol = np.loadtxt(os.path.join(load_path, "vol"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))

# initialize sac controller
controller = SACController(
    model_path=model_path, dynamics_func=dynamics_func, dt=dt, scaling=scaling_state
)
controller.init()


# reward functions
def check_if_state_in_roa(S, rho, x):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < rho, rad


def reward_func(observation, action):
    # define reward para according to robot type
    control_line = 0.4
    v_thresh = 8.0
    # v_thresh = 10.0
    vflag = False
    flag = False
    bonus = False

    # state
    s = observation
    # s = np.array(
    #     [
    #         observation[0] * np.pi + np.pi,  # [0, 2pi]
    #         (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
    #         observation[2] * max_velocity,
    #         observation[3] * max_velocity,
    #     ]
    # )

    u = 5.0 * action

    goal = [np.pi, 0.0, 0.0, 0.0]

    y = wrap_angles_diff(s)

    # criterion 1: control line
    p1 = y[0]
    p2 = y[1]
    ee1_pos_x = 0.2 * np.sin(p1)
    ee1_pos_y = -0.2 * np.cos(p1)

    ee2_pos_x = ee1_pos_x + 0.3 * np.sin(p1 + p2)
    ee2_pos_y = ee1_pos_y - 0.3 * np.cos(p1 + p2)

    if ee2_pos_y >= control_line:
        flag = True
    else:
        flag = False

    # criteria 2: roa check
    bonus, rad = check_if_state_in_roa(S, rho, y)

    # criteria 3: velocity check
    if flag and (np.abs(y[2]) > v_thresh or np.abs(y[3]) > v_thresh):
        vflag = True

    # reward calculation
    ## stage1: quadratic reward
    r = np.einsum("i, ij, j", s - goal, Q, s - goal) + np.einsum("i, ij, j", u, R, u)
    reward = -1.0 * r

    ## stage2: control line reward
    if flag:
        reward += r_line
        ## stage 3: roa reward
        if bonus:
            # roa method
            reward += r_lqr
        ## penalize on high velocity
        if vflag:
            reward -= r_vel
    else:
        reward = reward

    return reward


# offline data array
num_steps = 500000
num_sims = int(num_steps / (t_final / dt))
states_buf = []
actions_buf = []
rewards_buf = []
next_states_buf = []
dones_buf = []

for i in range(num_sims):
    rewards = []
    x0 = np.random.normal(loc=0, scale=0.05, size=4)
    T, X, U = sim.simulate(
        t0=0.0,
        x0=x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
    )

    # Get the reward for each state-action pair
    for x, u in zip(X, U):
        # Get the single torque value for reward function
        if robot == "pendubot":
            u = u[0]
        elif robot == "acrobot":
            u = u[1]
        u = np.array([u])
        x = np.array(x)

        # Get the reward for the current state-action pair
        rewards.append(reward_func(x, u))
    X = np.array(X)
    states_buf.append(X[:-1])
    actions_buf.append(np.array(U))
    rewards_buf.append(np.array(rewards))
    next_states_buf.append(X[1:])
    dones = np.zeros_like(rewards)
    dones[-1] = 1.0
    dones_buf.append(dones)
    print(f"Simulation {i+1} done")

states = np.concatenate(states_buf, axis=0)
actions = np.concatenate(actions_buf, axis=0)
rewards = np.concatenate(rewards_buf, axis=0)
next_states = np.concatenate(next_states_buf, axis=0)
dones = np.concatenate(dones_buf, axis=0)

with h5py.File("states_trajectory.h5", "w") as hdf:
    hdf.create_dataset("states_trajectory", data=states)

with h5py.File("actions_trajectory.h5", "w") as hdf:
    hdf.create_dataset("actions_trajectory", data=actions)

with h5py.File("rewards_trajectory.h5", "w") as hdf:
    hdf.create_dataset("rewards_trajectory", data=rewards)

with h5py.File("next_states_trajectory.h5", "w") as hdf:
    hdf.create_dataset("next_states_trajectory", data=next_states)

with h5py.File("dones_trajectory.h5", "w") as hdf:
    hdf.create_dataset("dones_trajectory", data=dones)
