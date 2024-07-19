import numpy as np
import os
from datetime import datetime
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt

from hyperparameters import DoublePendulumConfig
from algo import ControllerSHAC
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.utils.plotting import plot_timeseries

USE_ACTOR_PRIOR = True


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("-n", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    # Get config and CLI args
    params = DoublePendulumConfig()
    args = get_args()

    # model parameters
    design = "design_C.1"
    model = "model_1.1"
    robot = "pendubot"

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

    lqr_par_path = (
        f"../../../../../data/controller_parameters/{design}/{model}/{robot}/lqr"
    )

    # Initialize environments and SHAC instance
    plant = DoublePendulumPlant(model_pars=mpar)
    envs = Simulator(plant=plant)
    controller = ControllerSHAC(params=params, envs=envs, train=False)

    # simulation parameters
    x0 = [0.0, 0.0, 0.0, 0.0]
    dt = 0.002
    t_final = 10.0
    integrator = "runge_kutta"
    goal = [np.pi, 0.0, 0.0, 0.0]
    save_dir = os.path.join("data", design, model, robot, "shac")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if USE_ACTOR_PRIOR:
        controller.prior_actor = True
        controller.create_models(1, 4)
    else:
        controller.load(filename="./weights/2024-07-18_15-11-42/best_policy.pt")
    T, X, U = envs.simulate_and_animate(
        t0=0.0,
        x0=x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
        save_video=True,
        video_name=os.path.join(save_dir, f"{args.n}.mp4"),
    )

    # times = np.linspace(0, 4999, 5000)
    # actions = np.stack(controller.actions)[:, 0]

    # plt.figure()
    # plt.plot(times, actions, label="torques")
    # plt.xlabel("time")
    # plt.ylabel("torque")
    # plt.title(f"Torque of {args.n} run")
    # plt.legend()
    # plt.savefig(f"{args.n}.png")
