from hyperparameters import DoublePendulumConfig
from algo import ControllerSHAC
from simulation import Simulator
from plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters

import wandb


if __name__ == "__main__":
    # Get config
    params = DoublePendulumConfig()

    # Create the dictionary of wandb logged parameters
    log_params = {
        k: v
        for k, v in DoublePendulumConfig.__dict__.items()
        if not k.startswith("__")
        and not callable(v)
        and not isinstance(v, staticmethod)
        and not k == "project_name"
    }

    # Initialize wandb logger
    if params.do_wandb_logging:
        wandb.init(project=params.project_name, config=log_params)
        wandb.define_metric("epoch")
        wandb.define_metric("step")
        wandb.define_metric("step/*", step_metric="step")
        wandb.define_metric("epoch/*", step_metric="epoch")

    # model parameters
    design = "design_C.1"
    model = "model_1.1"
    robot = "acrobot"

    if robot == "acrobot":
        torque_limit = [0.0, 5.0]
        active_act = 1
        r_vel = 1e4
    if robot == "pendubot":
        torque_limit = [5.0, 0.0]
        active_act = 0
        r_vel = 0

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
    plant = DoublePendulumPlant(
        model_params=mpar, num_envs=params.num_envs, device=params.device
    )
    envs = Simulator(
        plant=plant, num_envs=params.num_envs, dt=params.dt, device=params.device
    )

    envs.set_reward_parameters(lqr_load_path=lqr_par_path, r_vel=r_vel)
    envs.reset_random_state()

    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]

    shac = ControllerSHAC(params=params, envs=envs)

    shac.create_models(act_dim=act_dim, obs_dim=obs_dim)
    shac.train()
