from torch.nn import ReLU
from torch import device, cuda


class DoublePendulumConfig:
    # Wandb project name
    project_name = "shac-double-pendulum"

    # Whether or not to log run with wandb (useful for debugging)
    do_wandb_logging = True

    # Configurations for the actor neural network
    actor_units = [256, 256]
    actor_activation = ReLU()
    actor_learning_rate = 2e-3
    actor_learning_rate_schedule = "constant"  # can be linear or constant

    # Configurations for the critic neural network
    critic_units = [256, 256]
    critic_activation = ReLU()
    critic_learning_rate = 5e-4
    critic_learning_rate_schedule = "constant"  # can be linear or constant
    critic_minibatches = 4

    # Hyperparameters for training
    gae_lambda = 0.95
    gamma = 0.99
    tau = 0.995
    num_steps = 256  # this is the length of the trajectory (h in the paper)
    num_envs = 64  # this is the number of parallel envs (N in the paper)
    max_epochs = 10000
    critic_iterations = 16
    dt = 0.01

    # Device to use for tensor storage
    device = device("cuda" if cuda.is_available() else "cpu")
