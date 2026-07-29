import torch

from emperor.datasets.rl._base import GymEnvironment


class Pendulum(GymEnvironment):
    """Pendulum-v1: swing a pendulum upright and keep it balanced.

    Observation: [cos(θ), sin(θ), θ_dot] — 3 floats
    Actions:     continuous torque in [-2, 2] — 1 float (Box action space)
    Reward:      -(θ² + 0.1·θ_dot² + 0.001·torque²)  (higher = better)
    Episode end: always truncated at 200 steps

    Note: actions are continuous, so num_actions = 0 and num_classes = 0.
    The action is a 1-D float tensor instead of a discrete integer.
    """

    env_id: str = "Pendulum-v1"
    observation_dim: int = 3
    num_actions: int = 0  # continuous action space
    num_classes: int = 0
    flattened_input_dim: int = observation_dim
    _action_dtype = torch.float32
