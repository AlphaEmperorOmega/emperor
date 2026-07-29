from emperor.datasets.rl._base import GymEnvironment


class Acrobot(GymEnvironment):
    """Acrobot-v1: swing a two-link chain up to a target height.

    Observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot] — 6 floats
    Actions:     0 = torque -1, 1 = torque 0, 2 = torque +1 (on joint 2)
    Reward:      -1 per step until goal is reached
    Episode end: free end reaches target height or 500 steps elapsed
    """

    env_id: str = "Acrobot-v1"
    observation_dim: int = 6
    num_actions: int = 3
    num_classes: int = num_actions
    flattened_input_dim: int = observation_dim
