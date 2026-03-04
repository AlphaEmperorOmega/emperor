from emperor.datasets.rl.base import GymEnvironment


class LunarLander(GymEnvironment):
    """LunarLander-v2: land a spacecraft between two flags.

    Observation: [x, y, vel_x, vel_y, angle, angular_vel, left_leg, right_leg] — 8 floats
    Actions:     0 = do nothing, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine
    Reward:      shaped reward; +100–140 for landing, -100 for crashing, -0.3 per engine fire
    Episode end: landing, crash, or out of bounds
    """

    env_id: str = "LunarLander-v2"
    observation_dim: int = 8
    num_actions: int = 4
    num_classes: int = num_actions
    flattened_input_dim: int = observation_dim

    def __init__(self, batch_size: int = 64, num_episodes: int = 500):
        super().__init__(batch_size=batch_size, num_episodes=num_episodes)
