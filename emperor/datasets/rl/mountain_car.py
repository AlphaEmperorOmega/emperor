from emperor.datasets.rl.base import GymEnvironment


class MountainCar(GymEnvironment):
    """MountainCar-v0: drive an underpowered car up a steep hill.

    Observation: [position, velocity] — 2 floats
    Actions:     0 = push left, 1 = no push, 2 = push right
    Reward:      -1 per step (sparse; +0 on reaching the goal)
    Episode end: car reaches the flag or 200 steps elapsed
    """

    env_id: str = "MountainCar-v0"
    observation_dim: int = 2
    num_actions: int = 3
    num_classes: int = num_actions
    flattened_input_dim: int = observation_dim

    def __init__(self, batch_size: int = 64, num_episodes: int = 500):
        super().__init__(batch_size=batch_size, num_episodes=num_episodes)
