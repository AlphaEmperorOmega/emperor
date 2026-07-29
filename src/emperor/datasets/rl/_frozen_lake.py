from emperor.datasets.rl._base import GymEnvironment


class FrozenLake(GymEnvironment):
    """FrozenLake-v1: navigate a slippery 4×4 grid from start to goal.

    Observation: single integer tile index in [0, 15] — one-hot encoded to 16 floats
    Actions:     0 = left, 1 = down, 2 = right, 3 = up
    Reward:      +1 for reaching the goal, 0 otherwise
    Episode end: agent reaches goal or falls into a hole

    The slippery flag (is_slippery=True by default) means actions succeed
    only 1/3 of the time — the classic stochastic exploration challenge.
    """

    env_id: str = "FrozenLake-v1"
    observation_dim: int = 16  # one-hot encoded tile index
    num_actions: int = 4
    num_classes: int = num_actions
    flattened_input_dim: int = observation_dim

    def __init__(
        self,
        batch_size: int = 64,
        num_episodes: int = 1000,
        is_slippery: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_episodes=num_episodes,
            seed=seed,
        )
        self.is_slippery = is_slippery

    def _make_env(self):
        import gymnasium as gym

        return gym.make(self.env_id, is_slippery=self.is_slippery)

    def _encode_state(self, tile_index: int) -> list:
        vec = [0.0] * self.observation_dim
        vec[tile_index] = 1.0
        return vec
