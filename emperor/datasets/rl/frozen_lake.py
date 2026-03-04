import torch
import numpy as np

from emperor.datasets.rl.base import GymEnvironment, _TransitionDataset


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

    def __init__(self, batch_size: int = 64, num_episodes: int = 1000, is_slippery: bool = True):
        super().__init__(batch_size=batch_size, num_episodes=num_episodes)
        self.is_slippery = is_slippery

    def _make_env(self):
        import gymnasium as gym
        return gym.make(self.env_id, is_slippery=self.is_slippery)

    def _setup_fit(self) -> None:
        self.env = self._make_env()
        self.train = self._collect_transitions(self.num_episodes)
        val_episodes = max(1, self.num_episodes // 5)
        self.val = self._collect_transitions(val_episodes)

    def _setup_validate(self) -> None:
        self.env = self._make_env()
        val_episodes = max(1, self.num_episodes // 5)
        self.val = self._collect_transitions(val_episodes)

    def _collect_transitions(self, num_episodes: int) -> _TransitionDataset:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                states.append(self._one_hot(state))
                actions.append(action)
                rewards.append(reward)
                next_states.append(self._one_hot(next_state))
                dones.append(float(done))
                state = next_state
        return _TransitionDataset(
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def _one_hot(self, tile_index: int) -> list:
        vec = [0.0] * self.observation_dim
        vec[tile_index] = 1.0
        return vec
