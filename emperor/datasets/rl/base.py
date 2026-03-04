import torch
import torch.utils.data
import numpy as np
import gymnasium as gym

from emperor.base.utils import DataModule


class _TransitionDataset(torch.utils.data.Dataset):
    """Dataset of (state, action, reward, next_state, done) transitions."""

    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class GymEnvironment(DataModule):
    """Base class for Gymnasium environments wrapped as a DataModule.

    Each item in the dataset is a transition tuple:
        (state, action, reward, next_state, done)

    The environment is also exposed as `self.env` for online interaction
    during training (e.g. for DQN or policy gradient loops).
    """

    env_id: str = ""
    observation_dim: int = 0
    num_actions: int = 0
    num_classes: int = 0       # = num_actions for discrete, 0 for continuous
    flattened_input_dim: int = 0  # = observation_dim

    def __init__(self, batch_size: int = 64, num_episodes: int = 500):
        super().__init__()
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.env = None

    def prepare_data(self) -> None:
        pass  # Gymnasium environments are downloaded/created on first use

    def _setup_fit(self) -> None:
        self.env = gym.make(self.env_id)
        transitions = self._collect_transitions(self.num_episodes)
        self.train = transitions
        # smaller validation rollout (20% of episodes)
        val_episodes = max(1, self.num_episodes // 5)
        self.val = self._collect_transitions(val_episodes)

    def _setup_validate(self) -> None:
        self.env = gym.make(self.env_id)
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
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(float(done))
                state = next_state
        return _TransitionDataset(
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def _text_labels(self, indices) -> list:
        raise NotImplementedError("RL environments do not have text labels.")
