import gymnasium as gym
import numpy as np
import torch
import torch.utils.data

from emperor.datasets._base import DataModule


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
    num_classes: int = 0  # = num_actions for discrete, 0 for continuous
    flattened_input_dim: int = 0  # = observation_dim
    _action_dtype: torch.dtype = torch.long

    def __init__(
        self,
        batch_size: int = 64,
        num_episodes: int = 500,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.seed = None if seed is None else int(seed)
        self.env = None

    def prepare_data(self) -> None:
        pass  # Gymnasium environments are downloaded/created on first use

    def _setup_fit(self) -> None:
        self._replace_environment()
        try:
            train = self._collect_transitions(
                self.num_episodes,
                seed=self._seed_for_stage("fit"),
            )
            val = self._collect_transitions(
                self._validation_episode_count(),
                seed=self._seed_for_stage("validate"),
            )
        except BaseException:
            self.close()
            raise
        self.train = train
        self.val = val

    def _setup_validate(self) -> None:
        self._replace_environment()
        try:
            val = self._collect_transitions(
                self._validation_episode_count(),
                seed=self._seed_for_stage("validate"),
            )
        except BaseException:
            self.close()
            raise
        self.val = val

    def _make_env(self):
        return gym.make(self.env_id)

    def _replace_environment(self) -> None:
        self.close()
        self.env = self._make_env()

    def _validation_episode_count(self) -> int:
        return max(1, self.num_episodes // 5)

    def _seed_for_stage(self, stage: str) -> int | None:
        if self.seed is None:
            return None
        return self.seed if stage == "fit" else self.seed + 1

    def _collect_transitions(
        self,
        num_episodes: int,
        *,
        seed: int | None,
    ) -> _TransitionDataset:
        if self.env is None:
            raise RuntimeError("Gym environment must be created before collection.")
        states, actions, rewards, next_states, dones = [], [], [], [], []
        if seed is not None:
            self.env.action_space.seed(seed)
        for episode_index in range(num_episodes):
            state, _ = self._reset_environment(
                seed if episode_index == 0 else None
            )
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                states.append(self._encode_state(state))
                actions.append(action)
                rewards.append(reward)
                next_states.append(self._encode_state(next_state))
                dones.append(float(done))
                state = next_state
        return _TransitionDataset(
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=self._action_dtype),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def _reset_environment(self, seed: int | None):
        if seed is None:
            return self.env.reset()
        return self.env.reset(seed=seed)

    def _encode_state(self, state):
        return state

    def close(self) -> None:
        environment = self.env
        self.env = None
        if environment is not None:
            environment.close()

    def teardown(self, stage: str | None = None) -> None:
        del stage
        self.close()

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
