from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.rl._acrobot import Acrobot
from emperor.datasets.rl._base import _TransitionDataset
from emperor.datasets.rl._cart_pole import CartPole
from emperor.datasets.rl._frozen_lake import FrozenLake
from emperor.datasets.rl._lunar_lander import LunarLander
from emperor.datasets.rl._mountain_car import MountainCar
from emperor.datasets.rl._pendulum import Pendulum


class _FakeActionSpace:
    def __init__(self, *, continuous: bool) -> None:
        self.continuous = continuous
        self.seed_calls: list[int] = []
        self._seed = 97
        self._sample_index = 0

    def seed(self, seed: int) -> None:
        self.seed_calls.append(seed)
        self._seed = seed
        self._sample_index = 0

    def sample(self):
        value = self._seed + self._sample_index
        self._sample_index += 1
        if self.continuous:
            return np.array([value / 10.0], dtype=np.float32)
        return value % 4


class _FakeEnvironment:
    def __init__(
        self,
        *,
        observation_dim: int,
        discrete_state: bool = False,
        continuous_action: bool = False,
        raise_on_step: bool = False,
    ) -> None:
        self.observation_dim = observation_dim
        self.discrete_state = discrete_state
        self.action_space = _FakeActionSpace(continuous=continuous_action)
        self.raise_on_step = raise_on_step
        self.reset_calls: list[int | None] = []
        self.close_calls = 0
        self._state_seed = 41
        self._episode_index = 0

    def reset(self, *, seed: int | None = None):
        self.reset_calls.append(seed)
        if seed is not None:
            self._state_seed = seed
            self._episode_index = 0
        else:
            self._state_seed += 1
        self._episode_index += 1
        return self._state(), {}

    def step(self, action):
        if self.raise_on_step:
            raise RuntimeError("step failed")
        action_value = float(np.asarray(action).sum())
        next_state = self._state(offset=action_value + 1.0)
        terminated = self._episode_index % 2 == 0
        truncated = not terminated
        return next_state, action_value, terminated, truncated, {}

    def close(self) -> None:
        self.close_calls += 1

    def _state(self, *, offset: float = 0.0):
        if self.discrete_state:
            return int((self._state_seed + self._episode_index + offset) % 16)
        return np.array(
            [
                self._state_seed + self._episode_index + offset + index
                for index in range(self.observation_dim)
            ],
            dtype=np.float32,
        )


class _EnvironmentFactory:
    def __init__(self, *, raise_on_step: bool = False) -> None:
        self.raise_on_step = raise_on_step
        self.calls: list[tuple[str, dict]] = []
        self.environments: list[_FakeEnvironment] = []

    def __call__(self, env_id: str, **kwargs):
        self.calls.append((env_id, kwargs))
        if env_id == "FrozenLake-v1":
            environment = _FakeEnvironment(
                observation_dim=1,
                discrete_state=True,
                raise_on_step=self.raise_on_step,
            )
        elif env_id == "Pendulum-v1":
            environment = _FakeEnvironment(
                observation_dim=3,
                continuous_action=True,
                raise_on_step=self.raise_on_step,
            )
        else:
            environment = _FakeEnvironment(
                observation_dim=4,
                raise_on_step=self.raise_on_step,
            )
        self.environments.append(environment)
        return environment


def _assert_transition_datasets_equal(
    test_case: unittest.TestCase,
    first,
    second,
) -> None:
    for name in ("states", "actions", "rewards", "next_states", "dones"):
        test_case.assertTrue(torch.equal(getattr(first, name), getattr(second, name)))


class GymDatasetLifecycleTests(unittest.TestCase):
    def test_transition_dataset_indexes_fields_in_canonical_order(self) -> None:
        dataset = _TransitionDataset(
            states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            actions=torch.tensor([5, 6]),
            rewards=torch.tensor([7.0, 8.0]),
            next_states=torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
            dones=torch.tensor([0.0, 1.0]),
        )

        transition = dataset[1]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(transition), 5)
        torch.testing.assert_close(transition[0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(transition[1], torch.tensor(6))
        torch.testing.assert_close(transition[2], torch.tensor(8.0))
        torch.testing.assert_close(transition[3], torch.tensor([11.0, 12.0]))
        torch.testing.assert_close(transition[4], torch.tensor(1.0))

    def test_prepare_is_a_noop_and_collection_requires_an_environment(self) -> None:
        dataset = CartPole(num_episodes=1)

        dataset.prepare_data()

        self.assertIsNone(dataset.env)
        with self.assertRaisesRegex(
            RuntimeError,
            "Gym environment must be created before collection",
        ):
            dataset._collect_transitions(1, seed=None)

    def test_equal_seeds_reproduce_train_and_validation_transitions(self) -> None:
        factory = _EnvironmentFactory()
        with patch("emperor.datasets.rl._base.gym.make", side_effect=factory):
            first = CartPole(num_episodes=3, seed=7)
            second = CartPole(num_episodes=3, seed=7)
            first.setup("fit")
            second.setup("fit")

        _assert_transition_datasets_equal(self, first.train, second.train)
        _assert_transition_datasets_equal(self, first.val, second.val)
        self.assertEqual(factory.environments[0].action_space.seed_calls, [7, 8])
        self.assertEqual(factory.environments[0].reset_calls, [7, None, None, 8])

    def test_different_seeds_produce_different_transitions(self) -> None:
        factory = _EnvironmentFactory()
        with patch("emperor.datasets.rl._base.gym.make", side_effect=factory):
            first = CartPole(num_episodes=2, seed=3)
            second = CartPole(num_episodes=2, seed=4)
            first.setup("fit")
            second.setup("fit")

        self.assertFalse(torch.equal(first.train.states, second.train.states))
        self.assertFalse(torch.equal(first.train.actions, second.train.actions))

    def test_none_is_unseeded_and_zero_is_an_explicit_seed(self) -> None:
        factory = _EnvironmentFactory()
        with patch("emperor.datasets.rl._base.gym.make", side_effect=factory):
            unseeded = CartPole(num_episodes=1, seed=None)
            seeded = CartPole(num_episodes=1, seed=0)
            unseeded.setup("fit")
            seeded.setup("fit")

        self.assertEqual(factory.environments[0].action_space.seed_calls, [])
        self.assertEqual(factory.environments[0].reset_calls, [None, None])
        self.assertEqual(factory.environments[1].action_space.seed_calls, [0, 1])
        self.assertEqual(factory.environments[1].reset_calls, [0, 1])

    def test_repeated_setup_closes_the_replaced_environment_once(self) -> None:
        factory = _EnvironmentFactory()
        dataset = CartPole(num_episodes=1, seed=5)
        with patch("emperor.datasets.rl._base.gym.make", side_effect=factory):
            dataset.setup("fit")
            first_environment = dataset.env
            dataset.setup("validate")

        self.assertEqual(first_environment.close_calls, 1)
        self.assertIs(dataset.env, factory.environments[1])
        self.assertEqual(dataset.env.close_calls, 0)

    def test_step_failure_closes_and_releases_the_environment(self) -> None:
        factory = _EnvironmentFactory(raise_on_step=True)
        dataset = CartPole(num_episodes=1, seed=5)

        with (
            patch("emperor.datasets.rl._base.gym.make", side_effect=factory),
            self.assertRaisesRegex(RuntimeError, "step failed"),
        ):
            dataset.setup("fit")

        self.assertEqual(factory.environments[0].close_calls, 1)
        self.assertIsNone(dataset.env)
        self.assertFalse(hasattr(dataset, "train"))

    def test_validation_failure_closes_and_releases_the_environment(self) -> None:
        factory = _EnvironmentFactory(raise_on_step=True)
        dataset = CartPole(num_episodes=1, seed=5)

        with (
            patch("emperor.datasets.rl._base.gym.make", side_effect=factory),
            self.assertRaisesRegex(RuntimeError, "step failed"),
        ):
            dataset.setup("validate")

        self.assertEqual(factory.environments[0].close_calls, 1)
        self.assertIsNone(dataset.env)
        self.assertFalse(hasattr(dataset, "val"))

    def test_loader_policies_and_text_label_rejection_are_explicit(self) -> None:
        transitions = _TransitionDataset(
            states=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            actions=torch.tensor([0, 1, 0]),
            rewards=torch.tensor([1.0, 2.0, 3.0]),
            next_states=torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
            dones=torch.tensor([0.0, 0.0, 1.0]),
        )
        dataset = CartPole(batch_size=2, num_episodes=1)
        dataset.num_workers = 0
        dataset.train = transitions
        dataset.val = transitions

        training_loader = dataset.get_dataloader(train=True)
        validation_loader = dataset.get_dataloader(train=False)

        self.assertIsInstance(training_loader.sampler, RandomSampler)
        self.assertIsInstance(validation_loader.sampler, SequentialSampler)
        self.assertTrue(training_loader.drop_last)
        self.assertTrue(validation_loader.drop_last)
        self.assertEqual(training_loader.batch_size, 2)
        self.assertEqual(validation_loader.batch_size, 2)
        with self.assertRaisesRegex(
            NotImplementedError,
            "RL environments do not have text labels",
        ):
            dataset._text_labels([0])

    def test_close_and_teardown_are_idempotent(self) -> None:
        factory = _EnvironmentFactory()
        dataset = CartPole(num_episodes=1, seed=5)
        with patch("emperor.datasets.rl._base.gym.make", side_effect=factory):
            dataset.setup("fit")

        environment = dataset.env
        dataset.close()
        dataset.close()
        dataset.teardown("fit")

        self.assertEqual(environment.close_calls, 1)
        self.assertIsNone(dataset.env)

    def test_termination_and_truncation_both_emit_done(self) -> None:
        factory = _EnvironmentFactory()
        dataset = CartPole(num_episodes=2, seed=2)
        with patch("emperor.datasets.rl._base.gym.make", side_effect=factory):
            dataset.setup("fit")

        torch.testing.assert_close(dataset.train.dones, torch.ones(2))
        self.assertEqual(len(dataset.train), 2)
        self.assertEqual(len(dataset.val), 1)
        self.assertIs(dataset.env, factory.environments[0])

    def test_frozen_lake_keeps_options_and_one_hot_state_schema(self) -> None:
        factory = _EnvironmentFactory()
        dataset = FrozenLake(num_episodes=2, is_slippery=False, seed=2)
        with patch("emperor.datasets.rl._base.gym.make", side_effect=factory):
            dataset.setup("fit")

        self.assertEqual(factory.calls, [("FrozenLake-v1", {"is_slippery": False})])
        self.assertEqual(dataset.train.states.shape, torch.Size([2, 16]))
        self.assertEqual(dataset.train.next_states.shape, torch.Size([2, 16]))
        self.assertEqual(dataset.train.states.dtype, torch.float32)
        self.assertTrue(torch.all(dataset.train.states.sum(dim=1) == 1))
        self.assertEqual(dataset.train.actions.dtype, torch.long)

    def test_pendulum_keeps_float_action_schema(self) -> None:
        factory = _EnvironmentFactory()
        dataset = Pendulum(num_episodes=2, seed=2)
        with patch("emperor.datasets.rl._base.gym.make", side_effect=factory):
            dataset.setup("fit")

        self.assertEqual(dataset.train.states.shape, torch.Size([2, 3]))
        self.assertEqual(dataset.train.actions.shape, torch.Size([2, 1]))
        self.assertEqual(dataset.train.actions.dtype, torch.float32)
        self.assertEqual(dataset.train.rewards.dtype, torch.float32)

    def test_thin_discrete_leaves_keep_their_state_and_action_schemas(self) -> None:
        cases = (
            (Acrobot, "Acrobot-v1", 6, 3),
            (CartPole, "CartPole-v1", 4, 2),
            (LunarLander, "LunarLander-v2", 8, 4),
            (MountainCar, "MountainCar-v0", 2, 3),
        )
        for dataset_type, env_id, observation_dim, num_actions in cases:
            with self.subTest(dataset=dataset_type.__name__):
                environment = _FakeEnvironment(observation_dim=observation_dim)
                dataset = dataset_type(num_episodes=1, seed=0)
                with patch.object(dataset, "_make_env", return_value=environment):
                    dataset.setup("fit")

                self.assertEqual(dataset.env_id, env_id)
                self.assertEqual(dataset.observation_dim, observation_dim)
                self.assertEqual(dataset.num_actions, num_actions)
                self.assertEqual(
                    dataset.train.states.shape,
                    torch.Size([1, observation_dim]),
                )
                self.assertEqual(dataset.train.actions.shape, torch.Size([1]))
                self.assertEqual(dataset.train.actions.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
