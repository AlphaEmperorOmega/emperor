import torch
import numpy as np

from emperor.datasets.rl.base import GymEnvironment, _TransitionDataset


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
    num_actions: int = 0   # continuous action space
    num_classes: int = 0
    flattened_input_dim: int = observation_dim

    def __init__(self, batch_size: int = 64, num_episodes: int = 500):
        super().__init__(batch_size=batch_size, num_episodes=num_episodes)

    def _collect_transitions(self, num_episodes: int) -> _TransitionDataset:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()  # shape (1,)
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
            torch.tensor(np.array(actions), dtype=torch.float32),  # float for continuous
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )
