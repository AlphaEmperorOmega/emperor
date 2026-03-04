from emperor.datasets.rl.base import GymEnvironment


class CartPole(GymEnvironment):
    """CartPole-v1: balance a pole on a moving cart.

    Observation: [cart_pos, cart_vel, pole_angle, pole_vel] — 4 floats
    Actions:     0 = push left, 1 = push right
    Reward:      +1 for every step the pole stays upright
    Episode end: pole angle > 12°, cart out of bounds, or 500 steps
    """

    env_id: str = "CartPole-v1"
    observation_dim: int = 4
    num_actions: int = 2
    num_classes: int = num_actions
    flattened_input_dim: int = observation_dim

    def __init__(self, batch_size: int = 64, num_episodes: int = 500):
        super().__init__(batch_size=batch_size, num_episodes=num_episodes)
