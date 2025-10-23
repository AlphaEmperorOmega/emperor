import unittest

from dataclasses import asdict

from Emperor.neuron.neuron import Terminal, TerminalConfig
from docs.config import default_unittest_config


class TestNeuronTerminal(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None

    def rebuild_presets(self, config: TerminalConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.neuron_terminal_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = Terminal(self.cfg)

        self.batch_size: int = self.cfg.batch_size
        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim


class Test___init(TestNeuronTerminal):
    def test_initialization(self):
        self.assertEqual(self.model.x_axis_position, self.config.x_axis_position)
        self.assertEqual(self.model.y_axis_position, self.config.y_axis_position)
        self.assertEqual(self.model.z_axis_position, self.config.z_axis_position)
        self.assertEqual(self.model.x_axis_range, self.config.x_axis_range)
        self.assertEqual(self.model.y_axis_range, self.config.y_axis_range)
        self.assertEqual(self.model.z_axis_range, self.config.z_axis_range)


class Test___compute_xy_axis_range(TestNeuronTerminal):
    def test_x_axis_range(self):
        range_options = [2, 4, 8, 16]
        position_options = [0, 1, 2, 3]
        for option in range_options:
            for position in position_options:
                config = TerminalConfig(
                    x_axis_position=position,
                    y_axis_position=position,
                    z_axis_position=position,
                    x_axis_range=option,
                    y_axis_range=option,
                    z_axis_range=option,
                )
                self.rebuild_presets(config=config)
                message = f"__compute_xy_axis_range at xyz position: ({position}) with xyz range: {option}"
                with self.subTest(msg=message):
                    x_axis_range_indices = self.model._Terminal__compute_xy_axis_range()
                    print(x_axis_range_indices)
