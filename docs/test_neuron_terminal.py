import random
import unittest

from torch import Tensor
from dataclasses import asdict
from Emperor.layers.utils.routers import RouterModel
from Emperor.layers.utils.samplers import SamplerModel
from Emperor.neuron.neuron import (
    Terminal,
    TerminalConfig,
    TerminalRangeOptions,
    TerminalZAxisOffsetOptions,
)
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
        self.assertEqual(self.model.xy_axis_range, self.config.xy_axis_range.value)
        self.assertEqual(self.model.z_axis_range, self.config.z_axis_range.value)
        self.assertIsInstance(self.model.router_model, RouterModel)
        self.assertIsInstance(self.model.sampler_model, SamplerModel)
        self.assertIsInstance(self.model.neuron_connections, Tensor)
        self.assertEqual(
            len(self.model.neuron_connections), self.model.total_neuron_connections
        )


class Test___compute_xy_axis_range(TestNeuronTerminal):
    def test_x_axis_range(self):
        num_positions = 3
        for xyz_range in TerminalRangeOptions:
            for _ in range(num_positions):
                x_position = random.randint(0, 10)
                y_position = random.randint(0, 10)
                z_position = random.randint(0, 10)
                config = TerminalConfig(
                    x_axis_position=x_position,
                    y_axis_position=y_position,
                    z_axis_position=z_position,
                    xy_axis_range=xyz_range,
                    z_axis_range=xyz_range,
                    z_axis_offset=TerminalZAxisOffsetOptions.ZERO,
                )
                self.rebuild_presets(config=config)
                message = f"__compute_xy_axis_range at xyz position: ({x_position}, {y_position}, {z_position}) with xyz range: {xyz_range.value}"
                with self.subTest(msg=message):
                    axis_range_indices = self.model._Terminal__compute_xy_axis_range()
                    expected_range_length = self.model.xy_axis_range * 2 + 1
                    expected_range_length = self.model.xy_axis_range * 2 + 1
                    expected_range_start = x_position - self.model.xy_axis_range
                    expected_range_end = x_position + self.model.xy_axis_range
                    self.assertIsInstance(axis_range_indices, Tensor)
                    self.assertEqual(axis_range_indices.numel(), expected_range_length)
                    self.assertEqual(axis_range_indices[0].item(), expected_range_start)
                    self.assertEqual(axis_range_indices[-1].item(), expected_range_end)

    def test_y_axis_range(self):
        num_positions = 3
        for xyz_range in TerminalRangeOptions:
            for _ in range(num_positions):
                x_position = random.randint(0, 10)
                y_position = random.randint(0, 10)
                z_position = random.randint(0, 10)
                config = TerminalConfig(
                    x_axis_position=x_position,
                    y_axis_position=y_position,
                    z_axis_position=z_position,
                    xy_axis_range=xyz_range,
                    z_axis_range=xyz_range,
                    z_axis_offset=TerminalZAxisOffsetOptions.ZERO,
                )
                self.rebuild_presets(config=config)
                message = f"__compute_xy_axis_range at xyz position: ({x_position}, {y_position}, {z_position}) with xy_axis_range: {xyz_range.value}"
                with self.subTest(msg=message):
                    is_y_axis_flag = True
                    axis_range_indices = self.model._Terminal__compute_xy_axis_range(
                        is_y_axis_flag
                    )
                    expected_range_length = self.model.xy_axis_range * 2 + 1
                    expected_range_start = y_position - self.model.xy_axis_range
                    expected_range_end = y_position + self.model.xy_axis_range

                    self.assertIsInstance(axis_range_indices, Tensor)
                    self.assertEqual(axis_range_indices.numel(), expected_range_length)
                    self.assertEqual(axis_range_indices[0].item(), expected_range_start)
                    self.assertEqual(axis_range_indices[-1].item(), expected_range_end)


class Test___compute_z_axis_range(TestNeuronTerminal):
    def test_without_triggering_value_errors(self):
        num_positions = 3
        for xyz_range in TerminalRangeOptions:
            for z_axis_offset in TerminalZAxisOffsetOptions:
                for _ in range(num_positions):
                    if xyz_range.value <= 2 and z_axis_offset.value > 0:
                        continue
                    if (xyz_range.value - z_axis_offset.value) <= 0:
                        continue
                    x_position = random.randint(0, 10)
                    y_position = random.randint(0, 10)
                    z_position = random.randint(0, 10)
                    config = TerminalConfig(
                        x_axis_position=x_position,
                        y_axis_position=y_position,
                        z_axis_position=z_position,
                        xy_axis_range=xyz_range,
                        z_axis_range=xyz_range,
                        z_axis_offset=z_axis_offset,
                    )
                    message = f"__compute_z_axis_range at xyz position: ({x_position}, {y_position}, {z_position}) with z range: {xyz_range.value} and z_offset: {z_axis_offset.value}"
                    self.rebuild_presets(config=config)
                    with self.subTest(msg=message):
                        axis_range_indices = (
                            self.model._Terminal__compute_z_axis_range()
                        )
                        z_range = self.model.z_axis_range
                        z_offset = self.model.z_axis_offset
                        z_offset = z_offset if z_range > 2 else 0
                        expected_range_length = z_range + 1
                        expected_range_start = z_position - z_offset
                        expected_range_end = z_position + z_range - z_offset

                        self.assertIsInstance(axis_range_indices, Tensor)
                        self.assertEqual(
                            axis_range_indices.numel(), expected_range_length
                        )
                        self.assertEqual(
                            axis_range_indices[0].item(), expected_range_start
                        )
                        self.assertEqual(
                            axis_range_indices[-1].item(), expected_range_end
                        )

    def test_ensure_value_errors_are_triggered(self):
        num_positions = 3
        for xyz_range in TerminalRangeOptions:
            for z_axis_offset in TerminalZAxisOffsetOptions:
                for _ in range(num_positions):
                    if not (xyz_range.value <= 2 and z_axis_offset.value > 0):
                        continue
                    if not ((xyz_range.value - z_axis_offset.value) <= 0):
                        continue

                    message = f"__compute_z_axis_range "
                    with self.subTest(msg=message):
                        with self.assertRaises(ValueError):
                            x_position = random.randint(0, 10)
                            y_position = random.randint(0, 10)
                            z_position = random.randint(0, 10)
                            config = TerminalConfig(
                                x_axis_position=x_position,
                                y_axis_position=y_position,
                                z_axis_position=z_position,
                                xy_axis_range=xyz_range,
                                z_axis_range=xyz_range,
                                z_axis_offset=z_axis_offset,
                            )
                            self.rebuild_presets(config=config)
                            self.model._Terminal__compute_z_axis_range()


# class Test___compute_z_axis_offset_limit(TestNeuronTerminal):
#     def test_method(self):
#         num_positions = 3
#         for xyz_range in TerminalRangeOptions:
#             for _ in range(num_positions):
#                 x_position = random.randint(0, 10)
#                 y_position = random.randint(0, 10)
#                 z_position = random.randint(0, 10)
#                 config = TerminalConfig(
#                     x_axis_position=x_position,
#                     y_axis_position=y_position,
#                     z_axis_position=z_position,
#                     xy_axis_range=xyz_range,
#                     z_axis_range=xyz_range,
#                 )
#                 message = f"__compute_z_axis_range at xyz position: ({x_position}, {y_position}, {z_position}) with z range: {xyz_range.value} and z_offset: {z_axis_offset.value}"
#                 self.rebuild_presets(config=config)
#                 with self.subTest(msg=message):
#                     axis_range_indices = (
#                         self.model._Terminal__compute_z_axis_offset_limit()
#                     )
#                     z_range = self.model.z_axis_range
#                     z_offset = self.model.z_axis_offset
#                     z_offset = z_offset if z_range > 2 else 0
#                     expected_range_length = z_range + 1
#                     expected_range_start = z_position - z_offset
#                     expected_range_end = z_position + z_range - z_offset
#
#                     self.assertIsInstance(axis_range_indices, Tensor)
#                     self.assertEqual(axis_range_indices.numel(), expected_range_length)
#                     self.assertEqual(axis_range_indices[0].item(), expected_range_start)
#                     self.assertEqual(axis_range_indices[-1].item(), expected_range_end)
