from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch import Tensor

from emperor.neuron import Neuron
from unit.test_neuron import FourFieldOnlySampler, NeuronTestCase


class TestNeuron(NeuronTestCase):
    def test_width_one_is_rejected_before_coordinate_embedding_and_counters(self):
        neuron = self.neuron_config(coordinate_embedding_flag=True).build()
        with self.assertRaisesRegex(ValueError, "feature dimension"):
            neuron(torch.ones(2, 1))
        self.assertEqual(neuron.batch_counter.item(), 0)

    def test_all_signal_interfaces_reject_wrong_width_before_processing(self):
        for embeddings in (False, True):
            neuron = self.neuron_config(coordinate_embedding_flag=embeddings).build()
            for interface in ("forward", "process_signal", "route_signal"):
                for width in (0, 1, 3, 5):
                    with self.subTest(
                        embeddings=embeddings, interface=interface, width=width
                    ):
                        with (
                            patch.object(
                                neuron.nucleus,
                                "forward",
                                side_effect=AssertionError("nucleus called"),
                            ),
                            patch.object(
                                neuron.terminal,
                                "forward",
                                side_effect=AssertionError("terminal called"),
                            ),
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "feature dimension"
                            ):
                                getattr(neuron, interface)(torch.ones(2, width))
                        self.assertEqual(neuron.batch_counter.item(), 0)
                        self.assertEqual(neuron.atrophy_counter.item(), 0)

    def test_signal_interfaces_accept_valid_empty_inputs(self):
        for embeddings in (False, True):
            for interface in ("forward", "process_signal", "route_signal"):
                with self.subTest(embeddings=embeddings, interface=interface):
                    neuron = self.neuron_config(
                        coordinate_embedding_flag=embeddings
                    ).build()
                    source = torch.empty(0, self.input_dim, requires_grad=True)
                    result = getattr(neuron, interface)(source)
                    output = result[0] if isinstance(result, tuple) else result
                    self.assertEqual(output.shape[0], 0)
                    output.sum().backward()
                    torch.testing.assert_close(source.grad, torch.zeros_like(source))
                    self.assertEqual(
                        neuron.batch_counter.item(), int(interface != "route_signal")
                    )

    def test_initialization_preserves_validation_component_and_buffer_order(self):
        initialization_events = []

        class RecordingValidator(Neuron.VALIDATOR):
            @classmethod
            def validate(cls, cfg):
                initialization_events.append(("validate",))
                super().validate(cfg)

        class RecordingNeuron(Neuron):
            VALIDATOR = RecordingValidator

            def register_buffer(self, name, tensor, persistent=True):
                initialization_events.append(("buffer", name, persistent))
                return super().register_buffer(name, tensor, persistent=persistent)

        for coordinate_embedding_flag in (None, False, True):
            with self.subTest(coordinate_embedding_flag=coordinate_embedding_flag):
                initialization_events.clear()
                config = self.neuron_config(coordinate_embedding_flag)
                with ExitStack() as patches:
                    for component_name in ("nucleus", "axons", "terminal"):
                        component_config = getattr(config, f"{component_name}_config")

                        def record_build(
                            build=component_config.build,
                            name=component_name,
                        ):
                            initialization_events.append(("build", name))
                            return build()

                        patches.enter_context(
                            patch.object(component_config, "build", record_build)
                        )
                    model = RecordingNeuron(config)

                expected_events = [
                    ("validate",),
                    ("build", "nucleus"),
                    ("build", "axons"),
                    ("build", "terminal"),
                    ("buffer", "batch_counter", True),
                    ("buffer", "atrophy_counter", True),
                ]
                if coordinate_embedding_flag:
                    expected_events.append(("buffer", "coordinate_embedding", False))
                self.assertEqual(initialization_events, expected_events)
                self.assertIs(model.cfg, config)
                self.assertEqual(
                    tuple(model._modules), ("nucleus", "axons", "terminal")
                )
                for counter in (model.batch_counter, model.atrophy_counter):
                    self.assertEqual(counter.shape, ())
                    self.assertEqual(counter.dtype, torch.int64)
                    self.assertEqual(counter.item(), 0)
                self.assertIsNot(model.batch_counter, model.atrophy_counter)

    def test_rejected_config_stops_before_component_or_buffer_initialization(self):
        config = self.neuron_config(coordinate_embedding_flag=1)
        rng_before = torch.random.get_rng_state().clone()

        with (
            patch.object(config.nucleus_config, "build") as build_nucleus,
            patch.object(config.axons_config, "build") as build_axons,
            patch.object(config.terminal_config, "build") as build_terminal,
            patch.object(Neuron, "register_buffer") as register_buffer,
        ):
            with self.assertRaisesRegex(
                TypeError, "coordinate_embedding_flag must be a bool"
            ):
                Neuron(config)

            build_nucleus.assert_not_called()
            build_axons.assert_not_called()
            build_terminal.assert_not_called()
            register_buffer.assert_not_called()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_composes_nucleus_axons_and_terminal(self):
        model = self.neuron_config().build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output, probabilities, selected_neurons, auxiliary_loss = model(input_batch)

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertIsInstance(auxiliary_loss, Tensor)
        self.assertEqual(model.batch_counter.item(), 1)

    def test_forward_uses_only_four_field_sampler_interface(self):
        model = self.neuron_config().build()
        model.terminal.sampler = FourFieldOnlySampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )

        _, probabilities, selected_neurons, auxiliary_loss = model(
            torch.randn(self.batch_size, self.input_dim)
        )

        torch.testing.assert_close(
            probabilities,
            torch.tensor([[0.25, 0.75]]).expand(self.batch_size, -1),
        )
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertEqual(auxiliary_loss.shape, ())

    def test_coordinate_embedding_disabled_by_default(self):
        model = self.neuron_config().build()

        self.assertIsNone(model.coordinate_embedding)

    def test_coordinate_embedding_matches_sinusoidal_encoding(self):
        model = self.neuron_config(coordinate_embedding_flag=True).build()

        torch.testing.assert_close(
            model.coordinate_embedding,
            self.expected_coordinate_embedding(1, 1, 1),
        )

    def test_coordinate_embedding_differs_across_coordinates(self):
        base_model = self.neuron_config(coordinate_embedding_flag=True).build()
        shifted_config = self.neuron_config(coordinate_embedding_flag=True)
        shifted_config.terminal_config.x_axis_position = 2
        shifted_model = shifted_config.build()

        self.assertFalse(
            torch.allclose(
                base_model.coordinate_embedding,
                shifted_model.coordinate_embedding,
            )
        )

    def test_process_signal_injects_coordinate_embedding_into_nucleus(self):
        model = self.neuron_config(coordinate_embedding_flag=True).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output = model.process_signal(input_batch)

        torch.testing.assert_close(
            output,
            model.nucleus(input_batch + model.coordinate_embedding),
        )

    def test_route_signal_injects_coordinate_embedding_into_terminal(self):
        model = self.neuron_config(coordinate_embedding_flag=True).build()
        model.eval()
        processed_signal = torch.randn(self.batch_size, self.input_dim)

        probabilities, selected_neurons, _ = model.route_signal(processed_signal)

        _, expected_probabilities, expected_selected_neurons, _ = model.terminal(
            processed_signal + model.coordinate_embedding
        )
        torch.testing.assert_close(probabilities, expected_probabilities)
        torch.testing.assert_close(selected_neurons, expected_selected_neurons)

    def test_route_signal_runs_terminal_forward_hook_once(self):
        model = self.neuron_config().build()
        processed_signal = torch.randn(self.batch_size, self.input_dim)
        hook_outputs: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        hook_handle = model.terminal.register_forward_hook(
            lambda _module, _inputs, output: hook_outputs.append(output)
        )
        try:
            probabilities, selected_neurons, auxiliary_loss = model.route_signal(
                processed_signal
            )
        finally:
            hook_handle.remove()

        self.assertEqual(len(hook_outputs), 1)
        self.assertEqual(len(hook_outputs[0]), 4)
        self.assertEqual(probabilities.shape[0], processed_signal.shape[0])
        self.assertEqual(selected_neurons.shape[0], processed_signal.shape[0])
        self.assertEqual(auxiliary_loss.shape, ())

    def test_route_signal_runs_terminal_backward_hook_once(self):
        model = self.neuron_config().build()
        processed_signal = torch.randn(
            self.batch_size,
            self.input_dim,
            requires_grad=True,
        )
        hook_calls: list[
            tuple[tuple[Tensor | None, ...], tuple[Tensor | None, ...]]
        ] = []
        hook_handle = model.terminal.register_full_backward_hook(
            lambda _module, grad_input, grad_output: hook_calls.append(
                (grad_input, grad_output)
            )
        )
        try:
            probabilities, _, _ = model.route_signal(processed_signal)
            probabilities.sum().backward()
        finally:
            hook_handle.remove()

        self.assertEqual(len(hook_calls), 1)
        self.assertIsNotNone(processed_signal.grad)
        self.assertTrue(torch.isfinite(processed_signal.grad).all().item())

    def test_coordinate_embedding_excluded_from_state_dict(self):
        model = self.neuron_config(coordinate_embedding_flag=True).build()

        self.assertNotIn("coordinate_embedding", model.state_dict())

    def test_coordinate_embedding_requires_minimum_input_dim(self):
        config = self.neuron_config(coordinate_embedding_flag=True)
        config.terminal_config.input_dim = 2

        with self.assertRaisesRegex(
            ValueError,
            "coordinate_embedding_flag requires terminal_config.input_dim",
        ):
            config.build()

    def test_coordinate_embedding_flag_rejects_non_bool(self):
        config = self.neuron_config()
        config.coordinate_embedding_flag = 1

        with self.assertRaisesRegex(
            TypeError,
            "coordinate_embedding_flag must be a bool",
        ):
            config.build()
