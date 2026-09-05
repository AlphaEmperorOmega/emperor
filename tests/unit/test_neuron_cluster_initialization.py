from itertools import product
from unittest.mock import patch

import torch

from emperor.neuron import NeuronCluster, NeuronClusterConfig
from unit.test_neuron import NeuronTestCase


class TestNeuronClusterInitialization(NeuronTestCase):
    def test_initialization_preserves_validation_buffer_component_and_hook_order(self):
        initialization_events = []

        class RecordingValidator(NeuronCluster.VALIDATOR):
            @classmethod
            def validate(cls, model):
                initialization_events.append(("validate",))
                super().validate(model)

        class RecordingCluster(NeuronCluster):
            VALIDATOR = RecordingValidator

            def register_buffer(self, name, tensor, persistent=True):
                initialization_events.append(("buffer", name, persistent))
                return super().register_buffer(name, tensor, persistent=persistent)

            def _initialize_neuron(self, x, y, z, runtime_template=None):
                initialization_events.append(("neuron", x, y, z))
                return super()._initialize_neuron(x, y, z, runtime_template)

            def __setattr__(self, name, value):
                if name in ("cluster", "entry_sampler", "halting_model"):
                    initialization_events.append(("component", name))
                super().__setattr__(name, value)

            def register_load_state_dict_pre_hook(self, hook):
                initialization_events.append(("load_hook", "pre", hook.__name__))
                return super().register_load_state_dict_pre_hook(hook)

            def register_load_state_dict_post_hook(self, hook):
                initialization_events.append(("load_hook", "post", hook.__name__))
                return super().register_load_state_dict_post_hook(hook)

        for escape_growth, cooldown, growth_budget, halting in product(
            (False, True), repeat=4
        ):
            with self.subTest(
                escape_growth=escape_growth,
                cooldown=cooldown,
                growth_budget=growth_budget,
                halting=halting,
            ):
                initialization_events.clear()
                config = NeuronClusterConfig(
                    x_axis_total_neurons=4,
                    y_axis_total_neurons=3,
                    z_axis_total_neurons=4,
                    initial_x_axis_total_neurons=2,
                    initial_y_axis_total_neurons=2,
                    initial_z_axis_total_neurons=2,
                    max_steps=1,
                    growth_threshold=2,
                    escape_driven_growth_flag=escape_growth,
                    growth_cooldown_steps=3 if cooldown else None,
                    max_total_growths=4 if growth_budget else None,
                    halting_config=self.halting_config() if halting else None,
                    neuron_config=self.neuron_config(),
                )

                model = RecordingCluster(config)

                optional_buffers = (
                    ("escape_counts", escape_growth, (4, 3, 4)),
                    ("forwards_since_last_growth", cooldown, ()),
                    ("total_growth_count", growth_budget, ()),
                )
                expected_buffers = ["entry_coordinates"]
                expected_events = [
                    ("validate",),
                    ("buffer", "entry_coordinates", False),
                ]
                for buffer_name, enabled, shape in optional_buffers:
                    buffer = getattr(model, buffer_name)
                    if enabled:
                        expected_buffers.append(buffer_name)
                        expected_events.append(("buffer", buffer_name, True))
                        self.assertEqual(buffer.shape, shape)
                        self.assertEqual(buffer.dtype, torch.long)
                        self.assertEqual(buffer.device, model.entry_coordinates.device)
                        self.assertEqual(torch.count_nonzero(buffer).item(), 0)
                    else:
                        self.assertIsNone(buffer)

                initial_positions = tuple(product((2, 3), (1, 2), (2, 3)))
                expected_events.extend(
                    ("neuron", *position) for position in initial_positions
                )
                expected_events.extend(
                    ("component", name)
                    for name in ("cluster", "entry_sampler", "halting_model")
                )
                expected_events.extend(
                    (
                        ("load_hook", "pre", "_reconcile_cluster_with_state_dict"),
                        (
                            "load_hook",
                            "post",
                            "_mark_growth_counters_global_after_load",
                        ),
                    )
                )
                self.assertEqual(initialization_events, expected_events)
                self.assertIs(model.cfg, config)
                self.assertEqual(tuple(model._buffers), tuple(expected_buffers))
                self.assertEqual(
                    model._non_persistent_buffers_set, {"entry_coordinates"}
                )
                expected_components = ("cluster", "entry_sampler")
                if halting:
                    expected_components += ("halting_model",)
                self.assertEqual(tuple(model._modules), expected_components)
                self.assertEqual(
                    tuple(model.cluster),
                    tuple(f"neuron_{x}_{y}_{z}" for x, y, z in initial_positions),
                )
                torch.testing.assert_close(
                    model.entry_coordinates,
                    torch.tensor([[2, 1, 2], [2, 2, 2], [3, 1, 2], [3, 2, 2]]),
                )
                self.assertFalse(model._growth_counters_are_global)
                self.assertEqual(model._checkpoint_removed_parameter_ids, set())
                self.assertEqual(len(model._load_state_dict_pre_hooks), 1)
                self.assertEqual(len(model._load_state_dict_post_hooks), 1)

    def test_rejected_config_stops_before_buffers_neurons_and_load_hooks(self):
        config = NeuronClusterConfig(
            x_axis_total_neurons=0,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            neuron_config=self.neuron_config(),
        )
        rng_before = torch.random.get_rng_state().clone()

        with (
            patch.object(NeuronCluster, "register_buffer") as register_buffer,
            patch.object(NeuronCluster, "_initialize_neuron") as initialize_neuron,
            patch.object(
                NeuronCluster, "register_load_state_dict_pre_hook"
            ) as pre_hook,
            patch.object(
                NeuronCluster, "register_load_state_dict_post_hook"
            ) as post_hook,
        ):
            with self.assertRaisesRegex(
                ValueError, "x_axis_total_neurons must be a positive integer"
            ):
                NeuronCluster(config)

            register_buffer.assert_not_called()
            initialize_neuron.assert_not_called()
            pre_hook.assert_not_called()
            post_hook.assert_not_called()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
