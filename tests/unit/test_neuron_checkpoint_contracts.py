from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from emperor.config import ConfigBase, optional_field
from emperor.neuron import NeuronClusterConfig, NeuronConfig, NucleusConfig
from emperor.nn import Module
from model_runtime.runs.checkpoints import (
    CheckpointContinuation,
    _LoadedCheckpointContinuation,
    validate_model_state,
)
from unit.test_neuron import NeuronTestCase


@dataclass
class CheckpointContextProjectionConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return CheckpointContextProjection


class CheckpointContextProjection(Module):
    def __init__(
        self,
        cfg: CheckpointContextProjectionConfig,
        overrides: CheckpointContextProjectionConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: CheckpointContextProjectionConfig = self._override_config(
            cfg,
            overrides,
        )
        self.weight = nn.Parameter(
            torch.full(
                (self.cfg.input_dim, self.cfg.output_dim),
                0.25,
            )
        )
        self.runtime_context_marker = nn.Parameter(torch.ones(()))
        self.register_buffer(
            "runtime_context_buffer",
            torch.ones(()),
            persistent=True,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor @ self.weight


class TestNeuronCheckpointTopology(NeuronTestCase):
    def build_cluster(
        self,
        *,
        capacity: int,
        initial: int | None = None,
        neuron_config: NeuronConfig | None = None,
    ):
        return NeuronClusterConfig(
            x_axis_total_neurons=capacity,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=initial,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=neuron_config or self.neuron_config(),
        ).build()

    def random_neuron_config(self) -> NeuronConfig:
        base = self.neuron_config()
        return NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.router_config(
                    self.input_dim, self.input_dim
                ).model_config
            ),
            axons_config=base.axons_config,
            terminal_config=base.terminal_config,
            coordinate_embedding_flag=base.coordinate_embedding_flag,
        )

    def renamed_neuron_state(self, state, source_name: str, target_name: str):
        renamed = state.__class__()
        for key, value in state.items():
            renamed[key.replace(source_name, target_name)] = value.clone()
        return renamed

    def test_strict_load_rejects_out_of_capacity_topology_before_mutation(self) -> None:
        for invalid_name in ("neuron_0_1_1", "neuron_2_1_1"):
            with self.subTest(invalid_name=invalid_name):
                model = self.build_cluster(capacity=1)
                state = self.renamed_neuron_state(
                    model.state_dict(),
                    "neuron_1_1_1",
                    invalid_name,
                )

                with self.assertRaises(RuntimeError) as raised:
                    model.load_state_dict(state, strict=True)

                self.assertIn(
                    "NeuronCluster checkpoint topology contains neurons outside "
                    "the configured cluster capacity: "
                    f"['{invalid_name}'].",
                    str(raised.exception),
                )
                self.assertEqual(set(model.cluster), {"neuron_1_1_1"})

    def test_strict_load_rejects_missing_entry_plane_neuron_before_mutation(
        self,
    ) -> None:
        model = self.build_cluster(capacity=2, initial=2)
        state = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if not key.startswith("cluster.neuron_2_1_1.")
        }

        with self.assertRaises(RuntimeError) as raised:
            model.load_state_dict(state, strict=True)

        self.assertIn(
            "NeuronCluster checkpoint topology is missing configured entry-plane "
            "neurons: ['neuron_2_1_1'].",
            str(raised.exception),
        )
        self.assertEqual(
            set(model.cluster),
            {"neuron_1_1_1", "neuron_2_1_1"},
        )

    def test_strict_load_rejects_noncanonical_neuron_alias_before_mutation(
        self,
    ) -> None:
        source = self.build_cluster(capacity=2, initial=1)
        source.cluster["neuron_2_1_1"] = source._initialize_neuron(2, 1, 1)
        aliased_state = self.renamed_neuron_state(
            source.state_dict(),
            "neuron_2_1_1",
            "neuron_02_1_1",
        )
        target = self.build_cluster(capacity=2, initial=1)
        original_neuron = target.cluster["neuron_1_1_1"]

        with self.assertRaises(RuntimeError) as raised:
            target.load_state_dict(aliased_state, strict=True)

        self.assertIn(
            "NeuronCluster checkpoint topology contains non-canonical neuron "
            "names: ['neuron_02_1_1'].",
            str(raised.exception),
        )
        self.assertEqual(tuple(target.cluster), ("neuron_1_1_1",))
        self.assertIs(target.cluster["neuron_1_1_1"], original_neuron)
        self.assertNotIn("neuron_02_1_1", target.cluster)

    def test_runtime_validation_reconstructs_dynamic_topology_before_training(
        self,
    ) -> None:
        source = self.build_cluster(capacity=2, initial=1)
        source.cluster["neuron_2_1_1"] = source._initialize_neuron(2, 1, 1)
        target = self.build_cluster(capacity=2, initial=1)
        continuation = _LoadedCheckpointContinuation(
            request=CheckpointContinuation(Path("dynamic.ckpt")),
            state_dict=source.state_dict(),
            epoch=0,
            global_step=1,
        )

        validate_model_state(continuation, target)

        self.assertEqual(set(target.cluster), set(source.cluster))
        target_state = target.state_dict()
        for key, expected in source.state_dict().items():
            torch.testing.assert_close(target_state[key], expected)

    def test_dynamic_strict_load_does_not_consume_rng_for_throwaway_initialization(
        self,
    ) -> None:
        neuron_config = self.random_neuron_config()
        source = self.build_cluster(
            capacity=2,
            initial=1,
            neuron_config=neuron_config,
        )
        source.cluster["neuron_2_1_1"] = source._initialize_neuron(2, 1, 1)
        target = self.build_cluster(
            capacity=2,
            initial=1,
            neuron_config=neuron_config,
        )
        torch.manual_seed(20260718)
        expected_rng_state = torch.random.get_rng_state().clone()

        target.load_state_dict(source.state_dict(), strict=True)

        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng_state)
        self.assertEqual(tuple(target.cluster), tuple(source.cluster))
        for key, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], expected)

    def test_dynamic_strict_load_inherits_eval_and_frozen_parameter_policy(
        self,
    ) -> None:
        source = self.build_cluster(capacity=2, initial=1)
        source.cluster["neuron_2_1_1"] = source._initialize_neuron(2, 1, 1)
        target = self.build_cluster(capacity=2, initial=1)
        target.eval()
        target.requires_grad_(False)

        target.load_state_dict(source.state_dict(), strict=True)

        reconstructed_neuron = target.cluster["neuron_2_1_1"]
        self.assertTrue(
            all(not module.training for module in reconstructed_neuron.modules())
        )
        self.assertTrue(tuple(reconstructed_neuron.parameters()))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in reconstructed_neuron.parameters()
            )
        )

    def test_dynamic_strict_load_inherits_template_role_runtime_policy(
        self,
    ) -> None:
        source = self.build_cluster(capacity=2, initial=1)
        source.cluster["neuron_2_1_1"] = source._initialize_neuron(2, 1, 1)
        target = self.build_cluster(capacity=2, initial=1)
        template = target.cluster["neuron_1_1_1"]
        template.nucleus.eval()
        template.terminal.train()
        template.terminal.sampler.eval()
        template.nucleus.model.weight.requires_grad_(False)
        template.terminal.sampler.router.model.layers[
            0
        ].model.bias_params.requires_grad_(False)
        expected_training_modes = {
            name: module.training
            for name, module in template.named_modules(remove_duplicate=False)
        }
        expected_trainability = {
            name: parameter.requires_grad
            for name, parameter in template.named_parameters(remove_duplicate=False)
        }

        target.load_state_dict(source.state_dict(), strict=True)

        reconstructed = target.cluster["neuron_2_1_1"]
        self.assertEqual(
            {
                name: module.training
                for name, module in reconstructed.named_modules(remove_duplicate=False)
            },
            expected_training_modes,
        )
        self.assertEqual(
            {
                name: parameter.requires_grad
                for name, parameter in reconstructed.named_parameters(
                    remove_duplicate=False
                )
            },
            expected_trainability,
        )

    def test_real_grown_warmup_buffer_strict_loads_and_saturates_at_zero(
        self,
    ) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            growth_warmup_steps=2,
            neuron_config=self.full_sampler_neuron_config(),
        )
        source = config.build()

        source(torch.randn(self.batch_size, self.input_dim))

        grown_name = "neuron_2_1_1"
        warmup_key = f"cluster.{grown_name}.warmup_remaining_steps"
        self.assertIn(grown_name, source.cluster)
        self.assertIn(warmup_key, source.state_dict())
        source_warmup = source.cluster[grown_name].warmup_remaining_steps
        self.assertEqual(source_warmup.dtype, torch.int64)
        self.assertEqual(
            source_warmup.device, source.cluster[grown_name].batch_counter.device
        )
        self.assertEqual(int(source_warmup), 2)

        target = config.build()
        target.load_state_dict(source.state_dict(), strict=True)
        restored_warmup = target.cluster[grown_name].warmup_remaining_steps
        observed_countdown = [int(restored_warmup)]
        for _ in range(3):
            target(torch.randn(self.batch_size, self.input_dim))
            observed_countdown.append(int(restored_warmup))

        self.assertEqual(observed_countdown, [2, 1, 0, 0])

    def test_dynamic_strict_load_inherits_template_context_by_role(self) -> None:
        base_neuron_config = self.neuron_config()
        context_neuron_config = NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=CheckpointContextProjectionConfig(
                    input_dim=self.input_dim,
                    output_dim=self.input_dim,
                )
            ),
            axons_config=base_neuron_config.axons_config,
            terminal_config=base_neuron_config.terminal_config,
            coordinate_embedding_flag=base_neuron_config.coordinate_embedding_flag,
        )
        source = self.build_cluster(
            capacity=2,
            initial=1,
            neuron_config=context_neuron_config,
        )
        source.cluster["neuron_2_1_1"] = source._initialize_neuron(2, 1, 1)
        target = self.build_cluster(
            capacity=2,
            initial=1,
            neuron_config=context_neuron_config,
        ).double()
        template = target.cluster["neuron_1_1_1"]
        template_model = template.nucleus.model
        template_model.runtime_context_marker.data = (
            template_model.runtime_context_marker.data.float()
        )
        template_model.runtime_context_buffer = (
            template_model.runtime_context_buffer.float()
        )
        expected_parameter_contexts = {
            name: (parameter.device, parameter.dtype)
            for name, parameter in template.named_parameters(remove_duplicate=False)
        }
        expected_buffer_contexts = {
            name: (buffer.device, buffer.dtype)
            for name, buffer in template.named_buffers(remove_duplicate=False)
        }

        target.load_state_dict(source.state_dict(), strict=True)

        reconstructed = target.cluster["neuron_2_1_1"]
        self.assertEqual(
            {
                name: (parameter.device, parameter.dtype)
                for name, parameter in reconstructed.named_parameters(
                    remove_duplicate=False
                )
            },
            expected_parameter_contexts,
        )
        self.assertEqual(
            {
                name: (buffer.device, buffer.dtype)
                for name, buffer in reconstructed.named_buffers(remove_duplicate=False)
            },
            expected_buffer_contexts,
        )

    def test_runtime_validation_preserves_supported_legacy_buffer_seeding(
        self,
    ) -> None:
        source = self.build_cluster(capacity=1)
        legacy_state = {
            key: value.clone()
            for key, value in source.state_dict().items()
            if not key.endswith(".atrophy_counter")
        }
        target = self.build_cluster(capacity=1)
        continuation = _LoadedCheckpointContinuation(
            request=CheckpointContinuation(Path("legacy.ckpt")),
            state_dict=legacy_state,
            epoch=0,
            global_step=1,
        )

        validate_model_state(continuation, target)

        atrophy_keys = [
            key for key in target.state_dict() if key.endswith(".atrophy_counter")
        ]
        self.assertTrue(atrophy_keys)
        for key in atrophy_keys:
            self.assertEqual(int(target.state_dict()[key]), 0)

    def test_empty_non_strict_state_does_not_erase_live_topology(self) -> None:
        model = self.build_cluster(capacity=2, initial=2)
        original_neurons = dict(model.cluster.items())

        incompatible = model.load_state_dict({}, strict=False)

        self.assertTrue(incompatible.missing_keys)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(tuple(model.cluster), tuple(original_neurons))
        for name, neuron in original_neurons.items():
            self.assertIs(model.cluster[name], neuron)

    def test_malformed_warmup_key_remains_unexpected_without_creating_topology(
        self,
    ) -> None:
        model = self.build_cluster(capacity=1)
        original_neuron = model.cluster["neuron_1_1_1"]
        state = model.state_dict()
        malformed_key = "cluster.metadata.warmup_remaining_steps"
        state[malformed_key] = torch.tensor(7, dtype=torch.int64)

        incompatible = model.load_state_dict(state, strict=False)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [malformed_key])
        self.assertEqual(tuple(model.cluster), ("neuron_1_1_1",))
        self.assertIs(model.cluster["neuron_1_1_1"], original_neuron)
        self.assertNotIn("metadata", model.cluster)

    def test_malformed_warmup_key_does_not_hide_later_valid_warmup_state(self) -> None:
        model = self.build_cluster(capacity=1)
        original_state = model.state_dict()
        malformed_key = "cluster.metadata.warmup_remaining_steps"
        valid_key = "cluster.neuron_1_1_1.warmup_remaining_steps"
        incoming_state = original_state.__class__()
        incoming_state[malformed_key] = torch.tensor(7, dtype=torch.int64)
        incoming_state.update(original_state)
        incoming_state[valid_key] = torch.tensor(3, dtype=torch.int64)

        incompatible = model.load_state_dict(incoming_state, strict=False)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [malformed_key])
        warmup_remaining = model.cluster["neuron_1_1_1"].warmup_remaining_steps
        self.assertEqual(warmup_remaining.dtype, torch.int64)
        self.assertEqual(int(warmup_remaining), 3)
        self.assertIn(valid_key, model.state_dict())

    def test_legacy_warmup_seed_continues_past_initial_neuron(self) -> None:
        source = self.build_cluster(capacity=2, initial=1)
        source.cluster["neuron_2_1_1"] = source._initialize_neuron(2, 1, 1)
        target = self.build_cluster(capacity=2, initial=1)
        target.cluster["neuron_2_1_1"] = target._initialize_neuron(2, 1, 1)
        target.cluster["neuron_2_1_1"].register_buffer(
            "warmup_remaining_steps",
            torch.tensor(9, dtype=torch.int64),
            persistent=True,
        )

        incompatible = target.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertFalse(
            hasattr(target.cluster["neuron_1_1_1"], "warmup_remaining_steps")
        )
        grown_warmup = target.cluster["neuron_2_1_1"].warmup_remaining_steps
        self.assertEqual(grown_warmup.dtype, torch.int64)
        self.assertEqual(int(grown_warmup), 0)

    def test_existing_warmup_buffer_is_loaded_without_reregistration(self) -> None:
        source = self.build_cluster(capacity=1)
        source_neuron = source.cluster["neuron_1_1_1"]
        source_neuron.register_buffer(
            "warmup_remaining_steps",
            torch.tensor(3, dtype=torch.int64),
            persistent=True,
        )
        target = self.build_cluster(capacity=1)
        target_neuron = target.cluster["neuron_1_1_1"]
        target_neuron.register_buffer(
            "warmup_remaining_steps",
            torch.tensor(9, dtype=torch.int64),
            persistent=True,
        )

        incompatible = target.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(int(target_neuron.warmup_remaining_steps), 3)

    def test_checkpoint_rebuild_preserves_optimizer_parameter_order(self) -> None:
        source = self.build_cluster(capacity=5, initial=1)
        source.cluster["neuron_4_1_1"] = source._initialize_neuron(4, 1, 1)
        source.cluster["neuron_2_1_1"] = source._initialize_neuron(2, 1, 1)
        self.assertEqual(
            tuple(source.cluster),
            ("neuron_3_1_1", "neuron_4_1_1", "neuron_2_1_1"),
        )
        source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
        sum(parameter.sum() for parameter in source.parameters()).backward()
        source_optimizer.step()
        for neuron_name, marker in (
            ("neuron_4_1_1", 4.0),
            ("neuron_2_1_1", 2.0),
        ):
            parameter = source.cluster[neuron_name].nucleus.model.weight
            source_optimizer.state[parameter]["exp_avg"].fill_(marker)

        target = self.build_cluster(capacity=5, initial=1)
        target.load_state_dict(source.state_dict(), strict=True)
        target_optimizer = torch.optim.Adam(target.parameters(), lr=0.01)
        target_optimizer.load_state_dict(source_optimizer.state_dict())

        self.assertEqual(tuple(target.cluster), tuple(source.cluster))
        for neuron_name, expected_marker in (
            ("neuron_4_1_1", 4.0),
            ("neuron_2_1_1", 2.0),
        ):
            parameter = target.cluster[neuron_name].nucleus.model.weight
            torch.testing.assert_close(
                target_optimizer.state[parameter]["exp_avg"],
                torch.full_like(parameter, expected_marker),
            )

    def test_checkpoint_reorders_preexisting_topology_without_replacing_modules(
        self,
    ) -> None:
        source = self.build_cluster(capacity=5, initial=1)
        source.cluster["neuron_4_1_1"] = source._initialize_neuron(4, 1, 1)
        source.cluster["neuron_2_1_1"] = source._initialize_neuron(2, 1, 1)
        source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
        sum(parameter.sum() for parameter in source.parameters()).backward()
        source_optimizer.step()
        for neuron_name, marker in (
            ("neuron_4_1_1", 4.0),
            ("neuron_2_1_1", 2.0),
        ):
            parameter = source.cluster[neuron_name].nucleus.model.weight
            source_optimizer.state[parameter]["exp_avg"].fill_(marker)

        target = self.build_cluster(capacity=5, initial=1)
        target.cluster["neuron_2_1_1"] = target._initialize_neuron(2, 1, 1)
        target.cluster["neuron_4_1_1"] = target._initialize_neuron(4, 1, 1)
        original_modules = dict(target.cluster.items())
        self.assertEqual(
            tuple(target.cluster),
            ("neuron_3_1_1", "neuron_2_1_1", "neuron_4_1_1"),
        )

        target.load_state_dict(source.state_dict(), strict=True)
        target_optimizer = torch.optim.Adam(target.parameters(), lr=0.01)
        target_optimizer.load_state_dict(source_optimizer.state_dict())

        self.assertEqual(tuple(target.cluster), tuple(source.cluster))
        for neuron_name, original_module in original_modules.items():
            self.assertIs(target.cluster[neuron_name], original_module)
        for neuron_name, expected_marker in (
            ("neuron_4_1_1", 4.0),
            ("neuron_2_1_1", 2.0),
        ):
            parameter = target.cluster[neuron_name].nucleus.model.weight
            torch.testing.assert_close(
                target_optimizer.state[parameter]["exp_avg"],
                torch.full_like(parameter, expected_marker),
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
