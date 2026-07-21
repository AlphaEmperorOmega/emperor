import math
import os
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
import torch.nn as nn
from torch import Tensor

from emperor.config import ConfigBase, optional_field
from emperor.halting import (
    HaltingHiddenStateModeOptions,
    HaltingStateBase,
    SoftHaltingConfig,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import GatedResidualDynamicMemoryConfig
from emperor.memory._variants.gated_residual import GatedResidualDynamicMemory
from emperor.neuron import (
    Axons,
    AxonsConfig,
    Neuron,
    NeuronCluster,
    NeuronClusterConfig,
    NeuronClusterTrace,
    NeuronConfig,
    Nucleus,
    NucleusConfig,
    Terminal,
    TerminalConfig,
    TerminalConnectionShapeOptions,
    TerminalRangeOptions,
    TerminalZAxisOffsetOptions,
)
from emperor.nn import Module
from emperor.sampler import RouterConfig, SamplerConfig
from unit.test_memory import make_memory_config

ROUTER_CONFIG_UNSET = object()


@dataclass
class TestProjectionConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    scale: float | None = optional_field("Initial constant weight value.")

    def _registry_owner(self) -> type:
        return TestProjectionModel


class TestProjectionModel(Module):
    def __init__(
        self,
        cfg: TestProjectionConfig,
        overrides: TestProjectionConfig | None = None,
    ):
        super().__init__()
        self.cfg: TestProjectionConfig = self._override_config(cfg, overrides)
        self.weight = nn.Parameter(
            torch.full(
                (self.cfg.input_dim, self.cfg.output_dim),
                float(self.cfg.scale),
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weight


@dataclass
class ModeAwareProjectionConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")

    def _registry_owner(self) -> type:
        return ModeAwareProjectionModel


class ModeAwareProjectionModel(Module):
    def __init__(
        self,
        cfg: ModeAwareProjectionConfig,
        overrides: ModeAwareProjectionConfig | None = None,
    ):
        super().__init__()
        self.cfg: ModeAwareProjectionConfig = self._override_config(cfg, overrides)
        self.mode_multiplier = 1.0

    def train(self, mode: bool = True):
        super().train(mode)
        self.mode_multiplier = 1.0 if mode else -1.0
        return self

    def forward(self, input: Tensor) -> Tensor:
        return input * self.mode_multiplier


class DtypeObservingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.observed_dtype = self.weight.dtype

    def train(self, mode: bool = True):
        super().train(mode)
        self.observed_dtype = self.weight.dtype
        return self


@dataclass
class LifecycleProjectionConfig(TestProjectionConfig):
    fixture: str | None = optional_field(
        "Optional lifecycle role fixture installed by the projection model."
    )

    def _registry_owner(self) -> type:
        return LifecycleProjectionModel


class LifecycleProjectionModel(TestProjectionModel):
    def __init__(
        self,
        cfg: LifecycleProjectionConfig,
        overrides: LifecycleProjectionConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        reference = self.weight

        match self.cfg.fixture:
            case "context_markers":
                self.runtime_context_marker = nn.Parameter(torch.ones(()))
                self.register_buffer(
                    "runtime_context_buffer",
                    torch.ones(()),
                    persistent=True,
                )
            case "tied_context_parameters":
                shared_parameter = nn.Parameter(torch.zeros_like(reference))
                self.context_role_a = shared_parameter
                self.context_role_b = shared_parameter
            case "tied_context_buffers":
                shared_buffer = torch.zeros_like(reference)
                self.register_buffer("context_buffer_a", shared_buffer)
                self.register_buffer("context_buffer_b", shared_buffer)
            case "dtype_observer":
                self.dtype_observer = DtypeObservingModule()
            case "tied_mode_modules":
                shared_mode = nn.Dropout(p=0.1)
                self.mode_role_a = shared_mode
                self.mode_role_b = shared_mode
            case "tied_policy_parameters":
                shared_parameter = nn.Parameter(torch.zeros_like(reference))
                self.policy_role_a = shared_parameter
                self.policy_role_b = shared_parameter
            case "distinct_mitosis_parameters":
                self.mitosis_role_a = nn.Parameter(torch.zeros_like(reference))
                self.mitosis_role_b = nn.Parameter(torch.zeros_like(reference))
            case "tied_mitosis_parameters":
                shared_parameter = nn.Parameter(torch.zeros_like(reference))
                self.mitosis_role_a = shared_parameter
                self.mitosis_role_b = shared_parameter
            case "integer_mitosis_parameter":
                self.integer_mitosis_role = nn.Parameter(
                    torch.tensor([1, 3], dtype=torch.long),
                    requires_grad=False,
                )
            case None:
                pass
            case fixture:
                raise ValueError(f"Unknown lifecycle projection fixture: {fixture}")


@dataclass
class MixedTrainingModeNeuronConfig(NeuronConfig):
    def _registry_owner(self) -> type:
        return MixedTrainingModeNeuron


class MixedTrainingModeNeuron(Neuron):
    def __init__(
        self,
        cfg: MixedTrainingModeNeuronConfig,
        overrides: MixedTrainingModeNeuronConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.nucleus.eval()
        self.terminal.train()
        self.terminal.sampler.eval()


class ScriptedTerminal(nn.Module):
    def __init__(self, routes: list[list[int]]):
        super().__init__()
        self.register_buffer(
            "neuron_connections",
            torch.tensor(routes, dtype=torch.long),
            persistent=False,
        )


class ScriptedNeuron(nn.Module):
    def __init__(
        self,
        routes: list[list[int]],
        probabilities: list[float],
        delta: list[float],
        auxiliary_loss: float = 0.0,
    ):
        super().__init__()
        self.terminal = ScriptedTerminal(routes)
        self.register_buffer(
            "routes",
            torch.tensor(routes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "probabilities",
            torch.tensor(probabilities, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "delta",
            torch.tensor(delta, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "batch_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "atrophy_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "route_call_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        self.auxiliary_loss = auxiliary_loss

    def process_signal(self, input: Tensor) -> Tensor:
        self.batch_counter += 1
        return input + self.delta.to(device=input.device, dtype=input.dtype)

    def route_signal(self, processed_signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.route_call_counter += 1
        batch_size = processed_signal.shape[0]
        probabilities = self.probabilities.to(
            device=processed_signal.device,
            dtype=processed_signal.dtype,
        ).expand(batch_size, -1)
        routes = self.routes.to(device=processed_signal.device).expand(
            batch_size, -1, -1
        )
        return (
            probabilities,
            routes,
            processed_signal.new_tensor(self.auxiliary_loss),
        )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self.process_signal(input)
        probabilities, routes, auxiliary_loss = self.route_signal(output)
        return output, probabilities, routes, auxiliary_loss


class ScriptedSampler(nn.Module):
    def __init__(
        self,
        indices: list[int],
        probabilities: list[float],
        auxiliary_loss: float = 0.0,
    ):
        super().__init__()
        self.register_buffer(
            "indices",
            torch.tensor(indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "probabilities",
            torch.tensor(probabilities, dtype=torch.float32),
            persistent=False,
        )
        self.auxiliary_loss = auxiliary_loss

    def sample_probabilities_and_indices(
        self,
        input: Tensor,
    ) -> tuple[Tensor, Tensor, None, Tensor]:
        batch_size = input.shape[0]
        return (
            self.probabilities.to(device=input.device, dtype=input.dtype).expand(
                batch_size,
                -1,
            ),
            self.indices.to(device=input.device).expand(batch_size, -1),
            None,
            input.new_tensor(self.auxiliary_loss),
        )


class FourFieldOnlySampler(ScriptedSampler):
    def __getattr__(self, name: str):
        if "log_scores" in name or "router_scores" in name:
            raise AssertionError(f"score interface requested: {name}")
        return super().__getattr__(name)


class LearnableFourFieldSampler(nn.Module):
    def __init__(self, indices: list[int], probabilities: list[float]):
        super().__init__()
        self.register_buffer(
            "indices",
            torch.tensor(indices, dtype=torch.long),
            persistent=False,
        )
        self.probabilities = nn.Parameter(torch.tensor(probabilities))

    def sample_probabilities_and_indices(self, input: Tensor):
        batch_size = input.shape[0]
        return (
            self.probabilities.to(dtype=input.dtype).expand(batch_size, -1),
            self.indices.to(device=input.device).expand(batch_size, -1),
            None,
            input.new_zeros(()),
        )


@dataclass(kw_only=True, init=False)
class RecordingHaltingState(HaltingStateBase):
    pass


class RecordingHaltingModel(nn.Module):
    def __init__(
        self,
        halt_after_updates: int | None = None,
        finalize_offset: float = 0.0,
        ponder_loss: float = 0.0,
    ):
        super().__init__()
        self.halt_after_updates = halt_after_updates
        self.finalize_offset = finalize_offset
        self.ponder_loss = ponder_loss
        self.inputs: list[Tensor] = []
        self.update_count = 0
        self.finalize_count = 0

    def update_halting_state(
        self,
        previous_state: RecordingHaltingState | None,
        model_hidden_state: Tensor,
    ) -> tuple[RecordingHaltingState, Tensor]:
        self.update_count += 1
        valid_mask = torch.ones(
            model_hidden_state.shape[:-1],
            dtype=torch.bool,
            device=model_hidden_state.device,
        )
        previous_halt_mask = (
            torch.zeros(
                model_hidden_state.shape[:-1],
                dtype=torch.bool,
                device=model_hidden_state.device,
            )
            if previous_state is None
            else previous_state.halt_mask
        )
        self.inputs.append(model_hidden_state)
        previous_output = (
            model_hidden_state
            if previous_state is None
            else previous_state.output_hidden
        )
        output_hidden = torch.where(
            previous_halt_mask.unsqueeze(-1),
            previous_output,
            model_hidden_state,
        )
        halt_mask = previous_halt_mask.clone()
        if (
            self.halt_after_updates is not None
            and self.update_count >= self.halt_after_updates
        ):
            halt_mask |= valid_mask
        state = RecordingHaltingState(
            output_hidden=output_hidden,
            accumulated_hidden=output_hidden,
            continuation_probability=torch.ones(
                model_hidden_state.shape[:-1],
                dtype=model_hidden_state.dtype,
                device=model_hidden_state.device,
            ),
            halt_mask=halt_mask,
            valid_mask=valid_mask,
            stop_requested=bool((halt_mask | ~valid_mask).all().item()),
        )
        return state, output_hidden

    def finalize_weighted_accumulation(
        self,
        state: RecordingHaltingState,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.finalize_count += 1
        state.finalized = True
        return (
            state.output_hidden + self.finalize_offset,
            current_hidden.new_full(current_hidden.shape[:-1], self.ponder_loss),
        )


class NeuronTestCase(unittest.TestCase):
    batch_size = 3
    input_dim = 4

    def projection_config(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        scale: float = 1.0,
    ) -> TestProjectionConfig:
        return TestProjectionConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            scale=scale,
        )

    def lifecycle_projection_config(
        self,
        fixture: str,
    ) -> LifecycleProjectionConfig:
        return LifecycleProjectionConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
            scale=0.25,
            fixture=fixture,
        )

    def terminal_total_connections(
        self,
        xy_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
        z_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
    ) -> int:
        return (xy_axis_range.value * 2 + 1) ** 2 * (z_axis_range.value + 1)

    def router_config(
        self,
        input_dim: int,
        num_experts: int,
    ) -> RouterConfig:
        return RouterConfig(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=False,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=max(input_dim, num_experts),
                output_dim=num_experts,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(bias_flag=True),
                ),
            ),
        )

    def halting_config(
        self,
        input_dim: int = 4,
        threshold: float = 0.99,
    ) -> StickBreakingConfig:
        return StickBreakingConfig(
            input_dim=input_dim,
            threshold=threshold,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=input_dim,
                output_dim=2,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(bias_flag=True),
                ),
            ),
        )

    def sampler_config(
        self,
        input_dim: int = 4,
        num_experts: int | None = None,
        top_k: int = 2,
        router_config=ROUTER_CONFIG_UNSET,
    ) -> SamplerConfig:
        num_experts = num_experts or self.terminal_total_connections()
        if router_config is ROUTER_CONFIG_UNSET:
            router_config = self.router_config(input_dim, num_experts)
        return SamplerConfig(
            top_k=top_k,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=router_config,
        )

    def terminal_config(
        self,
        input_dim: int = 4,
        xy_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
        z_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
        z_axis_offset: TerminalZAxisOffsetOptions = TerminalZAxisOffsetOptions.ZERO,
        sampler_config: SamplerConfig | None = None,
        connection_shape: TerminalConnectionShapeOptions | None = None,
    ) -> TerminalConfig:
        num_experts = self.terminal_total_connections(xy_axis_range, z_axis_range)
        return TerminalConfig(
            input_dim=input_dim,
            x_axis_position=1,
            y_axis_position=1,
            z_axis_position=1,
            xy_axis_range=xy_axis_range,
            z_axis_range=z_axis_range,
            z_axis_offset=z_axis_offset,
            sampler_config=sampler_config
            or self.sampler_config(input_dim=input_dim, num_experts=num_experts),
            connection_shape=connection_shape,
        )

    def shaped_terminal(
        self,
        connection_shape: TerminalConnectionShapeOptions,
        num_experts: int,
        xy_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
        z_axis_range: TerminalRangeOptions = TerminalRangeOptions.ONE,
        z_axis_offset: TerminalZAxisOffsetOptions = TerminalZAxisOffsetOptions.ZERO,
    ):
        return self.terminal_config(
            xy_axis_range=xy_axis_range,
            z_axis_range=z_axis_range,
            z_axis_offset=z_axis_offset,
            sampler_config=self.sampler_config(num_experts=num_experts),
            connection_shape=connection_shape,
        ).build()

    def terminal_connection_set(self, terminal) -> set[tuple[int, int, int]]:
        return {
            tuple(connection_row)
            for connection_row in terminal.neuron_connections.tolist()
        }

    def neuron_config(
        self,
        coordinate_embedding_flag: bool | None = None,
    ) -> NeuronConfig:
        return NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=self.input_dim,
                    output_dim=self.input_dim,
                    scale=0.25,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(input_dim=self.input_dim),
            coordinate_embedding_flag=coordinate_embedding_flag,
        )

    def full_sampler_neuron_config(
        self,
        coordinate_embedding_flag: bool | None = None,
        model_config: ConfigBase | None = None,
    ) -> NeuronConfig:
        total_connections = self.terminal_total_connections()
        return NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=(
                    model_config
                    if model_config is not None
                    else self.projection_config(
                        input_dim=self.input_dim,
                        output_dim=self.input_dim,
                        scale=0.25,
                    )
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(
                input_dim=self.input_dim,
                sampler_config=self.sampler_config(
                    input_dim=self.input_dim,
                    num_experts=total_connections,
                    top_k=total_connections,
                ),
            ),
            coordinate_embedding_flag=coordinate_embedding_flag,
        )

    def expected_coordinate_embedding(self, x: int, y: int, z: int) -> Tensor:
        # input_dim=4 splits channels (2, 1, 1) across the x, y, z axes; the
        # frequency exponents are all zero at these channel counts, so the
        # encoding reduces to [sin(x), cos(x), sin(y), sin(z)].
        return torch.tensor(
            [math.sin(x), math.cos(x), math.sin(y), math.sin(z)],
            dtype=torch.float32,
        )


class TestNeuronConfigs(NeuronTestCase):
    def test_config_build_dispatches_to_runtime_classes(self):
        cases = [
            (NucleusConfig(model_config=self.projection_config()), Nucleus),
            (AxonsConfig(memory_config=None), Axons),
            (self.terminal_config(), Terminal),
            (self.neuron_config(), Neuron),
            (
                NeuronClusterConfig(
                    x_axis_total_neurons=1,
                    y_axis_total_neurons=1,
                    z_axis_total_neurons=1,
                    max_steps=1,
                    growth_threshold=None,
                    neuron_config=self.neuron_config(),
                ),
                NeuronCluster,
            ),
        ]

        for cfg, expected_type in cases:
            with self.subTest(expected_type=expected_type.__name__):
                self.assertIsInstance(cfg.build(), expected_type)

    def test_cluster_config_builds_with_halting(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            halting_config=self.halting_config(input_dim=self.input_dim),
            neuron_config=self.neuron_config(),
        ).build()

        self.assertIsInstance(model, NeuronCluster)
        self.assertIsNotNone(model.halting_model)

    def test_cluster_rejects_soft_halting_until_it_implements_the_interface(self):
        config = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=2,
            growth_threshold=None,
            halting_config=SoftHaltingConfig(
                input_dim=self.input_dim,
                threshold=0.999,
                dropout_probability=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            ),
            neuron_config=self.neuron_config(),
        )

        with self.assertRaisesRegex(ValueError, "does not implement"):
            config.build()

    def test_overrides_replace_config_fields(self):
        cfg = NucleusConfig(model_config=self.projection_config(scale=1.0))
        overrides = NucleusConfig(model_config=self.projection_config(scale=2.0))

        model = cfg.build(overrides)
        output = model(torch.ones(1, self.input_dim))

        torch.testing.assert_close(
            output,
            torch.full((1, self.input_dim), 8.0),
        )

    def test_cluster_overrides_are_applied_without_mutating_base_config(self):
        config = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            beam_width=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        )
        overrides = NeuronClusterConfig(max_steps=3, beam_width=2)

        model = config.build(overrides)

        self.assertEqual(model.max_steps, 3)
        self.assertEqual(model.beam_width, 2)
        self.assertEqual(config.max_steps, 1)
        self.assertEqual(config.beam_width, 1)


class TestNucleus(NeuronTestCase):
    def test_delegates_to_config_owned_model(self):
        model = NucleusConfig(
            model_config=self.projection_config(
                input_dim=2,
                output_dim=3,
                scale=0.5,
            )
        ).build()
        input_batch = torch.tensor([[1.0, 3.0]])

        output = model(input_batch)

        torch.testing.assert_close(output, torch.full((1, 3), 2.0))

    def test_gradient_flows_through_nucleus_model(self):
        model = NucleusConfig(
            model_config=self.projection_config(
                input_dim=self.input_dim,
                output_dim=2,
                scale=0.5,
            )
        ).build()
        input_batch = torch.ones(self.batch_size, self.input_dim)

        model(input_batch).sum().backward()

        self.assertIsNotNone(model.model.weight.grad)
        self.assertTrue(torch.any(model.model.weight.grad.abs() > 0.0))


class TestAxons(NeuronTestCase):
    def test_identity_path_preserves_input(self):
        model = AxonsConfig(memory_config=None).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output = model(input_batch)

        self.assertIs(output, input_batch)

    def test_builds_and_applies_dynamic_memory_config(self):
        memory_config = make_memory_config(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=self.input_dim,
            output_dim=self.input_dim + 2,
        )
        model = AxonsConfig(memory_config=memory_config).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        output = model(input_batch)

        self.assertIsInstance(model.memory_model, GatedResidualDynamicMemory)
        self.assertEqual(model.memory_model.input_dim, self.input_dim)
        self.assertEqual(model.memory_model.output_dim, self.input_dim)
        self.assertEqual(output.shape, input_batch.shape)
        self.assertFalse(torch.allclose(output, input_batch))

    def test_rejects_non_memory_config_base(self):
        with self.assertRaises(TypeError):
            AxonsConfig(memory_config=self.projection_config()).build()


class TestTerminal(NeuronTestCase):
    def test_initializes_connection_math(self):
        model = self.terminal_config().build()

        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.total_neuron_connections, 18)
        self.assertEqual(model.neuron_connections.shape, (18, 3))
        torch.testing.assert_close(
            model.neuron_connections[:4],
            torch.tensor(
                [
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 1, 1],
                    [0, 1, 2],
                ]
            ),
        )

    def test_forward_returns_selected_coordinates(self):
        model = self.terminal_config().build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        result = model(input_batch)
        self.assertEqual(len(result), 4)
        output, probabilities, selected_neurons, auxiliary_loss = result

        self.assertIs(output, input_batch)
        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertIsInstance(auxiliary_loss, Tensor)
        self.assertEqual(auxiliary_loss.shape, ())

        with self.assertRaises(TypeError):
            model(input_batch, return_log_probabilities=True)

    def test_forward_uses_only_four_field_sampler_interface(self):
        model = self.terminal_config().build()
        model.sampler = FourFieldOnlySampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        input_batch = torch.randn(self.batch_size, self.input_dim)

        routed_input, probabilities, selected_neurons, auxiliary_loss = model(
            input_batch
        )

        self.assertIs(routed_input, input_batch)
        torch.testing.assert_close(
            probabilities,
            torch.tensor([[0.25, 0.75]]).expand(self.batch_size, -1),
        )
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertEqual(auxiliary_loss.shape, ())

    def test_routerless_full_selection_follows_input_device(self) -> None:
        total_connections = self.terminal_total_connections()
        model = self.terminal_config(
            input_dim=total_connections,
            sampler_config=self.sampler_config(
                input_dim=total_connections,
                num_experts=total_connections,
                top_k=total_connections,
                router_config=None,
            ),
        ).build()
        meta_input = torch.empty(2, total_connections, device="meta")

        output, probabilities, selected_neurons, _ = model(meta_input)

        self.assertEqual(output.device, meta_input.device)
        self.assertEqual(probabilities.device, meta_input.device)
        self.assertEqual(selected_neurons.device, meta_input.device)
        self.assertEqual(selected_neurons.shape, (2, total_connections, 3))

    def test_sparse_forward_returns_matrix_shapes(self):
        total_connections = self.terminal_total_connections()
        model = self.terminal_config(
            sampler_config=self.sampler_config(
                input_dim=self.input_dim,
                num_experts=total_connections,
                top_k=1,
            )
        ).build()

        _, probabilities, selected_neurons, _ = model(
            torch.randn(self.batch_size, self.input_dim)
        )

        self.assertEqual(probabilities.shape, (self.batch_size, 1))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 1, 3))

    def test_full_forward_returns_all_coordinate_shapes(self):
        total_connections = self.terminal_total_connections()
        model = self.terminal_config(
            sampler_config=self.sampler_config(
                input_dim=self.input_dim,
                num_experts=total_connections,
                top_k=total_connections,
            )
        ).build()

        _, probabilities, selected_neurons, _ = model(
            torch.randn(self.batch_size, self.input_dim)
        )

        self.assertEqual(probabilities.shape, (self.batch_size, total_connections))
        self.assertEqual(
            selected_neurons.shape, (self.batch_size, total_connections, 3)
        )

    def test_logits_only_path_works_without_router(self):
        total_connections = self.terminal_total_connections()
        sampler_config = self.sampler_config(
            input_dim=total_connections,
            num_experts=total_connections,
            router_config=None,
        )
        model = self.terminal_config(
            input_dim=total_connections,
            sampler_config=sampler_config,
        ).build()

        _, probabilities, selected_neurons, auxiliary_loss = model(
            torch.randn(self.batch_size, total_connections)
        )

        self.assertEqual(probabilities.shape, (self.batch_size, 2))
        self.assertEqual(selected_neurons.shape, (self.batch_size, 2, 3))
        self.assertIsInstance(auxiliary_loss, Tensor)


class TestNeuron(NeuronTestCase):
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


class TestNeuronCluster(NeuronTestCase):
    def scripted_cluster(
        self,
        max_steps: int = 1,
        halting_model: nn.Module | None = None,
        input_dim: int = 1,
        x_axis_total_neurons: int = 4,
        y_axis_total_neurons: int = 1,
        z_axis_total_neurons: int = 1,
        initial_x_axis_total_neurons: int | None = None,
        initial_y_axis_total_neurons: int | None = None,
        initial_z_axis_total_neurons: int | None = None,
        beam_width: int | None = None,
    ) -> NeuronCluster:
        neuron_config = NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    scale=0.25,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(input_dim=input_dim),
        )
        model = NeuronClusterConfig(
            x_axis_total_neurons=x_axis_total_neurons,
            y_axis_total_neurons=y_axis_total_neurons,
            z_axis_total_neurons=z_axis_total_neurons,
            initial_x_axis_total_neurons=initial_x_axis_total_neurons,
            initial_y_axis_total_neurons=initial_y_axis_total_neurons,
            initial_z_axis_total_neurons=initial_z_axis_total_neurons,
            max_steps=max_steps,
            beam_width=beam_width,
            growth_threshold=None,
            neuron_config=neuron_config,
        ).build()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        model.halting_model = halting_model
        return model

    def test_routing_uses_only_four_field_sampler_interface(self):
        model = self.scripted_cluster(max_steps=1)
        model.entry_sampler = FourFieldOnlySampler(
            indices=[0],
            probabilities=[1.0],
        )

        output, auxiliary_loss = model(torch.zeros(1, model.input_dim))

        self.assertEqual(output.shape, (1, model.input_dim))
        self.assertEqual(auxiliary_loss.shape, ())

    def test_top_one_probability_uses_ordinary_product_gradients(self):
        model = self.scripted_cluster(
            max_steps=1,
            halting_model=RecordingHaltingModel(halt_after_updates=1),
        )
        sampler = LearnableFourFieldSampler(indices=[0], probabilities=[0.25])
        model.entry_sampler = sampler
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[2.0],
                )
            }
        )
        input_tensor = torch.tensor([[1.0]], requires_grad=True)

        output, _ = model(input_tensor)
        output.sum().backward()

        torch.testing.assert_close(output, torch.tensor([[0.75]]))
        torch.testing.assert_close(input_tensor.grad, torch.tensor([[0.25]]))
        torch.testing.assert_close(sampler.probabilities.grad, torch.tensor([3.0]))

    def test_scaling_sampler_probabilities_scales_output(self):
        outputs = []
        for probability_scale in (1.0, 0.4):
            model = self.scripted_cluster(
                max_steps=1,
                halting_model=RecordingHaltingModel(halt_after_updates=1),
                x_axis_total_neurons=2,
                initial_x_axis_total_neurons=2,
            )
            model.entry_sampler = ScriptedSampler(
                indices=[0, 1],
                probabilities=[
                    probability_scale * 0.25,
                    probability_scale * 0.75,
                ],
            )
            model.cluster = nn.ModuleDict(
                {
                    "neuron_1_1_1": ScriptedNeuron(
                        routes=[[1, 1, 1]],
                        probabilities=[1.0],
                        delta=[2.0],
                    ),
                    "neuron_2_1_1": ScriptedNeuron(
                        routes=[[2, 1, 1]],
                        probabilities=[1.0],
                        delta=[6.0],
                    ),
                }
            )
            output, _ = model(torch.zeros(1, 1))
            outputs.append(output)

        torch.testing.assert_close(outputs[1], outputs[0] * 0.4)

    def test_real_sampler_probabilities_match_direct_weighted_sum(self):
        input_tensor = torch.tensor([[2.0, 1.0, 0.0]])
        branch_deltas = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )

        for normalize_probabilities in (False, True):
            with self.subTest(normalize_probabilities=normalize_probabilities):
                model = self.scripted_cluster(
                    max_steps=1,
                    halting_model=RecordingHaltingModel(halt_after_updates=1),
                    input_dim=3,
                    x_axis_total_neurons=3,
                    initial_x_axis_total_neurons=3,
                )
                sampler_config = self.sampler_config(
                    input_dim=3,
                    num_experts=3,
                    top_k=2,
                    router_config=None,
                )
                sampler_config.normalize_probabilities_flag = normalize_probabilities
                model.entry_sampler = sampler_config.build().eval()
                model.cluster = nn.ModuleDict(
                    {
                        f"neuron_{index + 1}_1_1": ScriptedNeuron(
                            routes=[[index + 1, 1, 1]],
                            probabilities=[1.0],
                            delta=delta.tolist(),
                        )
                        for index, delta in enumerate(branch_deltas)
                    }
                )
                probabilities, indices, _, _ = (
                    model.entry_sampler.sample_probabilities_and_indices(input_tensor)
                )
                selected_branch_outputs = (
                    input_tensor.unsqueeze(1) + branch_deltas[indices]
                )
                expected_output = (
                    selected_branch_outputs * probabilities.unsqueeze(-1)
                ).sum(dim=1)

                output, _ = model(input_tensor)

                torch.testing.assert_close(output, expected_output)

    def test_neuron_and_experts_share_probability_weighting_contract(self):
        from emperor.experts._layers.reduce import MixtureOfExpertsReduce
        from unit.test_expert_behavioral_contracts import _mixture_config

        experts_model = MixtureOfExpertsReduce(
            _mixture_config(input_dim=1, output_dim=1)
        )
        with torch.no_grad():
            for expert_stack in experts_model.expert_modules:
                expert_stack[0].model.weight_params.fill_(1.0)
        branch_outputs = torch.tensor([[2.0], [6.0]])

        for probability_values in ([0.25, 0.75], [0.2, 0.3]):
            with self.subTest(probabilities=probability_values):
                probabilities = torch.tensor([probability_values])
                experts_output, _, _ = experts_model(
                    branch_outputs,
                    probabilities=probabilities,
                    indices=None,
                )
                neuron_model = self.scripted_cluster(
                    max_steps=1,
                    halting_model=RecordingHaltingModel(halt_after_updates=1),
                    x_axis_total_neurons=2,
                    initial_x_axis_total_neurons=2,
                )
                neuron_model.entry_sampler = ScriptedSampler(
                    indices=[0, 1],
                    probabilities=probability_values,
                )
                neuron_model.cluster = nn.ModuleDict(
                    {
                        "neuron_1_1_1": ScriptedNeuron(
                            routes=[[1, 1, 1]],
                            probabilities=[1.0],
                            delta=[2.0],
                        ),
                        "neuron_2_1_1": ScriptedNeuron(
                            routes=[[2, 1, 1]],
                            probabilities=[1.0],
                            delta=[6.0],
                        ),
                    }
                )
                neuron_output, _ = neuron_model(torch.zeros(1, 1))

                torch.testing.assert_close(neuron_output, experts_output)

    def test_initializes_expected_coordinate_keys(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        self.assertEqual(
            set(model.cluster.keys()),
            {"neuron_1_1_1", "neuron_2_1_1"},
        )

    def test_initial_neurons_preserve_builder_owned_mixed_training_modes(self):
        base_neuron_config = self.full_sampler_neuron_config()
        neuron_config = MixedTrainingModeNeuronConfig(
            nucleus_config=base_neuron_config.nucleus_config,
            axons_config=base_neuron_config.axons_config,
            terminal_config=base_neuron_config.terminal_config,
            coordinate_embedding_flag=base_neuron_config.coordinate_embedding_flag,
        )
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=neuron_config,
        ).build()

        neuron = model.cluster["neuron_1_1_1"]
        self.assertTrue(neuron.training)
        self.assertFalse(neuron.nucleus.training)
        self.assertFalse(neuron.nucleus.model.training)
        self.assertTrue(neuron.terminal.training)
        self.assertFalse(neuron.terminal.sampler.training)
        self.assertTrue(
            all(
                not module.training
                for name, module in neuron.terminal.sampler.named_modules(
                    remove_duplicate=False
                )
                if name
            )
        )

    def test_sparse_initial_grid_centers_within_xy_capacity(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=10,
            y_axis_total_neurons=10,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=2,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        expected_keys = {f"neuron_{x}_{y}_1" for x in range(5, 7) for y in range(5, 7)}
        self.assertEqual(
            set(model.cluster.keys()),
            expected_keys,
        )
        torch.testing.assert_close(
            model.entry_coordinates,
            torch.tensor(
                [
                    [5, 5, 1],
                    [5, 6, 1],
                    [6, 5, 1],
                    [6, 6, 1],
                ]
            ),
        )

    def test_sparse_initial_grid_centers_within_xyz_capacity(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=4,
            z_axis_total_neurons=3,
            initial_x_axis_total_neurons=3,
            initial_y_axis_total_neurons=2,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        expected_keys = {f"neuron_{x}_{y}_2" for x in range(2, 5) for y in range(2, 4)}
        self.assertEqual(set(model.cluster.keys()), expected_keys)
        torch.testing.assert_close(
            model.entry_coordinates,
            torch.tensor(
                [
                    [2, 2, 2],
                    [2, 3, 2],
                    [3, 2, 2],
                    [3, 3, 2],
                    [4, 2, 2],
                    [4, 3, 2],
                ]
            ),
        )

    def test_feature_last_inputs_restore_original_non_feature_shape(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        for shape in (
            (self.batch_size, self.input_dim),
            (2, 3, self.input_dim),
            (2, 2, 3, self.input_dim),
        ):
            with self.subTest(shape=shape):
                output, auxiliary_loss = model(torch.randn(*shape))

                self.assertEqual(output.shape, shape)
                self.assertEqual(auxiliary_loss.shape, ())

    def test_forward_returns_output_and_grows_at_threshold(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        output, auxiliary_loss = model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertIsInstance(auxiliary_loss, Tensor)
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertEqual(len(model.cluster), 2)
        self.assertIn("neuron_1_1_1", model.cluster)

    def test_grown_neuron_inherits_frozen_parameter_policy(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.requires_grad_(False)

        model(torch.randn(self.batch_size, self.input_dim))

        grown_neuron = model.cluster["neuron_2_1_1"]
        self.assertTrue(grown_neuron.training)
        self.assertTrue(tuple(grown_neuron.parameters()))
        self.assertTrue(
            all(not parameter.requires_grad for parameter in grown_neuron.parameters())
        )

    def test_grown_neuron_inherits_parent_role_modes_and_trainability(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.eval()
        parent.terminal.train()
        parent.terminal.sampler.eval()
        parent.nucleus.model.weight.requires_grad_(False)
        parent.terminal.sampler.router.model.layers[0].model.bias_params.requires_grad_(
            False
        )
        expected_training_modes = {
            name: module.training
            for name, module in parent.named_modules(remove_duplicate=False)
        }
        expected_trainability = {
            name: parameter.requires_grad
            for name, parameter in parent.named_parameters(remove_duplicate=False)
        }

        model(torch.randn(self.batch_size, self.input_dim))

        grown_neuron = model.cluster["neuron_2_1_1"]
        self.assertEqual(
            {
                name: module.training
                for name, module in grown_neuron.named_modules(remove_duplicate=False)
            },
            expected_training_modes,
        )
        self.assertEqual(
            {
                name: parameter.requires_grad
                for name, parameter in grown_neuron.named_parameters(
                    remove_duplicate=False
                )
            },
            expected_trainability,
        )

    def test_grown_neuron_inherits_parameter_and_buffer_context_by_role(self):
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("context_markers")
        )
        model = (
            self.growth_cluster_config(
                growth_threshold=1,
                neuron_config=neuron_config,
            )
            .build()
            .double()
        )
        parent = model.cluster["neuron_1_1_1"]
        parent_model = parent.nucleus.model
        parent_model.runtime_context_marker.data = (
            parent_model.runtime_context_marker.data.float()
        )
        parent_model.runtime_context_buffer = (
            parent_model.runtime_context_buffer.float()
        )
        expected_parameter_contexts = {
            name: (parameter.device, parameter.dtype)
            for name, parameter in parent.named_parameters(remove_duplicate=False)
        }
        expected_buffer_contexts = {
            name: (buffer.device, buffer.dtype)
            for name, buffer in parent.named_buffers(remove_duplicate=False)
        }

        model(
            torch.randn(
                self.batch_size,
                self.input_dim,
                dtype=torch.float64,
            )
        )

        grown = model.cluster["neuron_2_1_1"]
        self.assertEqual(
            {
                name: (parameter.device, parameter.dtype)
                for name, parameter in grown.named_parameters(remove_duplicate=False)
            },
            expected_parameter_contexts,
        )
        self.assertEqual(
            {
                name: (buffer.device, buffer.dtype)
                for name, buffer in grown.named_buffers(remove_duplicate=False)
            },
            expected_buffer_contexts,
        )

    def test_growth_rejects_conflicting_contexts_for_tied_child_parameters(
        self,
    ) -> None:
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("tied_context_parameters")
        )
        model = self.growth_cluster_config(
            growth_threshold=1,
            neuron_config=neuron_config,
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.context_role_b = nn.Parameter(
            parent.nucleus.model.context_role_a.detach().double()
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "grown parameter roles.*context_role_a.*context_role_b.*"
            "device or dtype contexts differ",
        ):
            model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(tuple(model.cluster), ("neuron_1_1_1",))

    def test_growth_rejects_conflicting_contexts_for_tied_child_buffers(
        self,
    ) -> None:
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("tied_context_buffers")
        )
        model = self.growth_cluster_config(
            growth_threshold=1,
            neuron_config=neuron_config,
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.context_buffer_b = (
            parent.nucleus.model.context_buffer_a.double()
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "grown buffer roles.*context_buffer_a.*context_buffer_b.*"
            "device or dtype contexts differ",
        ):
            model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(tuple(model.cluster), ("neuron_1_1_1",))

    def test_growth_applies_inherited_mode_through_module_train_contract(self):
        neuron_config = self.full_sampler_neuron_config()
        neuron_config.nucleus_config = NucleusConfig(
            model_config=ModeAwareProjectionConfig(input_dim=self.input_dim)
        )
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=neuron_config,
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.eval()

        model(torch.randn(self.batch_size, self.input_dim))

        grown_model = model.cluster["neuron_2_1_1"].nucleus.model
        self.assertFalse(grown_model.training)
        self.assertEqual(grown_model.mode_multiplier, -1.0)
        input_batch = torch.ones(self.batch_size, self.input_dim)
        torch.testing.assert_close(grown_model(input_batch), -input_batch)

    def test_growth_applies_context_before_custom_train_contract(self):
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("dtype_observer")
        )
        model = NeuronClusterConfig(
            x_axis_total_neurons=3,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=neuron_config,
        ).build()
        parent = model.cluster["neuron_2_1_1"]
        parent_observer = parent.nucleus.model.dtype_observer
        parent_observer.to(dtype=torch.float64)
        parent_observer.eval()
        parent.batch_counter.fill_(100)

        model(torch.randn(self.batch_size, self.input_dim))

        grown_observer = model.cluster["neuron_3_1_1"].nucleus.model.dtype_observer
        self.assertFalse(grown_observer.training)
        self.assertEqual(grown_observer.weight.dtype, torch.float64)
        self.assertEqual(grown_observer.observed_dtype, torch.float64)

    def test_empty_cluster_initialization_inherits_owner_device_and_dtype(self):
        model = self.growth_cluster_config(growth_threshold=1).build().double()
        owner_parameter = next(model.entry_sampler.parameters())
        model.cluster = nn.ModuleDict()

        initialized_neuron = model._initialize_neuron(2, 1, 1)

        floating_tensors = [
            *(
                parameter
                for parameter in initialized_neuron.parameters()
                if parameter.is_floating_point() or parameter.is_complex()
            ),
            *(
                buffer
                for buffer in initialized_neuron.buffers()
                if buffer.is_floating_point() or buffer.is_complex()
            ),
        ]
        self.assertTrue(floating_tensors)
        for tensor in floating_tensors:
            with self.subTest(shape=tuple(tensor.shape)):
                self.assertEqual(tensor.device, owner_parameter.device)
                self.assertEqual(tensor.dtype, owner_parameter.dtype)

    def test_empty_cluster_initialization_inherits_meta_owner_device(self):
        model = self.growth_cluster_config(growth_threshold=1).build().to("meta")
        model.cluster = nn.ModuleDict()

        initialized_neuron = model._initialize_neuron(2, 1, 1)

        initialized_tensors = [
            *initialized_neuron.parameters(),
            *initialized_neuron.buffers(),
        ]
        self.assertTrue(initialized_tensors)
        for tensor in initialized_tensors:
            with self.subTest(shape=tuple(tensor.shape), dtype=tensor.dtype):
                self.assertEqual(tensor.device, torch.device("meta"))

    def test_growth_rejects_conflicting_parent_policy_for_tied_child_modules(
        self,
    ) -> None:
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("tied_mode_modules")
        )
        model = self.growth_cluster_config(
            growth_threshold=1,
            neuron_config=neuron_config,
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.mode_role_b = nn.Dropout(p=0.1)
        parent.nucleus.model.mode_role_a.eval()
        parent.nucleus.model.mode_role_b.train()

        with self.assertRaisesRegex(
            RuntimeError,
            "grown module roles.*mode_role_a.*mode_role_b.*training modes differ",
        ):
            model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(tuple(model.cluster), ("neuron_1_1_1",))

    def test_growth_rejects_conflicting_parent_policy_for_tied_child_parameters(
        self,
    ) -> None:
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("tied_policy_parameters")
        )
        model = self.growth_cluster_config(
            growth_threshold=1,
            neuron_config=neuron_config,
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.policy_role_b = nn.Parameter(
            parent.nucleus.model.policy_role_a.detach().clone()
        )
        parent.nucleus.model.policy_role_a.requires_grad_(False)
        parent.nucleus.model.policy_role_b.requires_grad_(True)

        with self.assertRaisesRegex(
            RuntimeError,
            "grown parameter roles.*policy_role_a.*policy_role_b.*"
            "requires_grad policies differ",
        ):
            model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(tuple(model.cluster), ("neuron_1_1_1",))

    def test_cluster_neurons_receive_their_own_coordinate_embedding(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(coordinate_embedding_flag=True),
        ).build()

        torch.testing.assert_close(
            model.cluster["neuron_1_1_1"].coordinate_embedding,
            self.expected_coordinate_embedding(1, 1, 1),
        )
        torch.testing.assert_close(
            model.cluster["neuron_2_1_1"].coordinate_embedding,
            self.expected_coordinate_embedding(2, 1, 1),
        )

    def test_grown_neuron_receives_embedding_for_grown_coordinate(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(
                coordinate_embedding_flag=True
            ),
        ).build()

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(len(model.cluster), 2)
        self.assertIn("neuron_2_1_1", model.cluster)
        torch.testing.assert_close(
            model.cluster["neuron_2_1_1"].coordinate_embedding,
            self.expected_coordinate_embedding(2, 1, 1),
        )

    def test_growth_expands_from_centered_seed(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=10,
            y_axis_total_neurons=10,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=2,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        seed_keys = {f"neuron_{x}_{y}_1" for x in range(5, 7) for y in range(5, 7)}
        expected_growth_keys = {
            "neuron_4_5_1",
            "neuron_4_6_1",
            "neuron_5_4_1",
            "neuron_5_7_1",
            "neuron_6_4_1",
            "neuron_6_7_1",
            "neuron_7_5_1",
            "neuron_7_6_1",
        }
        self.assertEqual(set(model.cluster.keys()), seed_keys)

        model(torch.randn(self.batch_size, self.input_dim))

        cluster_keys = set(model.cluster.keys())
        added_keys = cluster_keys - seed_keys
        self.assertTrue(seed_keys.issubset(cluster_keys))
        self.assertEqual(len(cluster_keys), 5)
        self.assertEqual(len(added_keys), 1)
        self.assertTrue(added_keys.issubset(expected_growth_keys))

    def test_propagation_fires_deeper_layer_branch(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=2,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        output, _ = model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 2)
        self.assertEqual(int(model.cluster["neuron_1_1_2"].batch_counter.item()), 1)

    def test_growth_prefers_neuron_with_highest_counter(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=4,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=100,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        self.assertEqual(
            set(model.cluster.keys()),
            {"neuron_2_1_1", "neuron_3_1_1"},
        )
        model.cluster["neuron_2_1_1"].batch_counter.fill_(100)
        model.cluster["neuron_3_1_1"].batch_counter.fill_(200)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_4_1_1", model.cluster)
        self.assertNotIn("neuron_1_1_1", model.cluster)
        self.assertEqual(
            int(model.cluster["neuron_3_1_1"].batch_counter.item()),
            0,
        )

    def escape_growth_cluster(self) -> NeuronCluster:
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            escape_driven_growth_flag=True,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

    def three_dimensional_growth_cluster(
        self,
        *,
        escape_driven_growth_flag: bool,
    ) -> NeuronCluster:
        model = NeuronClusterConfig(
            x_axis_total_neurons=3,
            y_axis_total_neurons=7,
            z_axis_total_neurons=9,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            escape_driven_growth_flag=escape_driven_growth_flag,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        model.cluster = nn.ModuleDict(
            {
                "neuron_2_4_5": ScriptedNeuron(
                    routes=[[1, 2, 4], [3, 5, 5]],
                    probabilities=[0.5, 0.5],
                    delta=[0.0] * self.input_dim,
                )
            }
        )
        return model

    def test_three_dimensional_growth_uses_true_manhattan_distance(self) -> None:
        model = self.three_dimensional_growth_cluster(escape_driven_growth_flag=False)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_3_5_5", model.cluster)
        self.assertNotIn("neuron_1_2_4", model.cluster)
        grown_terminal = model.cluster["neuron_3_5_5"].terminal
        self.assertEqual(
            (
                grown_terminal.x_axis_position,
                grown_terminal.y_axis_position,
                grown_terminal.z_axis_position,
            ),
            (3, 5, 5),
        )

    def test_three_dimensional_escape_signal_overrides_distance_and_resets(
        self,
    ) -> None:
        model = self.three_dimensional_growth_cluster(escape_driven_growth_flag=True)
        model.escape_counts[0, 1, 3] = 5

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_1_2_4", model.cluster)
        self.assertNotIn("neuron_3_5_5", model.cluster)
        self.assertEqual(int(model.escape_counts[0, 1, 3]), 0)
        grown_terminal = model.cluster["neuron_1_2_4"].terminal
        self.assertEqual(
            (
                grown_terminal.x_axis_position,
                grown_terminal.y_axis_position,
                grown_terminal.z_axis_position,
            ),
            (1, 2, 4),
        )

    def test_escape_driven_growth_targets_most_escaped_coordinate(self):
        model = self.escape_growth_cluster()
        self.assertEqual(set(model.cluster.keys()), {"neuron_3_1_1"})
        model.escape_counts[3, 0, 0] = 10_000

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_4_1_1", model.cluster)
        self.assertNotIn("neuron_2_1_1", model.cluster)
        self.assertEqual(int(model.escape_counts[3, 0, 0].item()), 0)

    def test_escape_driven_growth_without_planted_signal_keeps_closest(self):
        model = self.escape_growth_cluster()

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_2_1_1", model.cluster)
        self.assertNotIn("neuron_4_1_1", model.cluster)

    def test_zero_escape_counts_use_stable_manhattan_growth_fallback(self):
        model = self.escape_growth_cluster()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        parent = ScriptedNeuron(
            routes=[[3, 1, 1]],
            probabilities=[1.0],
            delta=[0.0] * self.input_dim,
        )
        parent.terminal.neuron_connections = torch.tensor(
            [[2, 1, 1], [3, 1, 1], [4, 1, 1]],
            dtype=torch.long,
        )
        model.cluster = nn.ModuleDict({"neuron_3_1_1": parent})

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_2_1_1", model.cluster)
        self.assertNotIn("neuron_4_1_1", model.cluster)
        torch.testing.assert_close(
            model.escape_counts,
            torch.zeros_like(model.escape_counts),
        )

    def test_mitosis_initialization_copies_parent_parameters(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            mitosis_initialization_flag=True,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.weight.data.fill_(0.7)

        model(torch.randn(self.batch_size, self.input_dim))

        child = model.cluster["neuron_2_1_1"]
        torch.testing.assert_close(
            child.nucleus.model.weight,
            torch.full_like(parent.nucleus.model.weight, 0.7),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(int(child.batch_counter.item()), 0)
        for child_param, parent_param in zip(
            child.parameters(),
            parent.parameters(),
            strict=True,
        ):
            with self.subTest(shape=tuple(child_param.shape)):
                parent_std = float(parent_param.detach().float().std(correction=0))
                max_difference = float((child_param - parent_param).abs().max())
                if parent_std > 1e-6:
                    self.assertGreater(max_difference, 0.0)
                    self.assertLess(max_difference, 0.1 * parent_std)
                else:
                    self.assertEqual(max_difference, 0.0)

    def test_mitosis_perturbs_float64_parent_using_float64_variance(self):
        torch.manual_seed(20260719)
        model = (
            NeuronClusterConfig(
                x_axis_total_neurons=2,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                initial_x_axis_total_neurons=1,
                initial_y_axis_total_neurons=1,
                initial_z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=1,
                mitosis_initialization_flag=True,
                neuron_config=self.full_sampler_neuron_config(),
            )
            .build()
            .double()
        )
        parent_weight = model.cluster["neuron_1_1_1"].nucleus.model.weight
        parent_values = (
            torch.arange(
                parent_weight.numel(),
                dtype=torch.float64,
                device=parent_weight.device,
            )
            .remainder(2)
            .mul(2.0)
            .add(100_000_000.0)
            .reshape_as(parent_weight)
        )
        with torch.no_grad():
            parent_weight.copy_(parent_values)
        parent_population_std = float(parent_weight.detach().std(correction=0).item())

        torch.manual_seed(20260719)
        model(
            torch.zeros(
                self.batch_size,
                self.input_dim,
                dtype=torch.float64,
                device=parent_weight.device,
            )
        )

        child_weight = model.cluster["neuron_2_1_1"].nucleus.model.weight
        maximum_difference = float((child_weight - parent_weight).abs().max().item())
        self.assertEqual(parent_population_std, 1.0)
        self.assertGreater(maximum_difference, 0.0)
        self.assertLess(maximum_difference, 0.1 * parent_population_std)

    def test_mitosis_copies_tied_parent_parameter_roles_by_name(self):
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("distinct_mitosis_parameters")
        )
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            mitosis_initialization_flag=True,
            neuron_config=neuron_config,
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        shared_parent_parameter = nn.Parameter(
            torch.arange(
                parent.nucleus.model.weight.numel(),
                dtype=parent.nucleus.model.weight.dtype,
                device=parent.nucleus.model.weight.device,
            ).reshape_as(parent.nucleus.model.weight),
            requires_grad=False,
        )
        parent.nucleus.model.mitosis_role_a = shared_parent_parameter
        parent.nucleus.model.mitosis_role_b = shared_parent_parameter

        torch.manual_seed(20260718)
        model(torch.randn(self.batch_size, self.input_dim))

        child = model.cluster["neuron_2_1_1"]
        child_roles = (
            child.nucleus.model.mitosis_role_a,
            child.nucleus.model.mitosis_role_b,
        )
        self.assertIsNot(child_roles[0], child_roles[1])
        for child_parameter in child_roles:
            self.assertFalse(child_parameter.requires_grad)
            maximum_difference = float(
                (child_parameter - shared_parent_parameter).abs().max()
            )
            parent_std = float(
                shared_parent_parameter.detach().float().std(correction=0)
            )
            self.assertGreater(maximum_difference, 0.0)
            self.assertLess(maximum_difference, 0.1 * parent_std)

    def test_mitosis_rejects_mismatched_parent_and_child_parameter_roles(self):
        config = self.growth_cluster_config(growth_threshold=1)
        config.mitosis_initialization_flag = True
        model = config.build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.parent_only_mitosis_role = nn.Parameter(
            torch.zeros_like(parent.nucleus.model.weight)
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Mitosis initialization requires the grown and parent neurons "
            "to expose the same parameter roles",
        ):
            model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(tuple(model.cluster), ("neuron_1_1_1",))

    def test_mitosis_rejects_distinct_parent_roles_tied_in_child(self):
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("tied_mitosis_parameters")
        )
        config = self.growth_cluster_config(
            growth_threshold=1,
            neuron_config=neuron_config,
        )
        config.mitosis_initialization_flag = True
        model = config.build()
        parent = model.cluster["neuron_1_1_1"]
        reference = parent.nucleus.model.weight
        parent.nucleus.model.mitosis_role_a = nn.Parameter(torch.zeros_like(reference))
        parent.nucleus.model.mitosis_role_b = nn.Parameter(torch.ones_like(reference))

        with self.assertRaisesRegex(
            RuntimeError,
            "Mitosis initialization cannot copy distinct parent parameter "
            "roles into one tied grown parameter",
        ):
            model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(tuple(model.cluster), ("neuron_1_1_1",))

    def test_mitosis_preserves_matching_tied_parent_and_child_roles(self):
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("tied_mitosis_parameters")
        )
        config = self.growth_cluster_config(
            growth_threshold=1,
            neuron_config=neuron_config,
        )
        config.mitosis_initialization_flag = True
        model = config.build()
        parent = model.cluster["neuron_1_1_1"]
        parent_parameter = parent.nucleus.model.mitosis_role_a
        parent_parameter.data.copy_(
            torch.arange(
                parent_parameter.numel(),
                dtype=parent_parameter.dtype,
                device=parent_parameter.device,
            ).reshape_as(parent_parameter)
        )

        model(torch.randn(self.batch_size, self.input_dim))

        child = model.cluster["neuron_2_1_1"]
        self.assertIs(
            child.nucleus.model.mitosis_role_a,
            child.nucleus.model.mitosis_role_b,
        )
        self.assertIs(
            parent.nucleus.model.mitosis_role_a,
            parent.nucleus.model.mitosis_role_b,
        )
        self.assertFalse(
            torch.equal(
                child.nucleus.model.mitosis_role_a,
                torch.zeros_like(child.nucleus.model.mitosis_role_a),
            )
        )

    def test_mitosis_copies_non_floating_parameter_without_perturbation(self):
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("integer_mitosis_parameter")
        )
        config = self.growth_cluster_config(
            growth_threshold=1,
            neuron_config=neuron_config,
        )
        config.mitosis_initialization_flag = True
        model = config.build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.integer_mitosis_role.copy_(
            torch.tensor([2, 7], dtype=torch.long)
        )

        model(torch.randn(self.batch_size, self.input_dim))

        child_parameter = model.cluster[
            "neuron_2_1_1"
        ].nucleus.model.integer_mitosis_role
        self.assertEqual(child_parameter.dtype, torch.long)
        self.assertFalse(child_parameter.requires_grad)
        torch.testing.assert_close(
            child_parameter,
            torch.tensor([2, 7], dtype=torch.long),
        )

    def test_mitosis_flag_disabled_uses_fresh_initialization(self):
        model = self.growth_cluster_config(growth_threshold=1).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.nucleus.model.weight.data.fill_(0.7)

        model(torch.randn(self.batch_size, self.input_dim))

        child = model.cluster["neuron_2_1_1"]
        torch.testing.assert_close(
            child.nucleus.model.weight,
            torch.full_like(parent.nucleus.model.weight, 0.25),
            rtol=0.0,
            atol=0.0,
        )

    def test_growth_only_targets_positive_coordinates(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        for _ in range(3):
            model(torch.randn(self.batch_size, self.input_dim))

        for neuron_name in model.cluster:
            _, x, y, z = neuron_name.split("_")
            self.assertTrue(1 <= int(x) <= 2)
            self.assertTrue(1 <= int(y) <= 1)
            self.assertTrue(1 <= int(z) <= 1)

    def test_eval_forward_does_not_grow_at_threshold(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.eval()

        with torch.no_grad():
            output, auxiliary_loss = model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertEqual(len(model.cluster), 1)

    def test_eval_forward_does_not_increment_batch_counters(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.eval()

        with torch.no_grad():
            model(torch.randn(self.batch_size, self.input_dim))

        for neuron_name, neuron in model.cluster.items():
            with self.subTest(neuron_name=neuron_name):
                self.assertEqual(int(neuron.batch_counter.item()), 0)

    def growth_cluster_config(
        self,
        growth_threshold: int | None,
        growth_cooldown_steps: int | None = None,
        max_total_growths: int | None = None,
        x_axis_total_neurons: int = 2,
        neuron_config: NeuronConfig | None = None,
    ) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=x_axis_total_neurons,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=growth_threshold,
            growth_cooldown_steps=growth_cooldown_steps,
            max_total_growths=max_total_growths,
            neuron_config=(
                neuron_config
                if neuron_config is not None
                else self.full_sampler_neuron_config()
            ),
        )

    def test_growth_cooldown_blocks_growth_until_elapsed(self):
        model = self.growth_cluster_config(
            growth_threshold=1,
            growth_cooldown_steps=2,
            x_axis_total_neurons=4,
        ).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        for forward_index, expected_size in enumerate([1, 2, 2, 3]):
            with self.subTest(forward_index=forward_index):
                model(input_batch)
                self.assertEqual(len(model.cluster), expected_size)

    def test_successful_growth_resets_cooldown_counter(self):
        model = self.growth_cluster_config(
            growth_threshold=1,
            growth_cooldown_steps=2,
            x_axis_total_neurons=4,
        ).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        model(input_batch)
        self.assertEqual(int(model.forwards_since_last_growth.item()), 1)

        model(input_batch)
        self.assertEqual(int(model.forwards_since_last_growth.item()), 0)

    def test_failed_growth_preserves_saturated_parent_counter_until_retry(self):
        neuron_config = self.full_sampler_neuron_config(
            model_config=self.lifecycle_projection_config("tied_context_parameters")
        )
        model = self.growth_cluster_config(
            growth_threshold=1,
            x_axis_total_neurons=2,
            neuron_config=neuron_config,
        ).build()
        parent = model.cluster["neuron_1_1_1"]
        parent.batch_counter.fill_(100)
        parent.nucleus.model.context_role_b = nn.Parameter(
            parent.nucleus.model.context_role_a.detach().double()
        )
        input_batch = torch.randn(self.batch_size, self.input_dim)

        with self.assertRaisesRegex(
            RuntimeError,
            "grown parameter roles.*context_role_a.*context_role_b.*"
            "device or dtype contexts differ",
        ):
            model(input_batch)

        self.assertGreaterEqual(int(parent.batch_counter.item()), 100)
        self.assertEqual(tuple(model.cluster), ("neuron_1_1_1",))

        parent.nucleus.model.context_role_b = parent.nucleus.model.context_role_a
        model(input_batch)

        self.assertIn("neuron_2_1_1", model.cluster)
        self.assertEqual(int(parent.batch_counter.item()), 0)

    def test_max_total_growths_caps_lifetime_growth(self):
        model = self.growth_cluster_config(
            growth_threshold=1,
            max_total_growths=1,
            x_axis_total_neurons=4,
        ).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        model(input_batch)
        self.assertEqual(len(model.cluster), 2)
        self.assertEqual(int(model.total_growth_count.item()), 1)

        for _ in range(3):
            model(input_batch)

        self.assertEqual(len(model.cluster), 2)
        self.assertEqual(int(model.total_growth_count.item()), 1)

    def test_growth_budget_buffers_disabled_when_options_none(self):
        model = self.growth_cluster_config(growth_threshold=1).build()

        self.assertIsNone(model.forwards_since_last_growth)
        self.assertIsNone(model.total_growth_count)
        self.assertNotIn("forwards_since_last_growth", model.state_dict())
        self.assertNotIn("total_growth_count", model.state_dict())

    def test_escape_counts_persist_through_strict_state_dict_round_trip(self):
        source_model = self.escape_growth_cluster()
        expected_counts = torch.arange(
            source_model.escape_counts.numel(),
            dtype=source_model.escape_counts.dtype,
        ).reshape_as(source_model.escape_counts)
        source_model.escape_counts.copy_(expected_counts)
        state_dict = source_model.state_dict()
        target_model = self.escape_growth_cluster()

        incompatible_keys = target_model.load_state_dict(state_dict, strict=True)

        self.assertIn("escape_counts", state_dict)
        self.assertEqual(incompatible_keys.missing_keys, [])
        self.assertEqual(incompatible_keys.unexpected_keys, [])
        torch.testing.assert_close(target_model.escape_counts, expected_counts)

    def test_load_state_dict_restores_growth_budget_buffers(self):
        source_model = self.growth_cluster_config(
            growth_threshold=1,
            growth_cooldown_steps=1,
            max_total_growths=2,
            x_axis_total_neurons=4,
        ).build()
        input_batch = torch.randn(self.batch_size, self.input_dim)
        source_model(input_batch)
        source_model(input_batch)
        self.assertEqual(int(source_model.total_growth_count.item()), 2)

        target_model = self.growth_cluster_config(
            growth_threshold=1,
            growth_cooldown_steps=1,
            max_total_growths=2,
            x_axis_total_neurons=4,
        ).build()
        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            int(target_model.forwards_since_last_growth.item()),
            int(source_model.forwards_since_last_growth.item()),
        )
        self.assertEqual(int(target_model.total_growth_count.item()), 2)

    def test_load_legacy_state_dict_seeds_growth_budget_buffers(self):
        legacy_model = self.growth_cluster_config(growth_threshold=1).build()
        target_model = self.growth_cluster_config(
            growth_threshold=1,
            growth_cooldown_steps=2,
            max_total_growths=3,
        ).build()
        target_model.forwards_since_last_growth.fill_(7)
        target_model.total_growth_count.fill_(7)

        target_model.load_state_dict(legacy_model.state_dict(), strict=True)

        self.assertEqual(int(target_model.forwards_since_last_growth.item()), 0)
        self.assertEqual(int(target_model.total_growth_count.item()), 0)

    def test_distributed_growth_rejects_missing_forward_topology_baseline(self):
        model = self.growth_cluster_config(growth_threshold=1).build()
        model._growth_counters_are_global = True

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            self.assertRaisesRegex(
                RuntimeError,
                "Distributed Neuron growth topology changed during a forward pass",
            ),
        ):
            model._check_neuron_growth(None)

    def test_public_forward_passes_captured_growth_baseline_to_growth_check(self):
        model = self.growth_cluster_config(growth_threshold=1).build()
        captured_baseline = object()
        input_batch = torch.randn(self.batch_size, self.input_dim)

        with (
            patch.object(
                model,
                "_capture_growth_counter_baseline",
                return_value=captured_baseline,
            ) as capture_baseline,
            patch.object(model, "_check_neuron_growth") as check_growth,
        ):
            model(input_batch)

        capture_baseline.assert_called_once_with()
        check_growth.assert_called_once_with(captured_baseline)

    def test_distributed_growth_rejects_topology_changed_during_forward(self):
        model = self.growth_cluster_config(
            growth_threshold=1,
            x_axis_total_neurons=3,
        ).build()
        model._growth_counters_are_global = True
        baseline = model._capture_growth_counter_baseline()
        model.cluster["neuron_1_1_1"] = model._initialize_neuron(1, 1, 1)

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            self.assertRaisesRegex(
                RuntimeError,
                "Distributed Neuron growth topology changed during a forward pass",
            ),
        ):
            model._check_neuron_growth(baseline)

    def test_distributed_growth_rejects_missing_escape_count_baseline(self):
        model = self.escape_growth_cluster()
        model._growth_counters_are_global = True
        captured_baseline = model._capture_growth_counter_baseline()
        self.assertIsNotNone(captured_baseline)
        baseline_without_escape_counts = type(captured_baseline)(
            neuron_names=captured_baseline.neuron_names,
            batch_counters=captured_baseline.batch_counters,
            escape_counts=None,
        )
        synchronize_escape_counts = (
            model._NeuronClusterPlasticityMixin__synchronize_escape_counts_across_ranks
        )

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            self.assertRaisesRegex(
                RuntimeError,
                "Distributed Neuron escape-count state changed during a forward pass",
            ),
        ):
            synchronize_escape_counts(baseline_without_escape_counts)

    def test_distributed_escape_counts_advance_from_captured_global_baseline(self):
        model = self.escape_growth_cluster()
        model._growth_counters_are_global = True
        baseline = model._capture_growth_counter_baseline()
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(baseline.escape_counts)
        model.escape_counts.add_(torch.ones_like(model.escape_counts))
        remote_escape_count_contribution = torch.full_like(model.escape_counts, 2)
        expected_escape_counts = (
            model.escape_counts.clone() + remote_escape_count_contribution
        )
        synchronize_escape_counts = (
            model._NeuronClusterPlasticityMixin__synchronize_escape_counts_across_ranks
        )

        def add_remote_escape_count_contribution(
            tensor: Tensor,
            op: torch.distributed.ReduceOp,
        ) -> None:
            self.assertIs(op, torch.distributed.ReduceOp.SUM)
            tensor.add_(remote_escape_count_contribution)

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "torch.distributed.all_reduce",
                side_effect=add_remote_escape_count_contribution,
            ),
        ):
            synchronized_escape_counts = synchronize_escape_counts(baseline)

        torch.testing.assert_close(
            synchronized_escape_counts,
            expected_escape_counts,
        )
        torch.testing.assert_close(model.escape_counts, expected_escape_counts)

    @unittest.skipUnless(
        torch.distributed.is_available() and torch.distributed.is_gloo_available(),
        "gloo process group support is required",
    )
    def test_growth_proceeds_under_initialized_process_group(self):
        model = self.growth_cluster_config(growth_threshold=1).build()

        with tempfile.TemporaryDirectory() as temp_dir:
            init_file = os.path.join(temp_dir, "process_group_init")
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=f"file://{init_file}",
                rank=0,
                world_size=1,
            )
            try:
                output, _ = model(torch.randn(self.batch_size, self.input_dim))
            finally:
                torch.distributed.destroy_process_group()

        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        self.assertIn("neuron_2_1_1", model.cluster)

    def test_distributed_training_without_growth_does_not_raise(self):
        eval_model = self.growth_cluster_config(growth_threshold=1).build()
        eval_model.eval()
        growthless_model = self.growth_cluster_config(growth_threshold=None).build()

        with patch("torch.distributed.is_initialized", return_value=True):
            with self.subTest(case="eval_forward"), torch.no_grad():
                output, _ = eval_model(torch.randn(self.batch_size, self.input_dim))
                self.assertEqual(output.shape, (self.batch_size, self.input_dim))
            with self.subTest(case="growth_disabled"):
                output, _ = growthless_model(
                    torch.randn(self.batch_size, self.input_dim)
                )
                self.assertEqual(output.shape, (self.batch_size, self.input_dim))

    def test_load_state_dict_rebuilds_grown_neurons(self):
        source_model = self.growth_cluster_config(growth_threshold=1).build()
        source_model(torch.randn(self.batch_size, self.input_dim))
        self.assertEqual(len(source_model.cluster), 2)

        target_model = self.growth_cluster_config(growth_threshold=1).build()
        self.assertEqual(len(target_model.cluster), 1)

        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            set(target_model.cluster.keys()),
            set(source_model.cluster.keys()),
        )
        for neuron_name, neuron in source_model.cluster.items():
            with self.subTest(neuron_name=neuron_name):
                self.assertEqual(
                    int(target_model.cluster[neuron_name].batch_counter.item()),
                    int(neuron.batch_counter.item()),
                )

        source_model.eval()
        target_model.eval()
        input_batch = torch.randn(self.batch_size, self.input_dim)
        with torch.no_grad():
            source_output, _ = source_model(input_batch)
            target_output, _ = target_model(input_batch)
        torch.testing.assert_close(target_output, source_output)

    def test_load_state_dict_without_grown_neurons_is_unchanged(self):
        source_model = self.growth_cluster_config(growth_threshold=None).build()
        target_model = self.growth_cluster_config(growth_threshold=None).build()

        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            set(target_model.cluster.keys()),
            set(source_model.cluster.keys()),
        )

    def pruning_cluster_model(
        self,
        pruning_threshold: int | None,
        growth_threshold: int | None = None,
    ) -> "NeuronCluster":
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=growth_threshold,
            pruning_threshold=pruning_threshold,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

    def plant_isolated_neuron(self, model, x_coordinate: int) -> str:
        """Adds a real neuron outside the entry neuron's terminal range
        (entry neuron_3_1_1 only reaches x in {2, 3, 4}), so routing never
        process-calls it and its atrophy counter accrues every forward."""
        name = f"neuron_{x_coordinate}_1_1"
        model.cluster[name] = model._initialize_neuron(x_coordinate, 1, 1)
        return name

    def test_forward_prunes_idle_neuron_at_threshold(self):
        model = self.pruning_cluster_model(pruning_threshold=2)
        planted_name = self.plant_isolated_neuron(model, 5)
        input_batch = torch.randn(self.batch_size, self.input_dim)

        model(input_batch)
        self.assertIn(planted_name, model.cluster)
        self.assertEqual(
            int(model.cluster[planted_name].atrophy_counter.item()),
            1,
        )

        model(input_batch)
        self.assertNotIn(planted_name, model.cluster)
        self.assertIn("neuron_3_1_1", model.cluster)

    def test_used_neuron_atrophy_counter_resets_to_zero(self):
        model = self.pruning_cluster_model(pruning_threshold=10)
        planted_name = self.plant_isolated_neuron(model, 5)
        model.cluster["neuron_3_1_1"].atrophy_counter.fill_(5)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(
            int(model.cluster["neuron_3_1_1"].atrophy_counter.item()),
            0,
        )
        self.assertEqual(
            int(model.cluster[planted_name].atrophy_counter.item()),
            1,
        )

    def test_prune_prefers_highest_atrophy_counter(self):
        model = self.pruning_cluster_model(pruning_threshold=2)
        lower_name = self.plant_isolated_neuron(model, 1)
        higher_name = self.plant_isolated_neuron(model, 5)
        model.cluster[lower_name].atrophy_counter.fill_(3)
        model.cluster[higher_name].atrophy_counter.fill_(10)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertNotIn(higher_name, model.cluster)
        self.assertIn(lower_name, model.cluster)

    def test_prune_tie_break_is_sorted_name(self):
        model = self.pruning_cluster_model(pruning_threshold=2)
        first_name = self.plant_isolated_neuron(model, 1)
        second_name = self.plant_isolated_neuron(model, 5)
        model.cluster[first_name].atrophy_counter.fill_(5)
        model.cluster[second_name].atrophy_counter.fill_(5)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertNotIn(first_name, model.cluster)
        self.assertIn(second_name, model.cluster)

    def test_at_most_one_prune_per_forward(self):
        model = self.pruning_cluster_model(pruning_threshold=2)
        self.plant_isolated_neuron(model, 1)
        self.plant_isolated_neuron(model, 5)
        model.cluster["neuron_1_1_1"].atrophy_counter.fill_(10)
        model.cluster["neuron_5_1_1"].atrophy_counter.fill_(10)
        input_batch = torch.randn(self.batch_size, self.input_dim)

        model(input_batch)
        self.assertEqual(len(model.cluster), 2)

        model(input_batch)
        self.assertEqual(set(model.cluster.keys()), {"neuron_3_1_1"})

    def test_entry_plane_neurons_are_never_pruned(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            pruning_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        model.cluster = nn.ModuleDict(
            {
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 0.0, 0.0, 0.0],
                ),
                "neuron_3_1_1": ScriptedNeuron(
                    routes=[[3, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 0.0, 0.0, 0.0],
                ),
            }
        )
        input_batch = torch.randn(self.batch_size, self.input_dim)

        for _ in range(3):
            model(input_batch)

        self.assertEqual(
            set(model.cluster.keys()),
            {"neuron_2_1_1", "neuron_3_1_1"},
        )
        self.assertGreaterEqual(
            int(model.cluster["neuron_3_1_1"].atrophy_counter.item()),
            1,
        )

    def test_eval_forward_does_not_prune(self):
        model = self.pruning_cluster_model(pruning_threshold=1)
        planted_name = self.plant_isolated_neuron(model, 5)
        model.cluster[planted_name].atrophy_counter.fill_(99)
        model.eval()

        with torch.no_grad():
            model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn(planted_name, model.cluster)
        self.assertEqual(
            int(model.cluster[planted_name].atrophy_counter.item()),
            99,
        )

    def test_pruning_disabled_when_threshold_none(self):
        model = self.pruning_cluster_model(pruning_threshold=None)
        planted_name = self.plant_isolated_neuron(model, 5)
        input_batch = torch.randn(self.batch_size, self.input_dim)

        for _ in range(3):
            model(input_batch)

        self.assertIn(planted_name, model.cluster)
        self.assertEqual(
            int(model.cluster[planted_name].atrophy_counter.item()),
            0,
        )

    def test_grown_neuron_counts_as_used_on_its_birth_forward(self):
        model = self.growth_cluster_config(growth_threshold=1).build()
        model.pruning_threshold = 1

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_2_1_1", model.cluster)
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].atrophy_counter.item()),
            0,
        )

    def test_grow_and_prune_same_forward(self):
        model = self.pruning_cluster_model(
            pruning_threshold=2,
            growth_threshold=50,
        )
        planted_name = self.plant_isolated_neuron(model, 5)
        model.cluster[planted_name].atrophy_counter.fill_(10)
        model.cluster["neuron_3_1_1"].batch_counter.fill_(100)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertIn("neuron_2_1_1", model.cluster)
        self.assertNotIn(planted_name, model.cluster)

    def test_pruned_coordinate_becomes_growth_candidate_again(self):
        model = self.pruning_cluster_model(
            pruning_threshold=1,
            growth_threshold=50,
        )
        planted_name = self.plant_isolated_neuron(model, 5)
        input_batch = torch.randn(self.batch_size, self.input_dim)

        model(input_batch)
        self.assertNotIn(planted_name, model.cluster)

        bridge_name = self.plant_isolated_neuron(model, 4)
        model.cluster[bridge_name].batch_counter.fill_(1000)
        model(input_batch)

        self.assertIn(planted_name, model.cluster)

    def test_load_state_dict_reconciles_pruned_neurons(self):
        source_model = self.pruning_cluster_model(pruning_threshold=None)
        target_model = self.pruning_cluster_model(pruning_threshold=None)
        self.plant_isolated_neuron(target_model, 5)

        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            set(target_model.cluster.keys()),
            set(source_model.cluster.keys()),
        )

    def test_load_legacy_checkpoint_without_atrophy_counter(self):
        source_model = self.pruning_cluster_model(pruning_threshold=None)
        legacy_state_dict = {
            key: value
            for key, value in source_model.state_dict().items()
            if not key.endswith(".atrophy_counter")
        }
        target_model = self.pruning_cluster_model(pruning_threshold=None)
        target_model.cluster["neuron_3_1_1"].atrophy_counter.fill_(7)

        target_model.load_state_dict(legacy_state_dict, strict=True)

        self.assertEqual(
            int(target_model.cluster["neuron_3_1_1"].atrophy_counter.item()),
            0,
        )

    def test_state_dict_roundtrip_preserves_atrophy_counter(self):
        source_model = self.pruning_cluster_model(pruning_threshold=None)
        source_model.cluster["neuron_3_1_1"].atrophy_counter.fill_(5)
        target_model = self.pruning_cluster_model(pruning_threshold=None)

        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            int(target_model.cluster["neuron_3_1_1"].atrophy_counter.item()),
            5,
        )

    def test_grown_neuron_inherits_cluster_dtype_and_device(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 0.0, 0.0, 0.0],
                )
            }
        )
        model = model.to(dtype=torch.float64)
        initial_keys = set(model.cluster.keys())

        model(torch.randn(self.batch_size, self.input_dim, dtype=torch.float64))

        added_keys = set(model.cluster.keys()) - initial_keys
        self.assertEqual(len(added_keys), 1)
        new_neuron = model.cluster[added_keys.pop()]
        expected_device = next(model.parameters()).device
        self.assertEqual(
            {parameter.device for parameter in new_neuron.parameters()},
            {expected_device},
        )
        self.assertEqual(
            {
                parameter.dtype
                for parameter in new_neuron.parameters()
                if parameter.is_floating_point()
            },
            {torch.float64},
        )

    def test_entry_fanout_processes_selected_neurons_and_weighted_sum_continues(
        self,
    ):
        model = self.scripted_cluster(
            max_steps=1,
            x_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
        )
        model.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        output, _ = model(torch.zeros(1, 1))

        # Entry mixes both branches (0.25 * 1 + 0.75 * 10 = 7.75); the route
        # then follows the argmax neuron, which adds its delta of 10.
        torch.testing.assert_close(output, torch.tensor([[17.75]]))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 2)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            0,
        )
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].route_call_counter.item()),
            1,
        )

    def test_entry_weighted_candidate_updates_cluster_halting(self):
        halting_model = RecordingHaltingModel(halt_after_updates=1)
        model = self.scripted_cluster(
            max_steps=5,
            halting_model=halting_model,
            x_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
        )
        model.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=[0.25, 0.75],
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        output, _ = model(torch.zeros(1, 1))

        expected_entry_candidate = torch.tensor([[7.75]])
        torch.testing.assert_close(halting_model.inputs[0], expected_entry_candidate)
        torch.testing.assert_close(output, expected_entry_candidate)
        self.assertEqual(halting_model.update_count, 1)
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 1)

    def test_saturated_neuron_keeps_counter(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=2,
            z_axis_total_neurons=2,
            max_steps=1,
            growth_threshold=1,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertTrue(int(model.cluster["neuron_1_1_1"].batch_counter.item()) >= 1)

    def test_topk_branches_run_and_weighted_sum_continues_along_highest_route(
        self,
    ):
        model = self.scripted_cluster(max_steps=1, input_dim=2)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1], [3, 1, 1], [4, 1, 1]],
                    probabilities=[0.2, 0.7, 0.1],
                    delta=[1.0, 0.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 2.0],
                ),
                "neuron_3_1_1": ScriptedNeuron(
                    routes=[[3, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 3.0],
                ),
                "neuron_4_1_1": ScriptedNeuron(
                    routes=[[4, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 4.0],
                ),
            }
        )

        output, auxiliary_loss = model(torch.zeros(2, 2))

        # Step mixes all three branches: 0.2 * 2 + 0.7 * 3 + 0.1 * 4 = 2.9.
        torch.testing.assert_close(output, torch.tensor([[1.0, 2.9], [1.0, 2.9]]))
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            1,
        )
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].route_call_counter.item()),
            0,
        )
        self.assertEqual(int(model.cluster["neuron_3_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_3_1_1"].route_call_counter.item()),
            0,
        )
        self.assertEqual(int(model.cluster["neuron_4_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_4_1_1"].route_call_counter.item()),
            0,
        )

    def test_weighted_topk_candidate_updates_halting_with_sampler_weights(self):
        halting_model = RecordingHaltingModel(ponder_loss=0.5)
        model = self.scripted_cluster(
            max_steps=1,
            halting_model=halting_model,
            input_dim=2,
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1], [99, 1, 1]],
                    probabilities=[0.25, 0.75],
                    delta=[1.0, 0.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 4.0],
                ),
            }
        )

        output, auxiliary_loss = model(torch.zeros(1, 2))

        torch.testing.assert_close(halting_model.inputs[0], torch.tensor([[1.0, 0.0]]))
        # The invalid branch keeps its sampler-assigned mass and passes the
        # unchanged source signal through: 0.25 * [1, 4] + 0.75 * [1, 0].
        expected_candidate = torch.tensor([[1.0, 1.0]])
        torch.testing.assert_close(halting_model.inputs[1], expected_candidate)
        torch.testing.assert_close(output, expected_candidate)
        torch.testing.assert_close(auxiliary_loss, torch.tensor(0.5))
        self.assertEqual(halting_model.finalize_count, 1)

    def test_self_route_stops_at_max_steps(self):
        model = self.scripted_cluster(max_steps=3)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )

        output, _ = model(torch.zeros(1, 1))

        torch.testing.assert_close(output, torch.tensor([[4.0]]))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 4)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            3,
        )

    def test_cluster_uses_split_phase_neuron_lifecycle(self):
        class ForwardCapableScriptedNeuron(ScriptedNeuron):
            def forward(self, input: Tensor):
                processed_signal = self.process_signal(input)
                probabilities, routes, auxiliary_loss = self.route_signal(
                    processed_signal
                )
                return processed_signal, probabilities, routes, auxiliary_loss

        model = self.scripted_cluster(
            max_steps=1,
            input_dim=1,
            x_axis_total_neurons=1,
        )
        neuron = ForwardCapableScriptedNeuron(
            routes=[[1, 1, 1]],
            probabilities=[1.0],
            delta=[1.0],
        )
        model.cluster = nn.ModuleDict({"neuron_1_1_1": neuron})
        pre_hook_inputs: list[Tensor] = []
        forward_hook_outputs: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        pre_hook_handle = neuron.register_forward_pre_hook(
            lambda _module, inputs: pre_hook_inputs.append(inputs[0])
        )
        forward_hook_handle = neuron.register_forward_hook(
            lambda _module, _inputs, output: forward_hook_outputs.append(output)
        )

        try:
            output, _ = model(torch.zeros(1, 1))

            torch.testing.assert_close(output, torch.tensor([[2.0]]))
            self.assertEqual(pre_hook_inputs, [])
            self.assertEqual(forward_hook_outputs, [])
            self.assertEqual(int(neuron.batch_counter.item()), 2)
            self.assertEqual(int(neuron.route_call_counter.item()), 1)

            direct_output = neuron(torch.zeros(1, 1))
        finally:
            forward_hook_handle.remove()
            pre_hook_handle.remove()

        self.assertEqual(len(pre_hook_inputs), 1)
        self.assertEqual(len(forward_hook_outputs), 1)
        self.assertEqual(len(direct_output), 4)
        self.assertEqual(int(neuron.batch_counter.item()), 3)
        self.assertEqual(int(neuron.route_call_counter.item()), 2)

    def test_ping_pong_route_stops_at_max_steps(self):
        model = self.scripted_cluster(max_steps=3)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        model(torch.zeros(1, 1))

        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 2)
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 2)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            2,
        )
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].route_call_counter.item()),
            1,
        )

    def test_backward_route_is_allowed_when_coordinate_exists(self):
        model = self.scripted_cluster(max_steps=2, z_axis_total_neurons=2)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 2]],
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_1_1_2": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[2.0],
                ),
            }
        )

        model(torch.zeros(1, 1))

        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 2)
        self.assertEqual(int(model.cluster["neuron_1_1_2"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            1,
        )
        self.assertEqual(
            int(model.cluster["neuron_1_1_2"].route_call_counter.item()),
            1,
        )

    def test_missing_target_marks_sample_inactive(self):
        model = self.scripted_cluster(max_steps=5)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[99, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )

        output, _ = model(torch.zeros(1, 1))

        torch.testing.assert_close(output, torch.tensor([[1.0]]))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            1,
        )

    def test_highest_probability_escape_exits_with_weighted_candidate(self):
        model = self.scripted_cluster(
            max_steps=3,
            x_axis_total_neurons=2,
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[99, 1, 1], [2, 1, 1]],
                    probabilities=[0.8, 0.2],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        output, _ = model(torch.zeros(1, 1))

        # The invalid branch passes through the source signal with its original
        # probability: 0.8 * 1 + 0.2 * (1 + 10) = 3.
        torch.testing.assert_close(output, torch.tensor([[3.0]]))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].route_call_counter.item()),
            0,
        )

    def beam_pair_cluster(
        self,
        entry_probabilities: list[float],
        first_neuron_routes: list[list[int]],
        max_steps: int = 1,
        halting_model: nn.Module | None = None,
        beam_width: int | None = 2,
    ) -> NeuronCluster:
        model = self.scripted_cluster(
            max_steps=max_steps,
            halting_model=halting_model,
            beam_width=beam_width,
            x_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
        )
        model.entry_sampler = ScriptedSampler(
            indices=[0, 1],
            probabilities=entry_probabilities,
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=first_neuron_routes,
                    probabilities=[1.0],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )
        return model

    def test_beam_width_zero_raises(self):
        with self.assertRaises(ValueError):
            self.scripted_cluster(beam_width=0)

    def test_beam_width_one_matches_single_route_weighted_continuation(self):
        model = self.beam_pair_cluster(
            entry_probabilities=[0.25, 0.75],
            first_neuron_routes=[[1, 1, 1]],
            beam_width=1,
        )

        output, _ = model(torch.zeros(1, 1))

        # Same value as the default-path entry fanout test: entry mixes both
        # branches to 7.75, then the argmax neuron adds its delta of 10.
        torch.testing.assert_close(output, torch.tensor([[17.75]]))

    def test_beam_search_continues_multiple_routes_and_merges_by_probability(self):
        model = self.beam_pair_cluster(
            entry_probabilities=[0.25, 0.75],
            first_neuron_routes=[[1, 1, 1]],
        )

        output, _ = model(torch.zeros(1, 1))

        # The 0.75-beam walks neuron_2 twice (0 + 10 + 10 = 20), the
        # 0.25-beam walks neuron_1 twice (0 + 1 + 1 = 2); the sampler
        # probabilities directly weight the final values -> 15.5.
        torch.testing.assert_close(output, torch.tensor([[15.5]]))
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 2)
        self.assertEqual(int(model.cluster["neuron_2_1_1"].batch_counter.item()), 2)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            1,
        )
        self.assertEqual(
            int(model.cluster["neuron_2_1_1"].route_call_counter.item()),
            1,
        )

    def test_beam_multiplies_probabilities_across_route_steps(self):
        model = self.beam_pair_cluster(
            entry_probabilities=[0.5, 0.4],
            first_neuron_routes=[[1, 1, 1]],
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[0.5],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[0.25],
                    delta=[10.0],
                ),
            }
        )

        output, _ = model(torch.zeros(1, 1))

        # Path masses are 0.5 * 0.5 and 0.4 * 0.25. Their branch values are
        # 2 and 20, so the unnormalized merge is 0.25 * 2 + 0.1 * 20.
        torch.testing.assert_close(output, torch.tensor([[2.5]]))

    def test_beam_pruning_discards_probability_mass_without_redistribution(self):
        model = self.scripted_cluster(
            max_steps=1,
            halting_model=RecordingHaltingModel(halt_after_updates=1),
            beam_width=2,
            x_axis_total_neurons=3,
            initial_x_axis_total_neurons=3,
        )
        model.entry_sampler = ScriptedSampler(
            indices=[0, 1, 2],
            probabilities=[0.5, 0.3, 0.2],
        )
        model.cluster = nn.ModuleDict(
            {
                f"neuron_{index}_1_1": ScriptedNeuron(
                    routes=[[index, 1, 1]],
                    probabilities=[1.0],
                    delta=[delta],
                )
                for index, delta in ((1, 1.0), (2, 10.0), (3, 100.0))
            }
        )

        output, _ = model(torch.zeros(1, 1))

        # Beam width two drops the 0.2 path entirely: 0.5 * 1 + 0.3 * 10.
        torch.testing.assert_close(output, torch.tensor([[3.5]]))

    def test_all_invalid_beam_branches_keep_identity_mass(self):
        model = self.scripted_cluster(
            max_steps=1,
            beam_width=2,
            x_axis_total_neurons=1,
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[98, 1, 1], [99, 1, 1]],
                    probabilities=[0.25, 0.75],
                    delta=[1.0],
                )
            }
        )

        output, _ = model(torch.zeros(1, 1))

        # Both invalid branches pass through the same source hidden state.
        torch.testing.assert_close(output, torch.tensor([[1.0]]))

    def test_beam_cycles_stop_at_required_max_steps_without_adaptive_halting(
        self,
    ) -> None:
        model = self.beam_pair_cluster(
            entry_probabilities=[0.5, 0.5],
            first_neuron_routes=[[1, 1, 1]],
            max_steps=3,
        )

        output, _ = model(torch.zeros(1, 1))

        torch.testing.assert_close(output, torch.tensor([[22.0]]))
        for neuron_name in ("neuron_1_1_1", "neuron_2_1_1"):
            with self.subTest(neuron_name=neuron_name):
                neuron = model.cluster[neuron_name]
                self.assertEqual(int(neuron.batch_counter.item()), 4)
                self.assertEqual(int(neuron.route_call_counter.item()), 3)

    def test_beam_escaped_route_competes_in_final_merge(self):
        model = self.beam_pair_cluster(
            entry_probabilities=[0.4, 0.6],
            first_neuron_routes=[[99, 1, 1]],
        )

        output, _ = model(torch.zeros(1, 1))

        # The 0.6-beam expands to 20; the 0.4-beam's only branch is invalid,
        # so it finishes at 1 and keeps its mass: 0.6 * 20 + 0.4 * 1 = 12.4.
        torch.testing.assert_close(output, torch.tensor([[12.4]]))

    def test_beam_escaped_route_is_final_and_inactive(self) -> None:
        model = self.beam_pair_cluster(
            entry_probabilities=[0.4, 0.6],
            first_neuron_routes=[[99, 1, 1]],
        ).eval()
        input_tensor = torch.zeros(1, 1)

        entry_state = model._NeuronClusterBeamRoutesMixin__run_entry_routes_with_beams(
            input_tensor
        )
        route_state = model._NeuronClusterBeamRoutesMixin__run_beam_route_step(
            entry_state,
            model._current_route_mask(entry_state),
        )

        torch.testing.assert_close(
            route_state.hidden,
            torch.tensor([[20.0], [1.0]]),
        )
        torch.testing.assert_close(
            route_state.positions,
            torch.tensor([[2, 1, 1], [99, 1, 1]]),
        )
        torch.testing.assert_close(
            route_state.active_mask,
            torch.tensor([True, False]),
        )
        torch.testing.assert_close(
            route_state.escaped_mask,
            torch.tensor([False, True]),
        )
        torch.testing.assert_close(
            route_state.final_mask,
            torch.tensor([False, True]),
        )
        torch.testing.assert_close(
            route_state.beam_path_probabilities,
            torch.tensor([0.6, 0.4]),
        )
        self.assertFalse(
            bool((route_state.active_mask & route_state.final_mask).any().item())
        )

    def test_beam_halting_updates_per_beam_and_merges_finalized_hidden(self):
        halting_model = RecordingHaltingModel(halt_after_updates=1)
        model = self.beam_pair_cluster(
            entry_probabilities=[0.25, 0.75],
            first_neuron_routes=[[1, 1, 1]],
            max_steps=3,
            halting_model=halting_model,
        )

        output, _ = model(torch.zeros(1, 1))

        # The entry update halts every beam, so the route loop never runs;
        # the finalized per-beam hidden states merge by entry probabilities.
        self.assertEqual(halting_model.update_count, 1)
        torch.testing.assert_close(
            halting_model.inputs[0],
            torch.tensor([[10.0], [1.0]]),
        )
        torch.testing.assert_close(output, torch.tensor([[7.75]]))

    def test_beam_merge_preserves_unnormalized_sampler_mass(self):
        model = self.beam_pair_cluster(
            entry_probabilities=[0.2, 0.3],
            first_neuron_routes=[[1, 1, 1]],
            max_steps=3,
            halting_model=RecordingHaltingModel(halt_after_updates=1),
        )

        output, _ = model(torch.zeros(1, 1))

        torch.testing.assert_close(output, torch.tensor([[3.2]]))

    def test_beam_halting_matches_single_route_owner_lifecycle(self):
        results: dict[int, tuple[Tensor, Tensor]] = {}

        for beam_width in (1, 2):
            with self.subTest(beam_width=beam_width):
                halting_model = RecordingHaltingModel(ponder_loss=0.5)
                model = self.scripted_cluster(
                    max_steps=1,
                    halting_model=halting_model,
                    input_dim=1,
                    x_axis_total_neurons=1,
                    beam_width=beam_width,
                ).eval()
                neuron = ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
                model.cluster = nn.ModuleDict({"neuron_1_1_1": neuron})

                results[beam_width] = model(torch.zeros(1, 1))

                self.assertEqual(int(neuron.batch_counter), 2)
                self.assertEqual(int(neuron.route_call_counter), 1)

        torch.testing.assert_close(results[2][0], results[1][0])
        torch.testing.assert_close(results[2][1], results[1][1])

    def test_return_trace_with_beam_width_raises(self):
        model = self.beam_pair_cluster(
            entry_probabilities=[0.25, 0.75],
            first_neuron_routes=[[1, 1, 1]],
        )

        with self.assertRaises(NotImplementedError):
            model(torch.zeros(1, 1), return_trace=True)

    def test_growth_warmup_steps_without_growth_threshold_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=2,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                growth_warmup_steps=5,
                neuron_config=self.neuron_config(),
            ).build()

    def test_grown_neuron_starts_warmup_countdown(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            growth_warmup_steps=5,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[0.0, 0.0, 0.0, 0.0],
                )
            }
        )

        model(torch.randn(self.batch_size, self.input_dim))

        grown_neuron = model.cluster["neuron_2_1_1"]
        self.assertEqual(int(grown_neuron.warmup_remaining_steps.item()), 5)

        model(torch.randn(self.batch_size, self.input_dim))

        self.assertEqual(int(grown_neuron.warmup_remaining_steps.item()), 4)

    def test_warmup_blends_grown_neuron_output_toward_input(self):
        model = self.scripted_cluster(max_steps=1)
        model.growth_warmup_steps = 4
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[99, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )
        model.cluster["neuron_1_1_1"].register_buffer(
            "warmup_remaining_steps",
            torch.tensor(4, dtype=torch.int64),
        )

        first_output, _ = model(torch.zeros(1, 1))
        second_output, _ = model(torch.zeros(1, 1))
        model.eval()
        eval_output, _ = model(torch.zeros(1, 1))

        # Fade-in weight ramps 1/4, 2/4, ... once per training forward, so
        # the delta of 1 surfaces as 0.25 then 0.5; eval applies the current
        # weight (3/4) without advancing the countdown.
        torch.testing.assert_close(first_output, torch.tensor([[0.25]]))
        torch.testing.assert_close(second_output, torch.tensor([[0.5]]))
        torch.testing.assert_close(eval_output, torch.tensor([[0.75]]))
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].warmup_remaining_steps.item()),
            2,
        )

    def test_state_dict_roundtrip_preserves_warmup_countdown(self):
        source_model = self.scripted_cluster(max_steps=1)
        source_model.cluster["neuron_1_1_1"].register_buffer(
            "warmup_remaining_steps",
            torch.tensor(2, dtype=torch.int64),
        )
        target_model = self.scripted_cluster(max_steps=1)

        target_model.load_state_dict(source_model.state_dict(), strict=True)

        self.assertEqual(
            int(target_model.cluster["neuron_1_1_1"].warmup_remaining_steps.item()),
            2,
        )

    def test_legacy_state_dict_zero_fills_warmup_countdown(self):
        legacy_state_dict = self.scripted_cluster(max_steps=1).state_dict()
        target_model = self.scripted_cluster(max_steps=1)
        target_model.cluster["neuron_1_1_1"].register_buffer(
            "warmup_remaining_steps",
            torch.tensor(3, dtype=torch.int64),
        )

        target_model.load_state_dict(legacy_state_dict, strict=True)

        self.assertEqual(
            int(target_model.cluster["neuron_1_1_1"].warmup_remaining_steps.item()),
            0,
        )

    def test_explicit_box_shape_matches_default_connections(self):
        default_terminal = self.terminal_config().build()
        box_terminal = self.terminal_config(
            connection_shape=TerminalConnectionShapeOptions.BOX,
        ).build()

        self.assertEqual(box_terminal.total_neuron_connections, 18)
        torch.testing.assert_close(
            box_terminal.neuron_connections,
            default_terminal.neuron_connections,
        )

    def test_cross_shape_keeps_axis_lines_only(self):
        terminal = self.shaped_terminal(
            TerminalConnectionShapeOptions.CROSS,
            num_experts=6,
        )

        self.assertEqual(terminal.total_neuron_connections, 6)
        self.assertEqual(
            self.terminal_connection_set(terminal),
            {(0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 2)},
        )

    def test_sphere_shape_keeps_ellipsoid_offsets(self):
        terminal = self.shaped_terminal(
            TerminalConnectionShapeOptions.SPHERE,
            num_experts=15,
            xy_axis_range=TerminalRangeOptions.TWO,
            z_axis_range=TerminalRangeOptions.TWO,
        )

        connection_set = self.terminal_connection_set(terminal)
        self.assertEqual(terminal.total_neuron_connections, 15)
        # Window poles survive only on the axis; the mid-z plane holds the
        # full disc; box corners fall outside the ellipsoid.
        self.assertIn((1, 1, 1), connection_set)
        self.assertIn((1, 1, 3), connection_set)
        self.assertIn((3, 1, 2), connection_set)
        self.assertIn((2, 2, 2), connection_set)
        self.assertNotIn((3, 3, 2), connection_set)
        self.assertNotIn((2, 1, 1), connection_set)

    def test_diagonal_x_shape_keeps_xy_diagonals(self):
        terminal = self.shaped_terminal(
            TerminalConnectionShapeOptions.DIAGONAL_X,
            num_experts=9,
            xy_axis_range=TerminalRangeOptions.TWO,
        )

        self.assertEqual(terminal.total_neuron_connections, 9)
        self.assertEqual(
            self.terminal_connection_set(terminal),
            {
                (-1, -1, 1),
                (-1, 3, 1),
                (0, 0, 1),
                (0, 2, 1),
                (1, 1, 1),
                (2, 0, 1),
                (2, 2, 1),
                (3, -1, 1),
                (3, 3, 1),
            },
        )

    def test_line_front_back_shape_spans_z_window(self):
        terminal = self.shaped_terminal(
            TerminalConnectionShapeOptions.LINE_FRONT_BACK,
            num_experts=3,
            z_axis_range=TerminalRangeOptions.TWO,
            z_axis_offset=TerminalZAxisOffsetOptions.ONE,
        )

        self.assertEqual(
            self.terminal_connection_set(terminal),
            {(1, 1, 0), (1, 1, 1), (1, 1, 2)},
        )

    def test_line_left_right_shape_spans_x_axis(self):
        terminal = self.shaped_terminal(
            TerminalConnectionShapeOptions.LINE_LEFT_RIGHT,
            num_experts=5,
            xy_axis_range=TerminalRangeOptions.TWO,
        )

        self.assertEqual(
            self.terminal_connection_set(terminal),
            {(-1, 1, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)},
        )

    def test_line_up_down_shape_spans_y_axis(self):
        terminal = self.shaped_terminal(
            TerminalConnectionShapeOptions.LINE_UP_DOWN,
            num_experts=3,
        )

        self.assertEqual(
            self.terminal_connection_set(terminal),
            {(1, 0, 1), (1, 1, 1), (1, 2, 1)},
        )

    def test_connection_shape_rejects_non_enum_value(self):
        with self.assertRaises(TypeError):
            self.terminal_config(connection_shape="cross").build()

    def test_halting_stops_routing_before_max_steps(self):
        halting_model = RecordingHaltingModel(halt_after_updates=1)
        model = self.scripted_cluster(max_steps=5, halting_model=halting_model)
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[1, 1, 1]],
                    probabilities=[1.0],
                    delta=[1.0],
                )
            }
        )

        model(torch.zeros(1, 1))

        self.assertEqual(halting_model.update_count, 1)
        self.assertEqual(halting_model.finalize_count, 1)
        self.assertEqual(int(model.cluster["neuron_1_1_1"].batch_counter.item()), 1)
        self.assertEqual(
            int(model.cluster["neuron_1_1_1"].route_call_counter.item()),
            0,
        )

    def test_extra_halting_methods_do_not_replace_the_supported_lifecycle(self) -> None:
        class PartialPreparedHaltingModel(RecordingHaltingModel):
            def prepare_owner_step(self, *args, **kwargs):
                raise AssertionError("partial prepared lifecycle must not be used")

            def complete_owner_step(self, *args, **kwargs):
                raise AssertionError("partial prepared lifecycle must not be used")

        for beam_width in (1, 2):
            with self.subTest(beam_width=beam_width):
                halting_model = PartialPreparedHaltingModel()
                model = self.scripted_cluster(
                    max_steps=1,
                    halting_model=halting_model,
                    input_dim=1,
                    x_axis_total_neurons=1,
                    beam_width=beam_width,
                )
                model.cluster = nn.ModuleDict(
                    {
                        "neuron_1_1_1": ScriptedNeuron(
                            routes=[[1, 1, 1]],
                            probabilities=[1.0],
                            delta=[1.0],
                        )
                    }
                )

                output, auxiliary_loss = model(torch.zeros(1, 1))

                torch.testing.assert_close(
                    output,
                    torch.full_like(output, 2.0),
                )
                torch.testing.assert_close(
                    auxiliary_loss,
                    torch.zeros_like(auxiliary_loss),
                )
                self.assertEqual(halting_model.update_count, 2)
                self.assertEqual(halting_model.finalize_count, 1)

    def test_return_trace_records_detached_entry_and_route_steps(self):
        model = self.scripted_cluster(
            max_steps=1,
            x_axis_total_neurons=2,
        )
        model.cluster = nn.ModuleDict(
            {
                "neuron_1_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1], [99, 1, 1]],
                    probabilities=[0.4, 0.6],
                    delta=[1.0],
                ),
                "neuron_2_1_1": ScriptedNeuron(
                    routes=[[2, 1, 1]],
                    probabilities=[1.0],
                    delta=[10.0],
                ),
            }
        )

        output, auxiliary_loss, trace = model(
            torch.zeros(1, 1, requires_grad=True),
            return_trace=True,
        )

        self.assertEqual(output.shape, (1, 1))
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertIsInstance(trace, NeuronClusterTrace)
        self.assertEqual(trace.input_shape, (1, 1))
        self.assertEqual(len(trace.steps), 1)
        self.assertFalse(trace.entry_probabilities.requires_grad)
        self.assertFalse(trace.entry_selected_coordinates.requires_grad)
        self.assertFalse(trace.steps[0].probabilities.requires_grad)
        self.assertFalse(trace.steps[0].selected_coordinates.requires_grad)
        torch.testing.assert_close(
            trace.entry_selected_coordinates,
            torch.tensor([[[1, 1, 1]]]),
        )
        torch.testing.assert_close(
            trace.steps[0].escape_mask,
            torch.tensor([[False, True]]),
        )
        torch.testing.assert_close(
            trace.steps[0].chosen_branch_indices,
            torch.tensor([1]),
        )

    def test_gradient_flows_through_cluster_branching_and_halting(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            halting_config=self.halting_config(input_dim=self.input_dim),
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model.eval()
        input_batch = torch.randn(
            self.batch_size,
            self.input_dim,
            requires_grad=True,
        )

        output, auxiliary_loss = model(input_batch)
        (output.sum() + auxiliary_loss).backward()

        self.assertIsNotNone(input_batch.grad)
        self.assertTrue(torch.any(input_batch.grad.abs() > 0.0))
        nonzero_grads = [
            param.grad
            for param in model.parameters()
            if param.requires_grad
            and param.grad is not None
            and torch.any(param.grad.abs() > 0.0)
        ]
        self.assertTrue(len(nonzero_grads) > 0)


def _init_distributed_growth_process_group(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _assert_equal_across_ranks(world_size: int, value, label: str) -> None:
    gathered_values = [None] * world_size
    torch.distributed.all_gather_object(gathered_values, value)
    assert all(
        gathered_value == gathered_values[0] for gathered_value in gathered_values
    ), f"{label} diverged across ranks: {gathered_values}"


def _assert_tensors_bitwise_equal_across_ranks(
    world_size: int,
    tensors: dict[str, Tensor],
    label: str,
) -> None:
    gathered_tensors = [None] * world_size
    torch.distributed.all_gather_object(gathered_tensors, tensors)
    reference_tensors = gathered_tensors[0]
    for rank_index, rank_tensors in enumerate(gathered_tensors[1:], start=1):
        assert rank_tensors.keys() == reference_tensors.keys(), (
            f"{label} keys diverged on rank {rank_index}"
        )
        for key in reference_tensors:
            assert torch.equal(rank_tensors[key], reference_tensors[key]), (
                f"{label} diverged across ranks for {key} on rank {rank_index}"
            )


def _distributed_growth_worker_assert_identical_growth(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
    expected_grown_neuron_name: str,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        model = config.build()
        torch.manual_seed(100 + rank)

        model(torch.randn(3, input_dim))

        cluster_keys = sorted(model.cluster.keys())
        _assert_equal_across_ranks(world_size, cluster_keys, "cluster keys")
        assert expected_grown_neuron_name in model.cluster, (
            f"rank {rank} expected {expected_grown_neuron_name} to be grown, "
            f"found {cluster_keys}"
        )
        grown_neuron_state = {
            key: value.detach().cpu().clone()
            for key, value in model.cluster[expected_grown_neuron_name]
            .state_dict()
            .items()
        }
        _assert_tensors_bitwise_equal_across_ranks(
            world_size,
            grown_neuron_state,
            "grown neuron state",
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_growth_worker_assert_counter_all_reduce(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        below_global_threshold_model = config.build()
        saturated_by_global_sum_model = config.build()
        torch.manual_seed(100 + rank)

        below_global_threshold_model.cluster["neuron_1_1_1"].batch_counter.fill_(10)
        below_global_threshold_model(torch.randn(3, input_dim))
        assert len(below_global_threshold_model.cluster) == 1, (
            f"rank {rank} grew a neuron although the all-reduced counter is "
            "below growth_threshold"
        )

        saturated_by_global_sum_model.cluster["neuron_1_1_1"].batch_counter.fill_(49)
        saturated_by_global_sum_model(torch.randn(3, input_dim))
        assert "neuron_2_1_1" in saturated_by_global_sum_model.cluster, (
            f"rank {rank} did not grow although the all-reduced counter "
            "crosses growth_threshold"
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_growth_worker_assert_escape_count_alignment(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        model = config.build()
        torch.manual_seed(100 + rank)
        if rank == 0:
            model.escape_counts[3, 0, 0] = 10_000

        model(torch.randn(3, input_dim))

        assert "neuron_4_1_1" in model.cluster, (
            f"rank {rank} ignored the all-reduced escape counts, "
            f"found {sorted(model.cluster.keys())}"
        )
        assert "neuron_2_1_1" not in model.cluster, (
            f"rank {rank} grew the Manhattan-closest neuron instead of the "
            "most escaped coordinate"
        )
        assert int(model.escape_counts[3, 0, 0].item()) == 0, (
            f"rank {rank} did not reset the grown coordinate's escape count"
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_growth_worker_assert_post_load_counter_deltas(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        torch.manual_seed(0)
        source = config.build()
        source.cluster["neuron_1_1_1"].batch_counter.fill_(5)
        source.escape_counts[1, 0, 0] = 7
        target = config.build()
        target.load_state_dict(source.state_dict(), strict=True)
        assert target._growth_counters_are_global

        contribution_sum = world_size * (world_size + 1) // 2
        observed_intervals = []
        for interval in (1, 2):
            baseline = target._capture_growth_counter_baseline()
            target.cluster["neuron_1_1_1"].batch_counter.add_(rank + 1)
            target.escape_counts[1, 0, 0].add_(2 * (rank + 1))

            target._check_neuron_growth(baseline)

            observed = (
                int(target.cluster["neuron_1_1_1"].batch_counter),
                int(target.escape_counts[1, 0, 0]),
            )
            expected = (
                5 + interval * contribution_sum,
                7 + 2 * interval * contribution_sum,
            )
            assert observed == expected, (
                f"rank {rank} duplicated or dropped a post-load distributed "
                f"counter delta at interval {interval}: {observed} != {expected}"
            )
            observed_intervals.append(observed)

        _assert_equal_across_ranks(
            world_size,
            observed_intervals,
            "post-load growth counter intervals",
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_pruning_worker_assert_identical_pruning(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        model = config.build()
        model.cluster["neuron_5_1_1"] = model._initialize_neuron(5, 1, 1)
        torch.manual_seed(100 + rank)

        model(torch.randn(3, input_dim))
        model(torch.randn(3, input_dim))

        cluster_keys = sorted(model.cluster.keys())
        _assert_equal_across_ranks(world_size, cluster_keys, "cluster keys")
        assert "neuron_5_1_1" not in model.cluster, (
            f"rank {rank} kept the idle neuron, found {cluster_keys}"
        )
    finally:
        torch.distributed.destroy_process_group()


def _distributed_pruning_worker_assert_min_reduce_protects_neuron(
    rank: int,
    world_size: int,
    init_file: str,
    config: NeuronClusterConfig,
) -> None:
    _init_distributed_growth_process_group(rank, world_size, init_file)
    try:
        input_dim = config.neuron_config.terminal_config.input_dim
        torch.manual_seed(0)
        model = config.build()
        model.cluster["neuron_5_1_1"] = model._initialize_neuron(5, 1, 1)
        torch.manual_seed(100 + rank)
        if rank == 0:
            model.cluster["neuron_5_1_1"].atrophy_counter.fill_(1000)

        model(torch.randn(3, input_dim))

        assert "neuron_5_1_1" in model.cluster, (
            f"rank {rank} pruned a neuron whose cross-rank minimum atrophy "
            "count is below pruning_threshold"
        )
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(
    torch.distributed.is_available() and torch.distributed.is_gloo_available(),
    "gloo process group support is required",
)
class TestNeuronClusterDistributedGrowth(NeuronTestCase):
    world_size = 2

    def distributed_growth_cluster_config(
        self,
        growth_threshold: int,
        mitosis_initialization_flag: bool = False,
    ) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=growth_threshold,
            mitosis_initialization_flag=mitosis_initialization_flag,
            neuron_config=self.full_sampler_neuron_config(),
        )

    def distributed_escape_growth_cluster_config(self) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=1,
            escape_driven_growth_flag=True,
            neuron_config=self.full_sampler_neuron_config(),
        )

    def spawn_distributed_growth_workers(self, worker, *worker_args) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            init_file = os.path.join(temp_dir, "process_group_init")
            torch.multiprocessing.spawn(
                worker,
                args=(self.world_size, init_file, *worker_args),
                nprocs=self.world_size,
                join=True,
            )

    def test_growth_is_identical_across_ranks(self):
        self.spawn_distributed_growth_workers(
            _distributed_growth_worker_assert_identical_growth,
            self.distributed_growth_cluster_config(growth_threshold=1),
            "neuron_2_1_1",
        )

    def test_mitosis_growth_is_bitwise_identical_across_ranks(self):
        self.spawn_distributed_growth_workers(
            _distributed_growth_worker_assert_identical_growth,
            self.distributed_growth_cluster_config(
                growth_threshold=1,
                mitosis_initialization_flag=True,
            ),
            "neuron_2_1_1",
        )

    def test_growth_triggers_on_all_reduced_batch_counters(self):
        self.spawn_distributed_growth_workers(
            _distributed_growth_worker_assert_counter_all_reduce,
            self.distributed_growth_cluster_config(growth_threshold=100),
        )

    def test_escape_driven_placement_uses_all_reduced_escape_counts(self):
        self.spawn_distributed_growth_workers(
            _distributed_growth_worker_assert_escape_count_alignment,
            self.distributed_escape_growth_cluster_config(),
        )

    def test_post_load_global_counters_add_each_rank_delta_once_per_interval(
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
            growth_threshold=10_000,
            escape_driven_growth_flag=True,
            neuron_config=self.full_sampler_neuron_config(),
        )
        for world_size in (1, 3):
            with self.subTest(world_size=world_size):
                with tempfile.TemporaryDirectory() as temp_dir:
                    init_file = os.path.join(temp_dir, "process_group_init")
                    torch.multiprocessing.spawn(
                        _distributed_growth_worker_assert_post_load_counter_deltas,
                        args=(world_size, init_file, config),
                        nprocs=world_size,
                        join=True,
                    )

    def distributed_pruning_cluster_config(
        self,
        pruning_threshold: int,
    ) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=5,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            initial_x_axis_total_neurons=1,
            initial_y_axis_total_neurons=1,
            initial_z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            pruning_threshold=pruning_threshold,
            neuron_config=self.full_sampler_neuron_config(),
        )

    def test_pruning_is_identical_across_ranks(self):
        self.spawn_distributed_growth_workers(
            _distributed_pruning_worker_assert_identical_pruning,
            self.distributed_pruning_cluster_config(pruning_threshold=2),
        )

    def test_pruning_uses_cross_rank_minimum_atrophy(self):
        self.spawn_distributed_growth_workers(
            _distributed_pruning_worker_assert_min_reduce_protects_neuron,
            self.distributed_pruning_cluster_config(pruning_threshold=50),
        )


class TestNeuronValidation(NeuronTestCase):
    def test_missing_child_config_raises(self):
        with self.assertRaises(ValueError):
            NeuronConfig(
                nucleus_config=None,
                axons_config=AxonsConfig(memory_config=None),
                terminal_config=self.terminal_config(),
            ).build()

    def test_invalid_axis_offset_raises(self):
        with self.assertRaises(ValueError):
            self.terminal_config(
                z_axis_range=TerminalRangeOptions.ONE,
                z_axis_offset=TerminalZAxisOffsetOptions.ONE,
            ).build()

    def test_non_positive_cluster_dimensions_raise(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=0,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                neuron_config=self.neuron_config(),
            ).build()

    def test_non_positive_max_steps_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=0,
                growth_threshold=None,
                neuron_config=self.neuron_config(),
            ).build()

    def test_initial_grid_larger_than_max_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                initial_x_axis_total_neurons=2,
                max_steps=1,
                growth_threshold=None,
                neuron_config=self.neuron_config(),
            ).build()

    def test_entry_sampler_num_experts_mismatch_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                entry_sampler_config=self.sampler_config(
                    input_dim=self.input_dim,
                    num_experts=2,
                    top_k=1,
                ),
                neuron_config=self.neuron_config(),
            ).build()

    def test_wrong_cluster_halting_config_type_raises(self):
        with self.assertRaises(TypeError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                halting_config=self.projection_config(),
                neuron_config=self.neuron_config(),
            ).build()

    def test_cluster_halting_input_dim_mismatch_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                halting_config=self.halting_config(input_dim=self.input_dim + 1),
                neuron_config=self.neuron_config(),
            ).build()

    def test_wrong_forward_tensor_rank_raises(self):
        model = self.neuron_config().build()

        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, self.input_dim))

    def test_cluster_feature_dim_mismatch_raises(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        with self.assertRaises(ValueError):
            model(torch.randn(2, self.input_dim + 1))

    def test_cluster_rank_one_input_raises(self):
        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.neuron_config(),
        ).build()

        with self.assertRaises(ValueError):
            model(torch.randn(self.input_dim))

    def test_missing_router_config_raises_when_terminal_requires_routing(self):
        total_connections = self.terminal_total_connections()
        sampler_config = self.sampler_config(
            input_dim=self.input_dim,
            num_experts=total_connections,
            router_config=None,
        )

        with self.assertRaises(ValueError):
            self.terminal_config(
                input_dim=self.input_dim,
                sampler_config=sampler_config,
            ).build()

    def logits_only_neuron_config(self) -> NeuronConfig:
        total_connections = self.terminal_total_connections()
        return NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=total_connections,
                    output_dim=total_connections,
                    scale=0.25,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(
                input_dim=total_connections,
                sampler_config=self.sampler_config(
                    input_dim=total_connections,
                    num_experts=total_connections,
                    router_config=None,
                ),
            ),
        )

    def test_growth_placement_flags_require_growth_threshold(self):
        for flag_name in (
            "escape_driven_growth_flag",
            "mitosis_initialization_flag",
        ):
            with self.subTest(flag=flag_name):
                with self.assertRaises(ValueError):
                    NeuronClusterConfig(
                        x_axis_total_neurons=1,
                        y_axis_total_neurons=1,
                        z_axis_total_neurons=1,
                        max_steps=1,
                        growth_threshold=None,
                        neuron_config=self.neuron_config(),
                        **{flag_name: True},
                    ).build()

    def test_growth_budget_options_require_growth_threshold(self):
        for option_name in (
            "growth_cooldown_steps",
            "max_total_growths",
        ):
            with self.subTest(option=option_name):
                with self.assertRaises(ValueError):
                    NeuronClusterConfig(
                        x_axis_total_neurons=1,
                        y_axis_total_neurons=1,
                        z_axis_total_neurons=1,
                        max_steps=1,
                        growth_threshold=None,
                        neuron_config=self.neuron_config(),
                        **{option_name: 1},
                    ).build()

    def test_non_positive_growth_budget_options_raise(self):
        for option_name in (
            "growth_cooldown_steps",
            "max_total_growths",
        ):
            with self.subTest(option=option_name):
                with self.assertRaises(ValueError):
                    NeuronClusterConfig(
                        x_axis_total_neurons=1,
                        y_axis_total_neurons=1,
                        z_axis_total_neurons=1,
                        max_steps=1,
                        growth_threshold=1,
                        neuron_config=self.neuron_config(),
                        **{option_name: 0},
                    ).build()

    def test_nucleus_model_dimension_mismatch_raises(self):
        neuron_config = NeuronConfig(
            nucleus_config=NucleusConfig(
                model_config=self.projection_config(
                    input_dim=self.input_dim,
                    output_dim=self.input_dim + 1,
                    scale=0.25,
                )
            ),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=self.terminal_config(input_dim=self.input_dim),
        )

        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=1,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                neuron_config=neuron_config,
            ).build()

    def test_derived_logits_only_entry_sampler_dim_mismatch_raises(self):
        with self.assertRaises(ValueError):
            NeuronClusterConfig(
                x_axis_total_neurons=2,
                y_axis_total_neurons=1,
                z_axis_total_neurons=1,
                max_steps=1,
                growth_threshold=None,
                neuron_config=self.logits_only_neuron_config(),
            ).build()

    def test_derived_logits_only_entry_sampler_matching_dim_builds(self):
        total_connections = self.terminal_total_connections()
        model = NeuronClusterConfig(
            x_axis_total_neurons=total_connections,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.logits_only_neuron_config(),
        ).build()

        self.assertEqual(len(model.cluster), total_connections)


if __name__ == "__main__":
    unittest.main()
