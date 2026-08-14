import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from emperor.attention import AttentionLayerState
from emperor.config import ConfigBase, optional_field
from emperor.halting import HaltingConfig, HaltingUsageTrackerManager
from emperor.layers import (
    AdditiveResidualConfig,
    AttentionResidualConfig,
    HierarchicalReasoningModelRecurrentConfig,
    LayerNormPositionOptions,
    LayerState,
    RowLayout,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model import (
    HierarchicalReasoningModelRecurrent,
    _HierarchicalReasoningModelState,
)
from emperor.memory import MemoryPositionOptions
from emperor.nn import Module
from support.layers import weighted_memory_config


@dataclass
class _IncrementBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    increment: float | None = optional_field("Value added by the transition.")
    auxiliary_loss: float | None = optional_field(
        "Optional loss emitted by the transition."
    )

    def _registry_owner(self) -> type:
        return _IncrementBlock


class _IncrementBlock(Module):
    def __init__(
        self,
        cfg: _IncrementBlockConfig,
        overrides: _IncrementBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.inputs: list[torch.Tensor] = []
        self.grad_modes: list[bool] = []
        self.row_layouts: list[RowLayout | None] = []

    def forward(self, state: LayerState) -> LayerState:
        self.inputs.append(state.hidden.detach().clone())
        self.grad_modes.append(torch.is_grad_enabled())
        self.row_layouts.append(state.row_layout)
        state.hidden = state.hidden * self.scale + self.cfg.increment
        if self.cfg.auxiliary_loss is not None:
            state.loss = state.hidden.new_tensor(self.cfg.auxiliary_loss)
        return state


@dataclass
class _StatefulDynamicBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return _StatefulDynamicBlock


class _StatefulDynamicBlock(Module):
    def __init__(
        self,
        cfg: _StatefulDynamicBlockConfig,
        overrides: _StatefulDynamicBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("transition_step", torch.zeros(()))
        self.inputs: list[torch.Tensor] = []

    def forward(self, state: LayerState) -> LayerState:
        self.inputs.append(state.hidden.detach().clone())
        self.transition_step.add_(1.0)
        state.hidden = state.hidden * self.scale + self.transition_step
        return state


@dataclass
class _RecordingHaltingState:
    update_count: int
    halt_mask: torch.Tensor


@dataclass
class _RecordingHaltingConfig(HaltingConfig):
    halt_after_updates: int | None = optional_field(
        "Number of eligible recurrent outputs observed before halting."
    )

    def _registry_owner(self) -> type:
        return _RecordingHalting


class _RecordingHalting(Module):
    def __init__(
        self,
        cfg: _RecordingHaltingConfig,
        overrides: _RecordingHaltingConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.update_inputs: list[torch.Tensor] = []
        self.finalize_calls = 0

    @classmethod
    def implements_halting_interface(cls) -> bool:
        return True

    def update_halting_state(
        self,
        previous_state: _RecordingHaltingState | None,
        model_hidden_state: torch.Tensor,
    ) -> tuple[_RecordingHaltingState, torch.Tensor]:
        self.update_inputs.append(model_hidden_state.detach().clone())
        update_count = 1 if previous_state is None else previous_state.update_count + 1
        halted = update_count >= self.cfg.halt_after_updates
        halt_mask = torch.full(
            model_hidden_state.shape[:-1],
            halted,
            dtype=torch.bool,
            device=model_hidden_state.device,
        )
        return _RecordingHaltingState(update_count, halt_mask), model_hidden_state

    def finalize_weighted_accumulation(
        self,
        state: _RecordingHaltingState,
        current_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.finalize_calls += 1
        return current_hidden, current_hidden.new_zeros(())


@dataclass
class _NoDimensionBlockConfig(ConfigBase):
    value: int | None = optional_field("A field unrelated to dimensions.")

    def _registry_owner(self) -> type:
        return _IncrementBlock


@dataclass
class _TensorReturningBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return _TensorReturningBlock


class _TensorReturningBlock(Module):
    def __init__(
        self,
        cfg: _TensorReturningBlockConfig,
        overrides: _TensorReturningBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)

    def forward(self, state: LayerState) -> torch.Tensor:
        return state.hidden


class TestHierarchicalReasoningModelRecurrentConfig(unittest.TestCase):
    def test_gradient_suffix_requires_enough_complete_halting_updates(self) -> None:
        config = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=2,
            low_cycles=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
            gradient_transition_count=4,
            halting_config=_RecordingHaltingConfig(
                halt_after_updates=10,
                min_steps=1,
            ),
        )

        with self.assertRaisesRegex(ValueError, "required_update_count=2"):
            config.build()

    def test_overrides_do_not_mutate_the_source_config(self) -> None:
        config = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=1,
            low_cycles=2,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        )

        runtime = config.build(
            overrides=HierarchicalReasoningModelRecurrentConfig(
                high_cycles=3, low_cycles=4
            )
        )

        self.assertEqual((runtime.high_cycles, runtime.low_cycles), (3, 4))
        self.assertEqual((config.high_cycles, config.low_cycles), (1, 2))

    def test_invalid_dimensions_clocks_resources_and_initialization_are_rejected(
        self,
    ) -> None:
        values = {
            "input_dim": 1,
            "output_dim": 1,
            "high_block_config": _IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
            ),
            "low_block_config": _IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            "high_cycles": 2,
            "low_cycles": 2,
            "initial_iterations": 2,
            "iteration_increment": 1,
            "forward_calls_before_iteration_increment": 1,
            "initialization_standard_deviation": 0.0,
        }
        cases = (
            ("input_dim", 2, ValueError, "input_dim and output_dim must be equal"),
            ("high_cycles", True, TypeError, "high_cycles must be int"),
            ("high_cycles", 0, ValueError, "high_cycles must be greater than 0"),
            ("low_cycles", True, TypeError, "low_cycles must be int"),
            ("low_cycles", 0, ValueError, "low_cycles must be greater than 0"),
            (
                "no_gradient_transition_count",
                True,
                TypeError,
                "no_gradient_transition_count must be int",
            ),
            (
                "no_gradient_transition_count",
                -1,
                ValueError,
                "greater than or equal to 0",
            ),
            (
                "no_gradient_transition_count",
                6,
                ValueError,
                "less than the minimum active 6 transitions",
            ),
            (
                "initialization_standard_deviation",
                True,
                ValueError,
                "finite non-negative",
            ),
            (
                "initialization_standard_deviation",
                -0.1,
                ValueError,
                "finite non-negative",
            ),
            (
                "initialization_standard_deviation",
                float("inf"),
                ValueError,
                "finite non-negative",
            ),
            (
                "high_block_config",
                object(),
                TypeError,
                "high_block_config must be an instance of ConfigBase",
            ),
            (
                "low_block_config",
                _NoDimensionBlockConfig(value=1),
                TypeError,
                "low_block_config must declare dataclass fields",
            ),
        )
        for field_name, value, error, message in cases:
            with self.subTest(field_name=field_name, value=value):
                invalid_values = dict(values)
                invalid_values[field_name] = value
                with self.assertRaisesRegex(error, message):
                    HierarchicalReasoningModelRecurrentConfig(**invalid_values).build()

    def test_config_builds_its_exact_private_runtime_owner(self) -> None:
        config = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=1,
            low_cycles=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        )

        self.assertIs(config.registry_owner(), HierarchicalReasoningModelRecurrent)
        runtime = config.build()
        self.assertIs(type(runtime), HierarchicalReasoningModelRecurrent)
        self.assertIsInstance(runtime, RecurrentCompositionAbstract)
        self.assertIs(
            runtime.recurrent_layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertIsNone(runtime.recurrent_gate)
        self.assertIsNone(runtime.residual_connection)
        self.assertIsNone(runtime.halting_model)
        self.assertIsNone(runtime.memory_model)
        self.assertIsNone(runtime.recurrent_layer_norm_module)

    def test_nested_recurrent_compositions_are_rejected(self) -> None:
        nested_recurrent = TinyRecursiveModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            latent_updates_per_answer_update=1,
            answer_update_count=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        )
        config = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=nested_recurrent,
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=1,
            low_cycles=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "high_block_config cannot contain another recurrent composition",
        ):
            config.build()

        config.high_block_config = _IncrementBlockConfig(
            input_dim=1,
            output_dim=1,
            increment=1.0,
        )
        config.low_block_config = nested_recurrent
        with self.assertRaisesRegex(
            ValueError,
            "low_block_config cannot contain another recurrent composition",
        ):
            config.build()


class TestHierarchicalReasoningModelRecurrentValidation(unittest.TestCase):
    @staticmethod
    def _config() -> HierarchicalReasoningModelRecurrentConfig:
        return HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=1,
            low_cycles=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        )

    def test_wrong_config_and_runtime_owner_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "HierarchicalReasoningModelRecurrent cfg must be a HierarchicalReasoningModelRecurrentConfig",
        ):
            HierarchicalReasoningModelRecurrent(TinyRecursiveModelRecurrentConfig())
        with self.assertRaisesRegex(
            TypeError, "builds HierarchicalReasoningModelRecurrent, not"
        ):
            HierarchicalReasoningModelRecurrentValidator.validate(
                SimpleNamespace(cfg=self._config())
            )

    def test_invalid_shared_recurrent_controller_config_is_rejected(self) -> None:
        cases = (
            (
                "recurrent_layer_norm_position",
                object(),
                TypeError,
                "recurrent_layer_norm_position must be None or a "
                "LayerNormPositionOptions value",
            ),
            ("gate_config", object(), TypeError, "instance of GateConfig"),
            ("residual_config", object(), TypeError, "instance of ResidualConfig"),
            (
                "residual_config",
                AttentionResidualConfig(),
                ValueError,
                "AttentionResidualConfig is not supported",
            ),
            ("halting_config", object(), TypeError, "instance of HaltingConfig"),
            ("memory_config", object(), TypeError, "instance of DynamicMemoryConfig"),
        )

        for field_name, value, exception, message in cases:
            config = self._config()
            setattr(config, field_name, value)
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaisesRegex(exception, message):
                    config.build()

    def test_state_validation_rejects_every_tensor_boundary_violation(self) -> None:
        cases = (
            (object(), TypeError, "instance of LayerState"),
            (
                LayerState(hidden=torch.ones(2, 1, dtype=torch.int64)),
                TypeError,
                "floating-point Tensor",
            ),
            (LayerState(hidden=torch.ones(1)), ValueError, "rank >= 2"),
            (
                LayerState(hidden=torch.empty(0, 1)),
                ValueError,
                "non-empty leading dimensions",
            ),
            (
                LayerState(hidden=torch.ones(2, 2)),
                ValueError,
                "last dimension must be 1 for HierarchicalReasoningModelRecurrent",
            ),
        )

        for state, exception, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception, message):
                    HierarchicalReasoningModelRecurrentValidator.validate_state(
                        state, 1
                    )

    def test_initial_buffer_validation_rejects_shape_dtype_and_device_drift(
        self,
    ) -> None:
        hidden = torch.ones(2, 1)
        cases = (
            (torch.zeros(2), "must have shape"),
            (torch.zeros(1, dtype=torch.float64), "dtype/device"),
            (torch.empty(1, device="meta"), "dtype/device"),
        )

        for buffer, message in cases:
            with self.subTest(buffer=buffer):
                with self.assertRaisesRegex(ValueError, message):
                    HierarchicalReasoningModelRecurrentValidator.validate_initial_buffer(
                        buffer,
                        hidden,
                        name="high_initial",
                        expected_feature_dim=1,
                    )

    def test_transition_validation_rejects_type_shape_dtype_device_and_layout(
        self,
    ) -> None:
        transition_input = torch.ones(2, 1)
        row_layout = RowLayout.rows(2, context_sharing_restricted=False)
        cases = (
            (
                object(),
                TypeError,
                "Hierarchical Reasoning Model transition block must return LayerState",
            ),
            (
                LayerState(hidden=torch.ones(1, 1), row_layout=row_layout),
                ValueError,
                "Hierarchical Reasoning Model transition block must preserve hidden shape",
            ),
            (
                LayerState(
                    hidden=torch.ones(2, 1, dtype=torch.float64),
                    row_layout=row_layout,
                ),
                ValueError,
                "Hierarchical Reasoning Model transition block must preserve hidden dtype",
            ),
            (
                LayerState(
                    hidden=torch.empty(2, 1, device="meta"),
                    row_layout=row_layout,
                ),
                ValueError,
                "Hierarchical Reasoning Model transition block must preserve hidden device",
            ),
            (
                LayerState(
                    hidden=torch.ones(2, 1),
                    row_layout=RowLayout.rows(
                        2,
                        context_sharing_restricted=False,
                    ),
                ),
                ValueError,
                "Hierarchical Reasoning Model transition block must preserve the exact row_layout",
            ),
        )

        for output_state, exception, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception, message):
                    HierarchicalReasoningModelRecurrentValidator.validate_transition_output(
                        output_state,
                        transition_input,
                        row_layout,
                        expected_feature_dim=1,
                    )


class TestHierarchicalReasoningModelRecurrentRuntime(unittest.TestCase):
    @staticmethod
    def _smooth_oracle_config(
        *,
        high_cycles: int,
        initial_iterations: int,
        smooth_iteration_growth_flag: bool | None = None,
    ) -> HierarchicalReasoningModelRecurrentConfig:
        return HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
                auxiliary_loss=2.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
                auxiliary_loss=1.0,
            ),
            high_cycles=high_cycles,
            low_cycles=2,
            initial_iterations=initial_iterations,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=smooth_iteration_growth_flag,
            initialization_standard_deviation=0.0,
        )

    def test_schedule_grows_only_by_complete_high_cycles(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=5,
            low_cycles=2,
            initialization_standard_deviation=0.0,
            initial_iterations=1,
            iteration_increment=2,
            forward_calls_before_iteration_increment=2,
        ).build()

        executed_transition_counts = []
        for _ in range(5):
            transition_count_before = len(recurrent.low_model.inputs) + len(
                recurrent.high_model.inputs
            )
            recurrent(LayerState(hidden=torch.ones(1, 1)))
            executed_transition_counts.append(
                len(recurrent.low_model.inputs)
                + len(recurrent.high_model.inputs)
                - transition_count_before
            )

        self.assertEqual(executed_transition_counts, [3, 3, 9, 9, 15])
        schedule = recurrent.recurrent_iteration_schedule
        self.assertEqual(schedule.iteration_unit, "high_cycle")
        self.assertEqual(schedule.active_iterations, 5)

    def test_fixed_gradient_suffix_tracks_scheduled_high_cycles(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=3,
            low_cycles=2,
            initialization_standard_deviation=0.0,
            gradient_transition_count=2,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
        ).build()

        for active_transition_count in (3, 6, 9):
            recurrent.low_model.grad_modes.clear()
            recurrent.high_model.grad_modes.clear()
            recurrent(LayerState(hidden=torch.ones(1, 1)))
            gradient_modes = []
            low_modes = iter(recurrent.low_model.grad_modes)
            high_modes = iter(recurrent.high_model.grad_modes)
            for transition_index in range(active_transition_count):
                if (transition_index + 1) % 3 == 0:
                    gradient_modes.append(next(high_modes))
                else:
                    gradient_modes.append(next(low_modes))
            self.assertEqual(
                gradient_modes,
                [False] * (active_transition_count - 2) + [True, True],
            )

    def test_smooth_growth_matches_adjacent_high_cycle_oracles(self) -> None:
        smooth = self._smooth_oracle_config(
            high_cycles=2,
            initial_iterations=1,
            smooth_iteration_growth_flag=True,
        ).build()
        source = self._smooth_oracle_config(
            high_cycles=1,
            initial_iterations=1,
        ).build()
        target = self._smooth_oracle_config(
            high_cycles=2,
            initial_iterations=2,
        ).build()
        for _ in range(4):
            smooth(LayerState(hidden=torch.ones(1, 1)))
        for block_model in (smooth.low_model, smooth.high_model):
            block_model.inputs.clear()
            block_model.grad_modes.clear()

        smooth_input = torch.ones(1, 1, requires_grad=True)
        source_input = torch.ones(1, 1, requires_grad=True)
        target_input = torch.ones(1, 1, requires_grad=True)
        smooth_output = smooth(LayerState(hidden=smooth_input, loss=torch.tensor(3.0)))
        source_output = source(LayerState(hidden=source_input, loss=torch.tensor(3.0)))
        target_output = target(LayerState(hidden=target_input, loss=torch.tensor(3.0)))

        smooth_output.hidden.sum().backward()
        source_output.hidden.sum().backward()
        target_output.hidden.sum().backward()

        torch.testing.assert_close(
            smooth_output.hidden,
            0.5 * source_output.hidden + 0.5 * target_output.hidden,
        )
        torch.testing.assert_close(
            smooth_output.loss,
            0.5 * source_output.loss + 0.5 * target_output.loss,
        )
        for smooth_block, source_block, target_block in (
            (smooth.low_model, source.low_model, target.low_model),
            (smooth.high_model, source.high_model, target.high_model),
        ):
            torch.testing.assert_close(
                smooth_block.scale.grad,
                0.5 * source_block.scale.grad + 0.5 * target_block.scale.grad,
            )
        self.assertEqual(len(smooth.low_model.inputs), 4)
        self.assertEqual(
            smooth.low_model.grad_modes,
            [False, True, False, True],
        )
        self.assertEqual(len(smooth.high_model.inputs), 3)
        self.assertEqual(smooth.high_model.grad_modes, [True, False, True])
        torch.testing.assert_close(
            smooth_input.grad,
            0.5 * source_input.grad + 0.5 * target_input.grad,
        )

    def test_smooth_checkpoint_restores_the_exact_next_high_cycle(self) -> None:
        config = self._smooth_oracle_config(
            high_cycles=2,
            initial_iterations=1,
            smooth_iteration_growth_flag=True,
        )
        source = config.build()
        source.recurrent_iteration_schedule.load_state_dict(
            {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
            strict=True,
        )
        restored = config.build()
        restored.load_state_dict(source.state_dict(), strict=True)
        source_input = torch.ones(1, 1, requires_grad=True)
        restored_input = torch.ones(1, 1, requires_grad=True)

        source_output = source(LayerState(hidden=source_input, loss=torch.tensor(3.0)))
        restored_output = restored(
            LayerState(hidden=restored_input, loss=torch.tensor(3.0))
        )
        source_output.hidden.sum().backward()
        restored_output.hidden.sum().backward()

        torch.testing.assert_close(restored_output.hidden, source_output.hidden)
        torch.testing.assert_close(restored_output.loss, source_output.loss)
        for source_block, restored_block in (
            (source.low_model, restored.low_model),
            (source.high_model, restored.high_model),
        ):
            torch.testing.assert_close(
                restored_block.scale.grad,
                source_block.scale.grad,
            )
        torch.testing.assert_close(restored_input.grad, source_input.grad)
        self.assertEqual(
            restored.recurrent_iteration_schedule.snapshot(),
            source.recurrent_iteration_schedule.snapshot(),
        )

    def test_smooth_growth_commits_target_mutable_block_state(self) -> None:
        def build_runtime(
            high_cycles: int,
            *,
            smooth: bool,
        ) -> HierarchicalReasoningModelRecurrent:
            block_config = _StatefulDynamicBlockConfig(
                input_dim=1,
                output_dim=1,
            )
            return HierarchicalReasoningModelRecurrentConfig(
                input_dim=1,
                output_dim=1,
                high_block_config=block_config,
                low_block_config=block_config,
                high_cycles=2 if smooth else high_cycles,
                low_cycles=2,
                initialization_standard_deviation=0.0,
                gradient_transition_count=2,
                initial_iterations=1 if smooth else high_cycles,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=smooth,
            ).build()

        smooth = build_runtime(1, smooth=True)
        for _ in range(4):
            smooth(LayerState(hidden=torch.ones(1, 1)))
        starting_low_step = smooth.low_model.transition_step.clone()
        starting_high_step = smooth.high_model.transition_step.clone()
        source = build_runtime(1, smooth=False)
        target = build_runtime(2, smooth=False)
        source.low_model.load_state_dict(smooth.low_model.state_dict())
        source.high_model.load_state_dict(smooth.high_model.state_dict())
        target.low_model.load_state_dict(smooth.low_model.state_dict())
        target.high_model.load_state_dict(smooth.high_model.state_dict())

        smooth_input = torch.ones(1, 1, requires_grad=True)
        source_input = torch.ones(1, 1, requires_grad=True)
        target_input = torch.ones(1, 1, requires_grad=True)
        low_call_count_before = len(smooth.low_model.inputs)
        high_call_count_before = len(smooth.high_model.inputs)
        smooth_output = smooth(LayerState(hidden=smooth_input))
        source_output = source(LayerState(hidden=source_input))
        target_output = target(LayerState(hidden=target_input))

        smooth_output.hidden.sum().backward()
        source_output.hidden.sum().backward()
        target_output.hidden.sum().backward()
        torch.testing.assert_close(
            smooth_output.hidden,
            0.5 * source_output.hidden + 0.5 * target_output.hidden,
        )
        for smooth_block, source_block, target_block in (
            (smooth.low_model, source.low_model, target.low_model),
            (smooth.high_model, source.high_model, target.high_model),
        ):
            torch.testing.assert_close(
                smooth_block.scale.grad,
                0.5 * source_block.scale.grad + 0.5 * target_block.scale.grad,
            )
            torch.testing.assert_close(
                smooth_block.transition_step,
                target_block.transition_step,
            )
        torch.testing.assert_close(
            smooth_input.grad,
            0.5 * source_input.grad + 0.5 * target_input.grad,
        )
        torch.testing.assert_close(
            smooth.low_model.transition_step,
            starting_low_step + 4.0,
        )
        torch.testing.assert_close(
            smooth.high_model.transition_step,
            starting_high_step + 2.0,
        )
        self.assertEqual(len(smooth.low_model.inputs) - low_call_count_before, 4)
        self.assertEqual(len(smooth.high_model.inputs) - high_call_count_before, 3)

    def test_smooth_growth_commits_only_target_branch_halting_usage(self) -> None:
        config = self._smooth_oracle_config(
            high_cycles=2,
            initial_iterations=1,
            smooth_iteration_growth_flag=True,
        )
        config.halting_config = _RecordingHaltingConfig(halt_after_updates=1)
        runtime = config.build()
        tracker_manager = HaltingUsageTrackerManager()
        tracker = tracker_manager.attach(runtime.halting_model)
        for _ in range(4):
            runtime(LayerState(hidden=torch.ones(1, 1)))
        update_count_before = len(runtime.halting_model.update_inputs)
        finalize_count_before = runtime.halting_model.finalize_calls

        runtime(LayerState(hidden=torch.ones(1, 1)))

        self.assertEqual(
            len(runtime.halting_model.update_inputs) - update_count_before,
            2,
        )
        self.assertEqual(
            runtime.halting_model.finalize_calls - finalize_count_before,
            2,
        )
        torch.testing.assert_close(tracker.last_step_count, torch.tensor(1.0))
        tracker_manager.detach(runtime.halting_model)

    def test_residual_uses_the_previous_low_and_high_target_states(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=1,
            low_cycles=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
            residual_config=AdditiveResidualConfig(),
        ).build()
        with torch.no_grad():
            recurrent.high_initial.fill_(2.0)
            recurrent.low_initial.fill_(3.0)

        result = recurrent(LayerState(hidden=torch.ones(1, 1)))

        torch.testing.assert_close(recurrent.low_model.inputs[0], torch.tensor([[6.0]]))
        torch.testing.assert_close(
            recurrent.high_model.inputs[0],
            torch.tensor([[12.0]]),
        )
        torch.testing.assert_close(result.hidden, torch.tensor([[15.0]]))

    def test_halting_observes_high_states_and_stops_after_a_complete_cycle(
        self,
    ) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=4,
            low_cycles=2,
            initial_iterations=4,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
            no_gradient_transition_count=0,
            halting_config=_RecordingHaltingConfig(halt_after_updates=2),
        ).build()
        owner_halting_state = SimpleNamespace(owner=True)
        state = LayerState(
            hidden=torch.ones(1, 1),
            halting_state=owner_halting_state,
        )

        result = recurrent(state)

        self.assertEqual(len(recurrent.low_model.inputs), 4)
        self.assertEqual(len(recurrent.high_model.inputs), 2)
        self.assertEqual(len(recurrent.halting_model.update_inputs), 2)
        self.assertEqual(recurrent.halting_model.finalize_calls, 1)
        self.assertIs(result.halting_state, owner_halting_state)

    def test_halting_restricts_one_internal_layout_for_low_and_high_transitions(
        self,
    ) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=1,
            low_cycles=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
            no_gradient_transition_count=0,
            halting_config=_RecordingHaltingConfig(halt_after_updates=2),
        ).build()
        owner_layout = RowLayout.rows(1, context_sharing_restricted=False)
        state = LayerState(hidden=torch.ones(1, 1), row_layout=owner_layout)

        result = recurrent(state)

        low_layout = recurrent.low_model.row_layouts[0]
        high_layout = recurrent.high_model.row_layouts[0]
        self.assertIsNot(low_layout, owner_layout)
        self.assertIs(low_layout, high_layout)
        self.assertTrue(low_layout.context_sharing_restricted)
        self.assertIs(result.row_layout, owner_layout)

    def test_memory_config_is_shared_by_both_clocks_and_receives_gradients(
        self,
    ) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=2,
            output_dim=2,
            high_block_config=_IncrementBlockConfig(
                input_dim=2,
                output_dim=2,
                increment=0.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=2,
                output_dim=2,
                increment=0.0,
            ),
            high_cycles=1,
            low_cycles=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
            no_gradient_transition_count=0,
            memory_config=weighted_memory_config(
                dim=2,
                position=MemoryPositionOptions.BEFORE_AFFINE,
            ),
        ).build()
        inputs = torch.ones(1, 2, requires_grad=True)

        recurrent(LayerState(hidden=inputs)).hidden.sum().backward()

        self.assertIsNotNone(recurrent.memory_model)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in recurrent.memory_model.parameters()
            )
        )
        self.assertIsNotNone(inputs.grad)

    def test_empty_leading_dimensions_are_rejected(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=0.25,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=0.5,
            ),
            high_cycles=1,
            low_cycles=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        ).build()

        with self.assertRaisesRegex(ValueError, "non-empty leading dimensions"):
            recurrent(LayerState(hidden=torch.empty(0, 1)))

    def test_batched_execution_matches_independent_items(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=0.25,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=0.5,
            ),
            high_cycles=2,
            low_cycles=3,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        ).build()
        inputs = torch.tensor([[1.0], [3.0], [5.0]])

        batched = recurrent(LayerState(hidden=inputs.clone())).hidden
        independent = torch.cat(
            [recurrent(LayerState(hidden=item.unsqueeze(0))).hidden for item in inputs],
            dim=0,
        )

        torch.testing.assert_close(batched, independent)

    def test_bfloat16_preserves_dtype_and_final_clock_gradients(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=0.25,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=0.5,
            ),
            high_cycles=2,
            low_cycles=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        ).build()
        recurrent = recurrent.to(dtype=torch.bfloat16)
        inputs = torch.ones(2, 1, dtype=torch.bfloat16, requires_grad=True)

        output = recurrent(LayerState(hidden=inputs))
        output.hidden.float().sum().backward()

        self.assertEqual(output.hidden.dtype, torch.bfloat16)
        self.assertEqual(recurrent.high_initial.dtype, torch.bfloat16)
        self.assertEqual(recurrent.low_initial.dtype, torch.bfloat16)
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_transition_blocks_must_return_layer_state(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_TensorReturningBlockConfig(
                input_dim=1,
                output_dim=1,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=1,
            low_cycles=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        ).build()

        with self.assertRaisesRegex(
            TypeError,
            "Hierarchical Reasoning Model transition block must return LayerState",
        ):
            recurrent(LayerState(hidden=torch.ones(1, 1)))

    def test_layer_state_metadata_and_auxiliary_losses_survive_all_clocks(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
                auxiliary_loss=2.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
                auxiliary_loss=3.0,
            ),
            high_cycles=2,
            low_cycles=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        ).build()
        row_layout = RowLayout.sequence(
            leading_shape=(1, 2),
            batch_axis=0,
            sequence_axis=1,
            context_sharing_restricted=False,
        )
        key_padding_mask = torch.tensor([[False, True]])
        attention_mask = torch.zeros(2, 2)
        state = AttentionLayerState(
            hidden=torch.ones(1, 2, 1),
            loss=torch.tensor(5.0),
            row_layout=row_layout,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )

        output = recurrent(state)

        self.assertIs(output, state)
        self.assertIsInstance(output, AttentionLayerState)
        self.assertIs(output.row_layout, row_layout)
        self.assertIs(output.key_padding_mask, key_padding_mask)
        self.assertIs(output.attention_mask, attention_mask)
        torch.testing.assert_close(output.loss, torch.tensor(21.0))
        self.assertTrue(
            all(layout is row_layout for layout in recurrent.high_model.row_layouts)
        )
        self.assertTrue(
            all(layout is row_layout for layout in recurrent.low_model.row_layouts)
        )

    def test_initial_states_and_transition_parameters_round_trip_directly(self) -> None:
        config = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=2,
            low_cycles=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.2,
        )
        torch.manual_seed(17)
        recurrent = config.build()
        inputs = torch.ones(2, 1)
        expected = recurrent(LayerState(hidden=inputs.clone())).hidden

        checkpoint = recurrent.state_dict()

        self.assertEqual(
            set(checkpoint),
            {
                "recurrent_iteration_schedule.forward_call_progress",
                "high_initial",
                "low_initial",
                "high_model.scale",
                "low_model.scale",
            },
        )
        self.assertIsNot(recurrent.high_model.scale, recurrent.low_model.scale)
        torch.manual_seed(99)
        restored = config.build()
        restored.load_state_dict(checkpoint, strict=True)
        actual = restored(LayerState(hidden=inputs.clone())).hidden
        torch.testing.assert_close(actual, expected)

    def test_state_validation_reports_the_hierarchical_reasoning_model_feature_contract(
        self,
    ) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=1,
            low_cycles=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        ).build()

        with self.assertRaisesRegex(
            ValueError,
            "state.hidden last dimension must be 1 for HierarchicalReasoningModelRecurrent",
        ):
            recurrent(LayerState(hidden=torch.ones(1, 2)))

    def test_nested_low_and_high_clocks_return_the_final_high_state(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=2,
            low_cycles=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        ).build()

        output = recurrent(LayerState(hidden=torch.ones(1, 1)))

        torch.testing.assert_close(output.hidden, torch.tensor([[60.0]]))
        self.assertEqual(
            [tensor.item() for tensor in recurrent.low_model.inputs],
            [1.0, 3.0, 19.0, 35.0],
        )
        self.assertEqual(
            [tensor.item() for tensor in recurrent.high_model.inputs],
            [4.0, 50.0],
        )
        self.assertEqual(
            recurrent.low_model.grad_modes,
            [False, False, False, True],
        )
        self.assertEqual(recurrent.high_model.grad_modes, [False, True])

    def test_only_the_final_low_high_pair_builds_an_autograd_graph(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=2,
            low_cycles=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
        ).build()
        inputs = torch.ones(1, 1, requires_grad=True)

        recurrent(LayerState(hidden=inputs)).hidden.sum().backward()

        gradients = {
            name: parameter.grad for name, parameter in recurrent.named_parameters()
        }
        torch.testing.assert_close(
            gradients["high_model.scale"],
            torch.tensor(50.0),
        )
        torch.testing.assert_close(
            gradients["low_model.scale"],
            torch.tensor(35.0),
        )
        torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))

    def test_no_gradient_transition_count_can_split_a_low_high_cycle(self) -> None:
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=1,
            output_dim=1,
            high_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=10.0,
            ),
            low_block_config=_IncrementBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            high_cycles=2,
            low_cycles=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
            no_gradient_transition_count=2,
        ).build()

        result = recurrent(LayerState(hidden=torch.ones(1, 1)))
        result.hidden.sum().backward()

        self.assertEqual(
            recurrent.low_model.grad_modes,
            [False, False, True, True],
        )
        self.assertEqual(recurrent.high_model.grad_modes, [True, True])
        self.assertIsNotNone(recurrent.low_model.scale.grad)
        self.assertIsNotNone(recurrent.high_model.scale.grad)

    def test_gradient_boundary_explicitly_detaches_high_and_low_states(self) -> None:
        fixed_input = torch.ones(1, 1, requires_grad=True)
        high = fixed_input * 2.0
        low = fixed_input * 3.0
        state = _HierarchicalReasoningModelState(
            fixed_input=fixed_input,
            high=high,
            low=low,
            initial_loss=None,
            auxiliary_losses=(),
            context_state=LayerState(hidden=fixed_input),
            row_layout=None,
            transition_index=2,
        )

        detached = (
            HierarchicalReasoningModelRecurrent._detach_recurrent_execution_state(state)
        )

        self.assertIs(detached.fixed_input, fixed_input)
        self.assertFalse(detached.high.requires_grad)
        self.assertFalse(detached.low.requires_grad)
        self.assertIsNone(detached.high.grad_fn)
        self.assertIsNone(detached.low.grad_fn)
        self.assertTrue(detached.fixed_input.requires_grad)

    def test_execution_state_accumulates_losses_without_mutating_prior_state(
        self,
    ) -> None:
        recurrent = self._smooth_oracle_config(
            high_cycles=1,
            initial_iterations=1,
        ).build()
        state = recurrent._initialize_recurrent_execution_state(
            LayerState(hidden=torch.ones(1, 1)),
            branch_base_loss=None,
        )
        transition_loss = torch.tensor(2.0, requires_grad=True)

        updated = recurrent._apply_recurrent_transition_result(
            state,
            SimpleNamespace(
                hidden=torch.ones(1, 1),
                loss=transition_loss,
                halting_state=None,
                all_items_halted=False,
            ),
        )

        self.assertEqual(state.auxiliary_losses, ())
        self.assertIsInstance(updated.auxiliary_losses, tuple)
        self.assertEqual(len(updated.auxiliary_losses), 1)
        loss_gradient = torch.autograd.grad(
            updated.auxiliary_losses[0],
            transition_loss,
        )[0]
        torch.testing.assert_close(loss_gradient, torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
