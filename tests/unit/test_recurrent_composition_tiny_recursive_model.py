import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from emperor.config import ConfigBase, optional_field
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    HaltingUsageTrackerManager,
    SoftHaltingConfig,
)
from emperor.layers import (
    AdditiveResidualConfig,
    AttentionResidualConfig,
    LayerNormPositionOptions,
    LayerState,
    RecurrentLayerConfig,
    RowLayout,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.validation import (
    RecurrentLayerValidator,
    TinyRecursiveModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.tiny_recursive_model import (
    TinyRecursiveModelRecurrent,
    _TinyRecursiveModelState,
)
from emperor.memory import MemoryPositionOptions
from emperor.nn import Module
from support.layers import weighted_memory_config


@dataclass
class _RecordingBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    increment: float | None = optional_field("Value added by each transition.")
    auxiliary_loss: float | None = optional_field("Loss emitted by each transition.")

    def _registry_owner(self) -> type:
        return _RecordingBlock


class _RecordingBlock(Module):
    def __init__(
        self,
        cfg: _RecordingBlockConfig,
        overrides: _RecordingBlockConfig | None = None,
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


class _RecordingTransform(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(offset))
        self.inputs: list[torch.Tensor] = []
        self.grad_modes: list[bool] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.inputs.append(hidden.detach().clone())
        self.grad_modes.append(torch.is_grad_enabled())
        return hidden + self.offset


class _RecordingMemory(_RecordingTransform):
    def __init__(self, offset: float, position: MemoryPositionOptions) -> None:
        super().__init__(offset)
        self.memory_position_option = position


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
        return _RecordingBlock


def _config(
    *,
    model_dim: int = 1,
    latent_updates_per_answer_update: int = 2,
    answer_update_count: int = 2,
    increment: float = 1.0,
    auxiliary_loss: float | None = None,
    initialization_standard_deviation: float = 0.0,
    no_gradient_transition_count: int | None = None,
    gradient_transition_count: int | None = None,
    initial_iterations: int | None = None,
    iteration_increment: int | None = None,
    forward_calls_before_iteration_increment: int | None = None,
    smooth_iteration_growth_flag: bool | None = None,
) -> TinyRecursiveModelRecurrentConfig:
    if initial_iterations is None:
        initial_iterations = answer_update_count
    if iteration_increment is None:
        iteration_increment = 1
    if forward_calls_before_iteration_increment is None:
        forward_calls_before_iteration_increment = 1
    return TinyRecursiveModelRecurrentConfig(
        input_dim=model_dim,
        output_dim=model_dim,
        block_config=_RecordingBlockConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            increment=increment,
            auxiliary_loss=auxiliary_loss,
        ),
        latent_updates_per_answer_update=latent_updates_per_answer_update,
        answer_update_count=answer_update_count,
        initialization_standard_deviation=initialization_standard_deviation,
        no_gradient_transition_count=no_gradient_transition_count,
        gradient_transition_count=gradient_transition_count,
        initial_iterations=initial_iterations,
        iteration_increment=iteration_increment,
        forward_calls_before_iteration_increment=(
            forward_calls_before_iteration_increment
        ),
        smooth_iteration_growth_flag=smooth_iteration_growth_flag,
    )


class TestTinyRecursiveModelRecurrentConfig(unittest.TestCase):
    def test_gradient_suffix_requires_enough_complete_halting_updates(self) -> None:
        config = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=2,
            gradient_transition_count=4,
        )
        config.halting_config = _RecordingHaltingConfig(
            halt_after_updates=10,
            min_steps=1,
        )

        with self.assertRaisesRegex(ValueError, "required_update_count=2"):
            config.build()

    def test_non_default_halting_minimum_is_owned_by_the_halting_strategy(self) -> None:
        config = _config()
        config.halting_config = SoftHaltingConfig(
            input_dim=1,
            threshold=0.999,
            ponder_cost_weight=1.0,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=None,
            min_steps=2,
        )

        runtime = config.build()

        self.assertEqual(runtime.halting_model.min_steps, 2)

    def test_config_builds_its_exact_private_runtime_owner(self) -> None:
        config = _config()

        self.assertIs(config.registry_owner(), TinyRecursiveModelRecurrent)
        runtime = config.build()
        self.assertIs(type(runtime), TinyRecursiveModelRecurrent)
        self.assertIsInstance(runtime, RecurrentCompositionAbstract)

    def test_omitted_recurrent_controllers_are_disabled(self) -> None:
        runtime = _config().build()

        self.assertIs(
            runtime.recurrent_layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertIsNone(runtime.recurrent_gate)
        self.assertIsNone(runtime.residual_connection)
        self.assertIsNone(runtime.halting_model)
        self.assertIsNone(runtime.memory_model)
        self.assertIsNone(runtime.recurrent_layer_norm_module)

    def test_abbreviated_schedule_field_names_are_not_supported(self) -> None:
        for retired_field_name in ("n", "T"):
            with self.subTest(retired_field_name=retired_field_name):
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
                    TinyRecursiveModelRecurrentConfig(
                        **{retired_field_name: 1},
                    )

    def test_config_rejects_invalid_schedule_and_dimensions(self) -> None:
        cases = (
            ("input_dim", 2, ValueError, "input_dim and output_dim must be equal"),
            (
                "latent_updates_per_answer_update",
                True,
                TypeError,
                "latent_updates_per_answer_update must be int",
            ),
            (
                "latent_updates_per_answer_update",
                0,
                ValueError,
                "latent_updates_per_answer_update must be greater than 0",
            ),
            (
                "answer_update_count",
                True,
                TypeError,
                "answer_update_count must be int",
            ),
            (
                "answer_update_count",
                0,
                ValueError,
                "answer_update_count must be greater than 0",
            ),
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
        )

        for field_name, value, exception, message in cases:
            config = _config()
            setattr(config, field_name, value)
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaisesRegex(exception, message):
                    config.build()

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
            config = _config()
            setattr(config, field_name, value)
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaisesRegex(exception, message):
                    config.build()

    def test_nested_recurrence_and_wrong_runtime_configs_are_rejected(self) -> None:
        nested = _config()
        nested.block_config = _config()

        with self.assertRaisesRegex(ValueError, "cannot contain another recurrent"):
            nested.build()
        with self.assertRaisesRegex(
            TypeError,
            "TinyRecursiveModelRecurrent cfg must be a TinyRecursiveModelRecurrentConfig",
        ):
            TinyRecursiveModelRecurrent(RecurrentLayerConfig())

    def test_overrides_do_not_mutate_the_source_config(self) -> None:
        config = _config(
            latent_updates_per_answer_update=1,
            answer_update_count=2,
        )

        runtime = config.build(
            overrides=TinyRecursiveModelRecurrentConfig(
                latent_updates_per_answer_update=3,
                answer_update_count=4,
                initial_iterations=4,
                iteration_increment=1,
                forward_calls_before_iteration_increment=1,
            )
        )

        self.assertEqual(
            (
                runtime.latent_updates_per_answer_update,
                runtime.answer_update_count,
            ),
            (3, 4),
        )
        self.assertEqual(
            (
                config.latent_updates_per_answer_update,
                config.answer_update_count,
            ),
            (1, 2),
        )


class TestTinyRecursiveModelRecurrentValidation(unittest.TestCase):
    def test_config_rejects_invalid_block_and_initialization_contracts(self) -> None:
        cases = (
            ("block_config", object(), TypeError, "instance of ConfigBase"),
            (
                "block_config",
                _NoDimensionBlockConfig(value=1),
                TypeError,
                "must declare dataclass fields input_dim and output_dim",
            ),
            (
                "initialization_standard_deviation",
                True,
                ValueError,
                "finite non-negative number",
            ),
            (
                "initialization_standard_deviation",
                float("inf"),
                ValueError,
                "finite non-negative number",
            ),
            (
                "initialization_standard_deviation",
                -0.1,
                ValueError,
                "finite non-negative number",
            ),
        )

        for field_name, value, exception, message in cases:
            config = _config()
            setattr(config, field_name, value)
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaisesRegex(exception, message):
                    config.build()

    def test_exact_runtime_ownership_is_validated_for_both_leaves(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "builds TinyRecursiveModelRecurrent, not"
        ):
            TinyRecursiveModelRecurrentValidator.validate(
                SimpleNamespace(cfg=_config())
            )

        standard_config = RecurrentLayerConfig(
            input_dim=1,
            output_dim=1,
            max_steps=1,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=_RecordingBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=0.0,
                auxiliary_loss=None,
            ),
            gate_config=None,
            residual_config=None,
            halting_config=None,
            memory_config=None,
        )
        with self.assertRaisesRegex(TypeError, "builds RecurrentLayer, not"):
            RecurrentLayerValidator.validate(SimpleNamespace(cfg=standard_config))
        with self.assertRaisesRegex(TypeError, "cfg must be a RecurrentLayerConfig"):
            RecurrentLayerValidator.validate(SimpleNamespace(cfg=_config()))

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
                LayerState(hidden=torch.ones(2, 2)),
                ValueError,
                "last dimension must be 1",
            ),
        )

        for state, exception, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception, message):
                    TinyRecursiveModelRecurrentValidator.validate_state(state, 1)

    def test_initial_buffer_validation_rejects_shape_dtype_and_device_drift(
        self,
    ) -> None:
        hidden = torch.ones(2, 1)
        cases = (
            (torch.zeros(2), ValueError, "must have shape"),
            (torch.zeros(1, dtype=torch.float64), ValueError, "dtype/device"),
            (torch.empty(1, device="meta"), ValueError, "dtype/device"),
        )

        for buffer, exception, message in cases:
            with self.subTest(buffer=buffer):
                with self.assertRaisesRegex(exception, message):
                    TinyRecursiveModelRecurrentValidator.validate_initial_buffer(
                        buffer,
                        hidden,
                        name="answer_initial",
                        expected_feature_dim=1,
                    )

    def test_transition_validation_rejects_type_shape_dtype_device_and_layout(
        self,
    ) -> None:
        transition_input = torch.ones(2, 1)
        row_layout = RowLayout.rows(2, context_sharing_restricted=False)
        cases = (
            (object(), TypeError, "must return LayerState"),
            (
                LayerState(hidden=torch.ones(1, 1), row_layout=row_layout),
                ValueError,
                "preserve hidden shape",
            ),
            (
                LayerState(
                    hidden=torch.ones(2, 1, dtype=torch.float64),
                    row_layout=row_layout,
                ),
                ValueError,
                "preserve hidden dtype",
            ),
            (
                LayerState(
                    hidden=torch.empty(2, 1, device="meta"),
                    row_layout=row_layout,
                ),
                ValueError,
                "preserve hidden device",
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
                "preserve the exact row_layout",
            ),
        )

        for output_state, exception, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception, message):
                    TinyRecursiveModelRecurrentValidator.validate_transition_output(
                        output_state,
                        transition_input,
                        row_layout,
                        expected_feature_dim=1,
                    )


class TestTinyRecursiveModelRecurrentRuntime(unittest.TestCase):
    def test_schedule_grows_only_by_complete_answer_cycles(self) -> None:
        recurrent = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=5,
            initial_iterations=1,
            iteration_increment=2,
            forward_calls_before_iteration_increment=2,
        ).build()

        executed_transition_counts = []
        for _ in range(5):
            transition_count_before = len(recurrent.block_model.inputs)
            recurrent(LayerState(hidden=torch.ones(1, 1)))
            executed_transition_counts.append(
                len(recurrent.block_model.inputs) - transition_count_before
            )

        self.assertEqual(executed_transition_counts, [3, 3, 9, 9, 15])
        schedule = recurrent.recurrent_iteration_schedule
        self.assertEqual(schedule.iteration_unit, "answer_cycle")
        self.assertEqual(schedule.active_iterations, 5)

    def test_fixed_gradient_suffix_tracks_scheduled_answer_cycles(self) -> None:
        recurrent = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=3,
            gradient_transition_count=2,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
        ).build()

        for active_transition_count in (3, 6, 9):
            recurrent.block_model.grad_modes.clear()
            recurrent(LayerState(hidden=torch.ones(1, 1)))
            self.assertEqual(
                recurrent.block_model.grad_modes,
                [False] * (active_transition_count - 2) + [True, True],
            )

    def test_smooth_growth_matches_adjacent_answer_cycle_oracles(self) -> None:
        smooth = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=2,
            auxiliary_loss=0.5,
            gradient_transition_count=2,
            initial_iterations=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
        ).build()
        source = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=1,
            auxiliary_loss=0.5,
            gradient_transition_count=2,
            initial_iterations=1,
            forward_calls_before_iteration_increment=4,
        ).build()
        target = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=2,
            auxiliary_loss=0.5,
            gradient_transition_count=2,
            initial_iterations=2,
            forward_calls_before_iteration_increment=4,
        ).build()
        for _ in range(4):
            smooth(LayerState(hidden=torch.ones(1, 1)))
        smooth.block_model.inputs.clear()
        smooth.block_model.grad_modes.clear()

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
        torch.testing.assert_close(
            smooth.block_model.scale.grad,
            0.5 * source.block_model.scale.grad + 0.5 * target.block_model.scale.grad,
        )
        torch.testing.assert_close(
            smooth_input.grad,
            0.5 * source_input.grad + 0.5 * target_input.grad,
        )
        self.assertEqual(len(smooth.block_model.inputs), 7)
        self.assertEqual(
            smooth.block_model.grad_modes,
            [False, True, True, False, False, True, True],
        )

    def test_full_gradient_smooth_growth_extends_one_answer_cycle_chain(self) -> None:
        smooth = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=2,
            increment=1.0,
            auxiliary_loss=0.5,
            no_gradient_transition_count=0,
            initial_iterations=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
        ).build()
        smooth.recurrent_iteration_schedule.forward_call_progress.fill_(4)
        smooth_input = torch.ones(1, 1, requires_grad=True)
        incoming_loss = torch.tensor(3.0, requires_grad=True)

        result = smooth(LayerState(hidden=smooth_input, loss=incoming_loss))
        input_gradient, scale_gradient = torch.autograd.grad(
            result.hidden.sum(),
            (smooth_input, smooth.block_model.scale),
        )

        oracle_input = torch.ones(1, 1, requires_grad=True)
        oracle_scale = smooth.block_model.scale.detach().clone().requires_grad_()
        answer = torch.zeros_like(oracle_input)
        latent = torch.zeros_like(oracle_input)
        source_answer = answer
        for transition_index in range(6):
            if transition_index % 3 == 2:
                answer = (answer + latent) * oracle_scale + 1.0
                if transition_index == 2:
                    source_answer = answer
            else:
                latent = (latent + answer + oracle_input) * oracle_scale + 1.0
        oracle_output = 0.5 * source_answer + 0.5 * answer
        oracle_input_gradient, oracle_scale_gradient = torch.autograd.grad(
            oracle_output.sum(),
            (oracle_input, oracle_scale),
        )

        torch.testing.assert_close(result.hidden, oracle_output)
        torch.testing.assert_close(result.loss, torch.tensor(5.25))
        torch.testing.assert_close(input_gradient, oracle_input_gradient)
        torch.testing.assert_close(scale_gradient, oracle_scale_gradient)
        self.assertEqual(len(smooth.block_model.inputs), 6)
        self.assertEqual(smooth.block_model.grad_modes, [True] * 6)
        recurrent_loss_gradient = torch.autograd.grad(result.loss, incoming_loss)[0]
        torch.testing.assert_close(recurrent_loss_gradient, torch.tensor(1.0))

    def test_full_gradient_smooth_growth_halts_at_the_source_answer_cycle(
        self,
    ) -> None:
        config = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=2,
            no_gradient_transition_count=0,
            initial_iterations=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
        )
        config.halting_config = _RecordingHaltingConfig(halt_after_updates=1)
        runtime = config.build()
        runtime.recurrent_iteration_schedule.forward_call_progress.fill_(4)

        runtime(LayerState(hidden=torch.ones(1, 1)))

        self.assertEqual(len(runtime.block_model.inputs), 3)
        self.assertEqual(len(runtime.halting_model.update_inputs), 1)
        self.assertEqual(runtime.halting_model.finalize_calls, 1)

    def test_smooth_checkpoint_restores_the_exact_next_answer_cycle(self) -> None:
        config = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=2,
            auxiliary_loss=0.5,
            gradient_transition_count=2,
            initial_iterations=1,
            forward_calls_before_iteration_increment=4,
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
        torch.testing.assert_close(
            restored.block_model.scale.grad,
            source.block_model.scale.grad,
        )
        torch.testing.assert_close(restored_input.grad, source_input.grad)
        self.assertEqual(
            restored.recurrent_iteration_schedule.snapshot(),
            source.recurrent_iteration_schedule.snapshot(),
        )

    def test_smooth_growth_commits_target_mutable_block_state(self) -> None:
        def build_runtime(
            answer_update_count: int,
            *,
            smooth: bool,
        ) -> TinyRecursiveModelRecurrent:
            return TinyRecursiveModelRecurrentConfig(
                input_dim=1,
                output_dim=1,
                block_config=_StatefulDynamicBlockConfig(
                    input_dim=1,
                    output_dim=1,
                ),
                latent_updates_per_answer_update=2,
                answer_update_count=2 if smooth else answer_update_count,
                initialization_standard_deviation=0.0,
                gradient_transition_count=2,
                initial_iterations=1 if smooth else answer_update_count,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=smooth,
            ).build()

        smooth = build_runtime(1, smooth=True)
        for _ in range(4):
            smooth(LayerState(hidden=torch.ones(1, 1)))
        starting_transition_step = smooth.block_model.transition_step.clone()
        source = build_runtime(1, smooth=False)
        target = build_runtime(2, smooth=False)
        source.block_model.load_state_dict(smooth.block_model.state_dict())
        target.block_model.load_state_dict(smooth.block_model.state_dict())

        smooth_input = torch.ones(1, 1, requires_grad=True)
        source_input = torch.ones(1, 1, requires_grad=True)
        target_input = torch.ones(1, 1, requires_grad=True)
        call_count_before = len(smooth.block_model.inputs)
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
        torch.testing.assert_close(
            smooth.block_model.scale.grad,
            0.5 * source.block_model.scale.grad + 0.5 * target.block_model.scale.grad,
        )
        torch.testing.assert_close(
            smooth_input.grad,
            0.5 * source_input.grad + 0.5 * target_input.grad,
        )
        torch.testing.assert_close(
            smooth.block_model.transition_step,
            starting_transition_step + 6.0,
        )
        torch.testing.assert_close(
            smooth.block_model.transition_step,
            target.block_model.transition_step,
        )
        self.assertEqual(len(smooth.block_model.inputs) - call_count_before, 7)

    def test_exact_schedule_reuses_one_block_and_returns_the_final_answer(self) -> None:
        runtime = _config().build()
        fixed_input = torch.full((2, 1), 2.0)
        row_layout = RowLayout.rows(2, context_sharing_restricted=False)
        state = LayerState(hidden=fixed_input, row_layout=row_layout)

        result = runtime(state)

        self.assertIs(result, state)
        torch.testing.assert_close(result.hidden, torch.full_like(fixed_input, 34.0))
        expected_inputs = (2.0, 5.0, 6.0, 15.0, 25.0, 33.0)
        self.assertEqual(len(runtime.block_model.inputs), len(expected_inputs))
        for actual, expected in zip(
            runtime.block_model.inputs,
            expected_inputs,
            strict=True,
        ):
            torch.testing.assert_close(actual, torch.full_like(actual, expected))
        self.assertTrue(
            all(layout is row_layout for layout in runtime.block_model.row_layouts)
        )

    def test_residual_uses_the_previous_target_state_for_each_transition(self) -> None:
        config = _config(
            latent_updates_per_answer_update=1,
            answer_update_count=1,
        )
        config.residual_config = AdditiveResidualConfig()
        runtime = config.build()
        with torch.no_grad():
            runtime.answer_initial.fill_(2.0)
            runtime.latent_initial.fill_(3.0)

        result = runtime(LayerState(hidden=torch.ones(1, 1)))

        self.assertEqual(len(runtime.block_model.inputs), 2)
        torch.testing.assert_close(
            runtime.block_model.inputs[0],
            torch.tensor([[6.0]]),
        )
        torch.testing.assert_close(
            runtime.block_model.inputs[1],
            torch.tensor([[12.0]]),
        )
        torch.testing.assert_close(result.hidden, torch.tensor([[15.0]]))

    def test_before_normalization_and_memory_wrap_each_transition_input(self) -> None:
        runtime = _config(
            latent_updates_per_answer_update=1,
            answer_update_count=1,
        ).build()
        with torch.no_grad():
            runtime.answer_initial.fill_(2.0)
            runtime.latent_initial.fill_(3.0)
        normalization = _RecordingTransform(offset=10.0)
        memory = _RecordingMemory(
            offset=20.0,
            position=MemoryPositionOptions.BEFORE_AFFINE,
        )
        runtime.recurrent_layer_norm_position = LayerNormPositionOptions.BEFORE
        runtime.recurrent_layer_norm_module = normalization
        runtime.memory_model = memory

        result = runtime(LayerState(hidden=torch.ones(1, 1)))

        self.assertEqual(len(runtime.block_model.inputs), 2)
        torch.testing.assert_close(
            runtime.block_model.inputs[0], torch.tensor([[36.0]])
        )
        torch.testing.assert_close(
            runtime.block_model.inputs[1], torch.tensor([[69.0]])
        )
        torch.testing.assert_close(normalization.inputs[0], torch.tensor([[6.0]]))
        torch.testing.assert_close(normalization.inputs[1], torch.tensor([[39.0]]))
        torch.testing.assert_close(memory.inputs[0], torch.tensor([[16.0]]))
        torch.testing.assert_close(memory.inputs[1], torch.tensor([[49.0]]))
        torch.testing.assert_close(result.hidden, torch.tensor([[70.0]]))

    def test_halting_observes_answers_and_stops_after_a_complete_cycle(self) -> None:
        config = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=4,
            no_gradient_transition_count=0,
        )
        config.halting_config = _RecordingHaltingConfig(halt_after_updates=2)
        runtime = config.build()
        owner_halting_state = SimpleNamespace(owner=True)
        state = LayerState(
            hidden=torch.ones(1, 1),
            halting_state=owner_halting_state,
        )

        result = runtime(state)

        self.assertEqual(len(runtime.block_model.inputs), 6)
        self.assertEqual(len(runtime.halting_model.update_inputs), 2)
        self.assertEqual(runtime.halting_model.finalize_calls, 1)
        self.assertIs(result.halting_state, owner_halting_state)

    def test_halting_monitor_preserves_answer_update_depth_semantics(self) -> None:
        config = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=4,
            no_gradient_transition_count=0,
        )
        config.halting_config = _RecordingHaltingConfig(halt_after_updates=2)
        runtime = config.build()
        tracker_manager = HaltingUsageTrackerManager()
        tracker = tracker_manager.attach(runtime.halting_model)

        runtime(LayerState(hidden=torch.ones(1, 1)))

        torch.testing.assert_close(tracker.last_step_count, torch.tensor(2.0))
        tracker_manager.detach(runtime.halting_model)

    def test_smooth_growth_commits_only_target_branch_halting_usage(self) -> None:
        config = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=2,
            gradient_transition_count=2,
            initial_iterations=1,
            forward_calls_before_iteration_increment=4,
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
            runtime.halting_model.finalize_calls - finalize_count_before, 2
        )
        torch.testing.assert_close(tracker.last_step_count, torch.tensor(1.0))
        tracker_manager.detach(runtime.halting_model)

    def test_halting_starts_on_the_first_eligible_trainable_answer(self) -> None:
        config = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=3,
        )
        config.halting_config = _RecordingHaltingConfig(halt_after_updates=10)
        runtime = config.build()

        runtime(LayerState(hidden=torch.ones(1, 1)))

        self.assertEqual(runtime.block_model.grad_modes, [False] * 6 + [True] * 3)
        self.assertEqual(len(runtime.halting_model.update_inputs), 1)
        self.assertEqual(runtime.halting_model.finalize_calls, 1)

    def test_halting_restricts_one_internal_layout_without_replacing_the_owner_layout(
        self,
    ) -> None:
        config = _config(
            latent_updates_per_answer_update=1,
            answer_update_count=1,
            no_gradient_transition_count=0,
        )
        config.halting_config = _RecordingHaltingConfig(halt_after_updates=2)
        runtime = config.build()
        owner_layout = RowLayout.rows(1, context_sharing_restricted=False)
        state = LayerState(hidden=torch.ones(1, 1), row_layout=owner_layout)

        result = runtime(state)

        internal_layouts = runtime.block_model.row_layouts
        self.assertEqual(len(internal_layouts), 2)
        self.assertIsNot(internal_layouts[0], owner_layout)
        self.assertIs(internal_layouts[0], internal_layouts[1])
        self.assertTrue(internal_layouts[0].context_sharing_restricted)
        self.assertIs(result.row_layout, owner_layout)

    def test_shared_controllers_follow_the_transition_gradient_window(self) -> None:
        runtime = _config(
            latent_updates_per_answer_update=1,
            answer_update_count=2,
            increment=0.0,
            no_gradient_transition_count=2,
        ).build()
        normalization = _RecordingTransform(offset=1.0)
        runtime.recurrent_layer_norm_position = LayerNormPositionOptions.BEFORE
        runtime.recurrent_layer_norm_module = normalization
        fixed_input = torch.ones(1, 1, requires_grad=True)

        result = runtime(LayerState(hidden=fixed_input))
        result.hidden.sum().backward()

        self.assertEqual(normalization.grad_modes, [False, False, True, True])
        self.assertIsNotNone(normalization.offset.grad)
        self.assertIsNotNone(runtime.block_model.scale.grad)
        self.assertIsNotNone(fixed_input.grad)

    def test_memory_config_builds_once_and_receives_suffix_gradients(self) -> None:
        config = _config(
            model_dim=2,
            latent_updates_per_answer_update=1,
            answer_update_count=1,
            no_gradient_transition_count=0,
        )
        config.memory_config = weighted_memory_config(
            dim=2,
            position=MemoryPositionOptions.BEFORE_AFFINE,
        )
        runtime = config.build()
        fixed_input = torch.ones(1, 2, requires_grad=True)

        result = runtime(LayerState(hidden=fixed_input))
        result.hidden.sum().backward()

        self.assertIsNotNone(runtime.memory_model)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in runtime.memory_model.parameters()
            )
        )
        self.assertIsNotNone(fixed_input.grad)

    def test_only_the_final_complete_process_builds_an_autograd_graph(self) -> None:
        runtime = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=3,
            increment=0.0,
        ).build()
        inputs = torch.ones(2, 1, requires_grad=True)

        result = runtime(LayerState(hidden=inputs))
        result.hidden.sum().backward()

        self.assertEqual(
            runtime.block_model.grad_modes,
            [False] * 6 + [True] * 3,
        )
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(runtime.block_model.scale.grad)

    def test_no_gradient_transition_count_can_split_a_complete_process(self) -> None:
        runtime = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=3,
            increment=0.0,
            no_gradient_transition_count=4,
        ).build()

        result = runtime(LayerState(hidden=torch.ones(2, 1)))
        result.hidden.sum().backward()

        self.assertEqual(runtime.block_model.grad_modes, [False] * 4 + [True] * 5)
        self.assertIsNotNone(runtime.block_model.scale.grad)

    def test_gradient_boundary_explicitly_detaches_answer_and_latent(self) -> None:
        source = torch.ones(2, 1, requires_grad=True)
        answer = source * 2.0
        latent = source * 3.0
        recurrent_state = _TinyRecursiveModelState(
            fixed_input=source,
            answer=answer,
            latent=latent,
            initial_loss=None,
            auxiliary_losses=(),
            context_state=LayerState(hidden=source),
            row_layout=None,
            transition_index=2,
        )
        detached = TinyRecursiveModelRecurrent._detach_recurrent_execution_state(
            recurrent_state
        )

        self.assertFalse(detached.answer.requires_grad)
        self.assertFalse(detached.latent.requires_grad)
        self.assertIsNone(detached.answer.grad_fn)
        self.assertIsNone(detached.latent.grad_fn)
        self.assertTrue(source.requires_grad)

    def test_execution_state_accumulates_losses_without_mutating_prior_state(
        self,
    ) -> None:
        runtime = _config(
            latent_updates_per_answer_update=1,
            answer_update_count=1,
            auxiliary_loss=0.5,
        ).build()
        state = runtime._initialize_recurrent_execution_state(
            LayerState(hidden=torch.ones(1, 1)),
            branch_base_loss=None,
        )
        transition_loss = torch.tensor(2.0, requires_grad=True)

        updated = runtime._apply_recurrent_transition_result(
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

    def test_initializers_are_persistent_buffers_and_losses_accumulate(self) -> None:
        runtime = _config(
            latent_updates_per_answer_update=1,
            answer_update_count=2,
            auxiliary_loss=0.5,
        ).build()
        initial_loss = torch.tensor(2.0, requires_grad=True)

        result = runtime(LayerState(hidden=torch.ones(2, 1), loss=initial_loss))

        self.assertEqual(
            set(runtime.state_dict()),
            {
                "recurrent_iteration_schedule.forward_call_progress",
                "answer_initial",
                "latent_initial",
                "block_model.scale",
            },
        )
        self.assertEqual(dict(runtime.named_parameters()).keys(), {"block_model.scale"})
        torch.testing.assert_close(result.loss, torch.tensor(4.0))
        result.loss.backward()
        torch.testing.assert_close(initial_loss.grad, torch.tensor(1.0))

    def test_empty_batches_are_rejected_before_transition_execution(self) -> None:
        runtime = _config().build()

        with self.assertRaisesRegex(ValueError, "non-empty leading dimensions"):
            runtime(LayerState(hidden=torch.empty(0, 1)))

    def test_seeded_initialization_and_strict_checkpoint_round_trip(self) -> None:
        torch.manual_seed(17)
        first = _config(initialization_standard_deviation=0.2).build()
        torch.manual_seed(17)
        second = _config(initialization_standard_deviation=0.2).build()

        torch.testing.assert_close(first.answer_initial, second.answer_initial)
        torch.testing.assert_close(first.latent_initial, second.latent_initial)
        second.load_state_dict(first.state_dict(), strict=True)
        inputs = torch.randn(3, 1)
        torch.testing.assert_close(
            first(LayerState(hidden=inputs.clone())).hidden,
            second(LayerState(hidden=inputs.clone())).hidden,
        )

    def test_bfloat16_preserves_state_dtype_and_gradients(self) -> None:
        runtime = (
            _config(
                latent_updates_per_answer_update=1,
                answer_update_count=2,
                increment=0.0,
            )
            .build()
            .to(torch.bfloat16)
        )
        inputs = torch.ones(2, 1, dtype=torch.bfloat16, requires_grad=True)

        result = runtime(LayerState(hidden=inputs))
        result.hidden.float().sum().backward()

        self.assertEqual(result.hidden.dtype, torch.bfloat16)
        self.assertEqual(runtime.answer_initial.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(result.hidden).all())
        self.assertIsNotNone(inputs.grad)

    def test_batched_execution_matches_independent_items(self) -> None:
        runtime = _config(
            latent_updates_per_answer_update=2,
            answer_update_count=2,
            increment=0.25,
        ).build()
        inputs = torch.tensor([[1.0], [3.0], [5.0]])

        batched = runtime(LayerState(hidden=inputs.clone())).hidden
        independent = torch.cat(
            [runtime(LayerState(hidden=item.unsqueeze(0))).hidden for item in inputs],
            dim=0,
        )

        torch.testing.assert_close(batched, independent)


if __name__ == "__main__":
    unittest.main()
