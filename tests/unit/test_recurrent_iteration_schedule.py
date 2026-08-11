import inspect
import unittest

import torch

from emperor.layers._composition.recurrent.config import (
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentIterationSchedule,
)


def _standard_schedule(**overrides: object) -> RecurrentIterationSchedule:
    values = {
        "max_steps": 7,
        "initial_iterations": 2,
        "iteration_increment": 2,
        "forward_calls_before_iteration_increment": 3,
    }
    values.update(overrides)
    return RecurrentIterationSchedule(RecurrentLayerConfig(**values))


class TestRecurrentIterationSchedule(unittest.TestCase):
    def test_constructor_accepts_only_the_recurrent_config(self) -> None:
        parameters = inspect.signature(RecurrentIterationSchedule.__init__).parameters

        self.assertEqual(tuple(parameters), ("self", "config"))

    def test_resolves_every_concrete_recurrent_config(self) -> None:
        cases = (
            (
                RecurrentLayerConfig(
                    max_steps=5,
                    initial_iterations=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=3,
                ),
                ("transition", 5, 1, 2, 0),
            ),
            (
                TinyRecursiveModelRecurrentConfig(
                    latent_updates_per_answer_update=2,
                    answer_update_count=5,
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=3,
                ),
                ("answer_cycle", 5, 3, 3, 0),
            ),
            (
                HierarchicalReasoningModelRecurrentConfig(
                    high_cycles=5,
                    low_cycles=2,
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=3,
                ),
                ("high_cycle", 5, 3, 3, 1),
            ),
        )

        for config, expected in cases:
            with self.subTest(config_type=type(config).__name__):
                schedule = RecurrentIterationSchedule(config)
                self.assertEqual(
                    (
                        schedule.iteration_unit,
                        schedule.maximum_iterations,
                        schedule.transitions_per_iteration,
                        schedule.active_transition_count,
                        schedule.no_gradient_transition_count,
                    ),
                    expected,
                )

    def test_rejects_the_abstract_recurrent_config_through_its_validator(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "config must be a concrete RecurrentCompositionConfig",
        ):
            RecurrentIterationSchedule(
                RecurrentCompositionConfig(
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
                )
            )

    def test_successful_forwards_grow_complete_iterations_and_cap_at_capacity(
        self,
    ) -> None:
        schedule = _standard_schedule()

        active_iterations = []
        for _ in range(10):
            schedule.record_successful_forward()
            active_iterations.append(schedule.active_iterations)

        self.assertEqual(active_iterations, [2, 2, 4, 4, 4, 6, 6, 6, 7, 7])
        self.assertEqual(schedule.snapshot().forward_call_progress, 9)
        self.assertTrue(schedule.complete)

    def test_complete_schedule_does_not_count_forwards(self) -> None:
        schedule = _standard_schedule(max_steps=2)

        schedule.record_successful_forward()

        self.assertEqual(schedule.snapshot().forward_call_progress, 0)
        self.assertEqual(schedule.active_iterations, 2)
        self.assertTrue(schedule.complete)

    def test_gradient_suffix_moves_as_active_depth_grows(self) -> None:
        schedule = _standard_schedule(
            max_steps=5,
            initial_iterations=3,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
        )

        observed_gradient_modes = []
        for transition_index in range(schedule.active_transition_count):
            with schedule.gradient_context(transition_index):
                observed_gradient_modes.append(torch.is_grad_enabled())

        self.assertEqual(observed_gradient_modes, [False, True, True])
        self.assertTrue(schedule.starts_gradient_suffix(1))
        self.assertFalse(schedule.tracks_gradients(0))

        schedule.record_successful_forward()

        self.assertEqual(schedule.active_iterations, 4)
        self.assertEqual(schedule.no_gradient_transition_count, 2)

    def test_explicit_no_gradient_prefix_stays_fixed_as_depth_grows(self) -> None:
        schedule = _standard_schedule(
            no_gradient_transition_count=1,
            gradient_transition_count=None,
        )

        for _ in range(6):
            schedule.record_successful_forward()

        self.assertEqual(schedule.active_iterations, 6)
        self.assertEqual(schedule.no_gradient_transition_count, 1)

    def test_progress_buffer_moves_with_the_schedule_module(self) -> None:
        schedule = _standard_schedule()
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        schedule.to(target_device)

        self.assertEqual(schedule.forward_call_progress.device, target_device)
        self.assertIs(
            dict(schedule.named_buffers())["forward_call_progress"],
            schedule.forward_call_progress,
        )

    def test_progress_checkpoint_round_trip_restores_derived_runtime(self) -> None:
        source = _standard_schedule()
        for _ in range(3):
            source.record_successful_forward()

        checkpoint = source.state_dict()
        restored = _standard_schedule()
        restored.load_state_dict(checkpoint, strict=True)

        self.assertEqual(set(checkpoint), {"forward_call_progress"})
        self.assertEqual(restored.snapshot().forward_call_progress, 3)
        self.assertEqual(restored.active_iterations, 4)
        self.assertEqual(restored.active_transition_count, 4)

    def test_checkpoint_without_progress_loads_strictly_from_zero(self) -> None:
        restored = _standard_schedule()

        restored.load_state_dict({}, strict=True)

        self.assertEqual(restored.snapshot().forward_call_progress, 0)
        self.assertEqual(restored.active_iterations, 2)

    def test_checkpoint_progress_is_validated_and_clamped(self) -> None:
        schedule = _standard_schedule()
        malformed_progress = (
            (torch.tensor(-1, dtype=torch.long), ValueError, "must be non-negative"),
            (torch.tensor([1], dtype=torch.long), ValueError, "scalar Tensor"),
            (torch.tensor(1.0), TypeError, "torch.long dtype"),
            ("1", TypeError, "must be a Tensor"),
        )

        for value, error_type, message in malformed_progress:
            with self.subTest(value=value):
                with self.assertRaisesRegex(error_type, message):
                    schedule.load_state_dict(
                        {"forward_call_progress": value},
                        strict=True,
                    )

        schedule.load_state_dict(
            {"forward_call_progress": torch.tensor(999, dtype=torch.long)},
            strict=True,
        )
        self.assertEqual(schedule.snapshot().forward_call_progress, 9)
        self.assertEqual(schedule.active_iterations, 7)
