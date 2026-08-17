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
    RecurrentNestedSmoothHandoffExecutionPlan,
    RecurrentSmoothHandoffExecutionPlan,
)
from emperor.layers._composition.recurrent.validation.iteration_schedule import (
    RecurrentIterationScheduleValidator,
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
    def test_execution_plan_models_only_reachable_schedule_states(self) -> None:
        execution_plan = _standard_schedule().execution_plan()

        self.assertFalse(hasattr(execution_plan, "transitioning"))
        self.assertFalse(hasattr(execution_plan, "common_prefix_transition_count"))
        self.assertFalse(hasattr(execution_plan, "transition_weight"))
        self.assertFalse(
            hasattr(
                RecurrentIterationScheduleValidator,
                "validate_smooth_handoff_source_branch",
            )
        )

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
                self.assertNotIsInstance(
                    schedule.execution_plan(),
                    RecurrentSmoothHandoffExecutionPlan,
                )
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

    def test_smooth_growth_hands_depth_off_during_each_first_half_interval(
        self,
    ) -> None:
        schedule = _standard_schedule(
            max_steps=4,
            initial_iterations=2,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
        )

        observed = []
        for _ in range(11):
            snapshot = schedule.snapshot()
            observed.append(
                (
                    snapshot.forward_call_progress,
                    snapshot.settled_iterations,
                    snapshot.active_iterations,
                    snapshot.transitioning,
                    snapshot.transition_source_iterations,
                    snapshot.transition_target_iterations,
                    snapshot.transition_forward_index,
                    snapshot.transition_weight,
                    snapshot.complete,
                )
            )
            schedule.record_successful_forward()

        self.assertEqual(
            observed,
            [
                (0, 2, 2, False, None, None, None, 0.0, False),
                (1, 2, 2, False, None, None, None, 0.0, False),
                (2, 2, 2, False, None, None, None, 0.0, False),
                (3, 2, 2, False, None, None, None, 0.0, False),
                (4, 2, 3, True, 2, 3, 1, 0.5, False),
                (5, 2, 3, True, 2, 3, 2, 1.0, False),
                (6, 3, 3, False, None, None, None, 0.0, False),
                (7, 3, 3, False, None, None, None, 0.0, False),
                (8, 3, 4, True, 3, 4, 1, 0.5, False),
                (9, 3, 4, True, 3, 4, 2, 1.0, False),
                (10, 4, 4, False, None, None, None, 0.0, True),
            ],
        )

    def test_smooth_growth_matches_production_handoff_boundaries_without_replay(
        self,
    ) -> None:
        schedule = _standard_schedule(
            max_steps=5,
            initial_iterations=2,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=20_000,
            smooth_iteration_growth_flag=True,
        )
        expected_boundaries = {
            19_999: (2, 2, False, None, None, None, 0.0),
            20_000: (2, 3, True, 2, 3, 1, 0.0001),
            24_999: (2, 3, True, 2, 3, 5_000, 0.5),
            25_000: (2, 3, True, 2, 3, 5_001, 0.5001),
            29_999: (2, 3, True, 2, 3, 10_000, 1.0),
            30_000: (3, 3, False, None, None, None, 0.0),
            39_999: (3, 3, False, None, None, None, 0.0),
            40_000: (3, 4, True, 3, 4, 1, 0.0001),
        }

        for progress, expected in expected_boundaries.items():
            with self.subTest(progress=progress):
                schedule.load_state_dict(
                    {
                        "forward_call_progress": torch.tensor(
                            progress,
                            dtype=torch.long,
                        )
                    },
                    strict=True,
                )
                snapshot = schedule.snapshot()

                self.assertEqual(
                    (
                        snapshot.settled_iterations,
                        snapshot.active_iterations,
                        snapshot.transitioning,
                        snapshot.transition_source_iterations,
                        snapshot.transition_target_iterations,
                        snapshot.transition_forward_index,
                        snapshot.transition_weight,
                    ),
                    expected,
                )

    def test_smooth_execution_plan_uses_transition_units_for_every_variant(
        self,
    ) -> None:
        cases = (
            (
                RecurrentLayerConfig(
                    max_steps=3,
                    initial_iterations=2,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=True,
                ),
                (3, 0, 2, 0, 3, 1),
            ),
            (
                TinyRecursiveModelRecurrentConfig(
                    latent_updates_per_answer_update=2,
                    answer_update_count=2,
                    initial_iterations=1,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=True,
                ),
                (6, 1, 3, 1, 6, 4),
            ),
            (
                HierarchicalReasoningModelRecurrentConfig(
                    high_cycles=2,
                    low_cycles=2,
                    initial_iterations=1,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=True,
                ),
                (6, 1, 3, 1, 6, 4),
            ),
        )

        for config, expected in cases:
            with self.subTest(config_type=type(config).__name__):
                schedule = RecurrentIterationSchedule(config)
                for _ in range(4):
                    schedule.record_successful_forward()

                execution_plan = schedule.execution_plan()
                self.assertIsInstance(
                    execution_plan,
                    RecurrentSmoothHandoffExecutionPlan,
                )
                source_plan = execution_plan.source_branch
                self.assertEqual(
                    (
                        schedule.active_transition_count,
                        execution_plan.common_prefix_transition_count,
                        source_plan.transition_count,
                        source_plan.no_gradient_transition_count,
                        execution_plan.target_branch.transition_count,
                        execution_plan.target_branch.no_gradient_transition_count,
                    ),
                    expected,
                )
                self.assertEqual(execution_plan.transition_weight, 0.5)

    def test_full_gradient_smooth_execution_plan_uses_one_nested_chain(self) -> None:
        cases = (
            (
                RecurrentLayerConfig(
                    max_steps=3,
                    initial_iterations=2,
                    no_gradient_transition_count=0,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=True,
                ),
                (2, 3),
            ),
            (
                TinyRecursiveModelRecurrentConfig(
                    latent_updates_per_answer_update=2,
                    answer_update_count=2,
                    initial_iterations=1,
                    no_gradient_transition_count=0,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=True,
                ),
                (3, 6),
            ),
            (
                HierarchicalReasoningModelRecurrentConfig(
                    high_cycles=2,
                    low_cycles=2,
                    initial_iterations=1,
                    no_gradient_transition_count=0,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=True,
                ),
                (3, 6),
            ),
        )

        for config, expected_transition_counts in cases:
            with self.subTest(config_type=type(config).__name__):
                schedule = RecurrentIterationSchedule(config)
                for _ in range(4):
                    schedule.record_successful_forward()

                execution_plan = schedule.execution_plan()

                self.assertIsInstance(
                    execution_plan,
                    RecurrentNestedSmoothHandoffExecutionPlan,
                )
                self.assertNotIsInstance(
                    execution_plan,
                    RecurrentSmoothHandoffExecutionPlan,
                )
                self.assertEqual(
                    (
                        execution_plan.source_branch.transition_count,
                        execution_plan.target_branch.transition_count,
                    ),
                    expected_transition_counts,
                )
                self.assertEqual(
                    execution_plan.source_branch.no_gradient_transition_count,
                    0,
                )
                self.assertEqual(
                    execution_plan.target_branch.no_gradient_transition_count,
                    0,
                )
                self.assertEqual(execution_plan.transition_weight, 0.5)

    def test_explicit_zero_prefix_keeps_the_stable_execution_plan(self) -> None:
        schedule = _standard_schedule(
            no_gradient_transition_count=0,
            gradient_transition_count=None,
            smooth_iteration_growth_flag=False,
        )

        execution_plan = schedule.execution_plan()

        self.assertNotIsInstance(
            execution_plan,
            RecurrentNestedSmoothHandoffExecutionPlan,
        )
        self.assertNotIsInstance(
            execution_plan,
            RecurrentSmoothHandoffExecutionPlan,
        )
        self.assertEqual(
            execution_plan.target_branch.no_gradient_transition_count,
            0,
        )

    def test_smooth_schedule_constructor_rejects_an_odd_handoff_cadence(self):
        with self.assertRaisesRegex(
            ValueError,
            "even integer greater than or equal to 2",
        ):
            _standard_schedule(
                initial_iterations=2,
                gradient_transition_count=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=3,
                smooth_iteration_growth_flag=True,
            )

    def test_smooth_schedule_constructor_rejects_every_unsupported_control(self):
        invalid_cases = (
            (
                {"gradient_transition_count": None},
                "requires either gradient_transition_count or explicit "
                "no_gradient_transition_count=0",
            ),
            (
                {"iteration_increment": 2},
                "iteration_increment to equal 1",
            ),
            (
                {"no_gradient_transition_count": 0},
                "mutually exclusive",
            ),
            (
                {
                    "gradient_transition_count": None,
                    "no_gradient_transition_count": 1,
                },
                "only supports no_gradient_transition_count equal to 0",
            ),
        )

        for overrides, message in invalid_cases:
            values = {
                "max_steps": 4,
                "initial_iterations": 2,
                "gradient_transition_count": 2,
                "iteration_increment": 1,
                "forward_calls_before_iteration_increment": 4,
                "smooth_iteration_growth_flag": True,
            }
            values.update(overrides)
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(
                    ValueError,
                    message,
                ),
            ):
                _standard_schedule(**values)

    def test_smooth_schedule_rejects_invalid_gradient_windows_directly(self):
        cases = (
            (
                True,
                TypeError,
                "gradient_transition_count must be int",
            ),
            (
                3,
                ValueError,
                "minimum active transition count of 2",
            ),
        )

        for gradient_transition_count, error_type, message in cases:
            with self.subTest(gradient_transition_count=gradient_transition_count):
                with self.assertRaisesRegex(error_type, message):
                    _standard_schedule(
                        max_steps=3,
                        initial_iterations=2,
                        gradient_transition_count=gradient_transition_count,
                        iteration_increment=1,
                        forward_calls_before_iteration_increment=4,
                        smooth_iteration_growth_flag=True,
                    )

    def test_fixed_smooth_depth_is_an_immediately_complete_no_op(self):
        schedule = _standard_schedule(
            max_steps=2,
            initial_iterations=2,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
        )

        schedule.record_successful_forward()

        snapshot = schedule.snapshot()
        self.assertTrue(snapshot.smooth_iteration_growth)
        self.assertEqual(snapshot.settled_iterations, 2)
        self.assertEqual(snapshot.active_iterations, 2)
        self.assertFalse(snapshot.transitioning)
        self.assertTrue(snapshot.complete)
        self.assertEqual(snapshot.forward_call_progress, 0)

    def test_gradient_suffix_moves_as_active_depth_grows(self) -> None:
        schedule = _standard_schedule(
            max_steps=5,
            initial_iterations=3,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
        )

        branch_plan = schedule.execution_plan().target_branch
        observed_gradient_modes = []
        for transition_index in range(schedule.active_transition_count):
            with branch_plan.gradient_context(transition_index):
                observed_gradient_modes.append(torch.is_grad_enabled())

        self.assertEqual(observed_gradient_modes, [False, True, True])
        self.assertTrue(branch_plan.starts_gradient_suffix(1))
        self.assertFalse(branch_plan.tracks_gradients(0))
        self.assertFalse(hasattr(schedule, "gradient_context"))
        self.assertFalse(hasattr(schedule, "starts_gradient_suffix"))
        self.assertFalse(hasattr(schedule, "tracks_gradients"))

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

    def test_progress_buffer_is_the_only_schedule_runtime_state(self) -> None:
        schedule = _standard_schedule()

        schedule.forward_call_progress.fill_(3)

        snapshot = schedule.snapshot()
        self.assertEqual(snapshot.forward_call_progress, 3)
        self.assertEqual(snapshot.active_iterations, 4)
        self.assertEqual(schedule.execution_plan().target_branch.transition_count, 4)

    def test_smooth_checkpoint_round_trip_restores_every_handoff_phase(self):
        values = {
            "max_steps": 4,
            "initial_iterations": 2,
            "gradient_transition_count": 2,
            "iteration_increment": 1,
            "forward_calls_before_iteration_increment": 4,
            "smooth_iteration_growth_flag": True,
        }
        for progress in (3, 4, 5, 6, 8, 9, 10):
            with self.subTest(progress=progress):
                source = _standard_schedule(**values)
                for _ in range(progress):
                    source.record_successful_forward()

                restored = _standard_schedule(**values)
                restored.load_state_dict(source.state_dict(), strict=True)

                self.assertEqual(restored.snapshot(), source.snapshot())
                self.assertEqual(restored.execution_plan(), source.execution_plan())
                self.assertEqual(
                    set(restored.state_dict()),
                    {"forward_call_progress"},
                )

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
