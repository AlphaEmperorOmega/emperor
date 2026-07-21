import math
import unittest

import torch
from lightning import LightningModule, Trainer
from torch import Tensor

from emperor.halting import (
    HaltingBase,
    HaltingHiddenStateModeOptions,
    HaltingMonitorCallback,
    HaltingStateBase,
    HaltingUsageTracker,
    HaltingUsageTrackerManager,
    SoftHalting,
    SoftHaltingConfig,
    SoftHaltingState,
    StickBreaking,
    StickBreakingConfig,
    StickBreakingState,
)
from emperor.halting._monitoring.diagnostics import _HaltingDiagnostics
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from support.monitor import orchestration_calls


def gate_config(input_dim: int = 2) -> LayerStackConfig:
    return LayerStackConfig(
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
    )


def stick_model(input_dim: int = 2) -> StickBreaking:
    return StickBreaking(
        StickBreakingConfig(
            input_dim=input_dim,
            threshold=1.0,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=gate_config(input_dim),
        )
    ).eval()


def soft_model(input_dim: int = 2) -> SoftHalting:
    return SoftHalting(
        SoftHaltingConfig(
            input_dim=input_dim,
            threshold=0.99,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=gate_config(input_dim),
        )
    ).eval()


def stick_state(
    *,
    halt_mask: Tensor,
    accumulated_probabilities: Tensor,
    ponder_cost: Tensor,
) -> StickBreakingState:
    hidden = torch.zeros(*halt_mask.shape, 2)
    valid_mask = torch.ones_like(halt_mask, dtype=torch.bool)
    state = StickBreakingState(
        halt_mask=halt_mask,
        log_continuation=torch.zeros_like(accumulated_probabilities),
        accumulated_hidden=hidden,
        output_hidden=hidden.clone(),
        accumulated_halt_probabilities=accumulated_probabilities,
        step_count=1,
        accumulated_ponder_cost=ponder_cost,
    )
    state.raw_hidden = hidden.clone()
    state.continuation_probability = 1.0 - accumulated_probabilities
    state.valid_mask = valid_mask
    state.stop_requested = False
    state.step_indices = torch.ones_like(accumulated_probabilities)
    state.advanced_mask = valid_mask
    return state


def base_state(
    continuation_probability: Tensor,
    *,
    halt_mask: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> HaltingStateBase:
    halt_mask = (
        torch.zeros_like(continuation_probability, dtype=torch.bool)
        if halt_mask is None
        else halt_mask
    )
    valid_mask = (
        torch.ones_like(continuation_probability, dtype=torch.bool)
        if valid_mask is None
        else valid_mask
    )
    hidden = continuation_probability.new_zeros(*continuation_probability.shape, 2)
    return HaltingStateBase(
        output_hidden=hidden.clone(),
        accumulated_hidden=hidden.clone(),
        continuation_probability=continuation_probability,
        halt_mask=halt_mask,
        valid_mask=valid_mask,
        stop_requested=False,
    )


class _LifecycleHalting(HaltingBase[HaltingStateBase]):
    def update_halting_state(
        self,
        previous_state: HaltingStateBase | None,
        model_hidden_state: Tensor,
    ) -> tuple[HaltingStateBase, Tensor]:
        if previous_state is not None:
            return previous_state, model_hidden_state
        state = base_state(model_hidden_state.new_ones(model_hidden_state.shape[:-1]))
        return state, model_hidden_state

    def finalize_weighted_accumulation(
        self,
        state: HaltingStateBase,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return state.output_hidden, current_hidden.new_zeros(())


class _HaltingOwner(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.halting = stick_model()


class _NoHaltingOwner(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


class _TwoHaltingOwner(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.first = stick_model()
        self.second = stick_model()
        self.logged_names: list[str] = []

    def log(
        self,
        name: str,
        value: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.logged_names.append(name)


class HaltingUsageTrackerTests(unittest.TestCase):
    def test_fresh_tracker_has_exact_empty_dynamic_and_zero_scalar_state(self) -> None:
        tracker = HaltingUsageTracker()

        self.assertEqual(
            tuple(tracker.state_dict()),
            (
                "last_survival",
                "last_ponder_cost_mean",
                "last_ponder_cost_std",
                "last_ponder_cost",
                "last_step_count",
                "last_halted_fraction",
                "last_accumulated_halt_prob_mean",
                "last_remaining_mass_mean",
                "last_ponder_loss",
            ),
        )
        self.assertEqual(tuple(tracker.last_survival.shape), (0,))
        self.assertEqual(tuple(tracker.last_ponder_cost.shape), (0,))
        for name in (
            "last_ponder_cost_mean",
            "last_ponder_cost_std",
            "last_step_count",
            "last_halted_fraction",
            "last_accumulated_halt_prob_mean",
            "last_remaining_mass_mean",
            "last_ponder_loss",
        ):
            torch.testing.assert_close(getattr(tracker, name), torch.zeros(()))
        self.assertEqual(tracker._survival_stage, [])

    def test_records_exact_stick_breaking_survival_and_final_scalars(self) -> None:
        tracker = HaltingUsageTracker()
        first = stick_state(
            halt_mask=torch.tensor([[False, True], [False, False]]),
            accumulated_probabilities=torch.tensor([[0.4, 1.0], [0.2, 0.3]]),
            ponder_cost=torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        )
        final = stick_state(
            halt_mask=torch.tensor([[True, True], [False, True]]),
            accumulated_probabilities=torch.tensor([[0.8, 1.0], [0.25, 0.9]]),
            ponder_cost=torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        )

        tracker.begin_forward()
        tracker.record_step(first)
        tracker.record_step(final)
        tracker.record_final(torch.tensor([2.0, 4.0]), final)

        torch.testing.assert_close(
            tracker.last_survival,
            torch.tensor([0.75, 0.25]),
        )
        torch.testing.assert_close(tracker.last_ponder_cost_mean, torch.tensor(4.0))
        torch.testing.assert_close(
            tracker.last_ponder_cost_std,
            torch.tensor(math.sqrt(5.0)),
        )
        torch.testing.assert_close(
            tracker.last_ponder_cost,
            torch.tensor([1.0, 3.0, 5.0, 7.0]),
        )
        torch.testing.assert_close(tracker.last_step_count, torch.tensor(2.0))
        torch.testing.assert_close(
            tracker.last_halted_fraction,
            torch.tensor(0.75),
        )
        torch.testing.assert_close(
            tracker.last_accumulated_halt_prob_mean,
            torch.tensor(0.7375),
        )
        torch.testing.assert_close(
            tracker.last_remaining_mass_mean,
            torch.tensor(0.1875),
        )
        torch.testing.assert_close(tracker.last_ponder_loss, torch.tensor(3.0))

    def test_soft_state_uses_normalized_halt_and_valid_masks(self) -> None:
        tracker = HaltingUsageTracker()
        hidden = torch.zeros(3, 2)
        state = SoftHaltingState(
            raw_hidden=hidden,
            output_hidden=hidden.clone(),
            step_count=torch.full((3,), 2.0),
            log_continuation=torch.zeros(3),
            accumulated_hidden=hidden.clone(),
            accumulated_ponder_cost=torch.tensor([2.0, 3.0, 1.0]),
            continuation_probability=torch.tensor([0.0, 0.5, 1.0]),
            halt_mask=torch.tensor([True, False, False]),
            valid_mask=torch.tensor([True, True, False]),
            stop_requested=False,
            halt_probability=torch.tensor([1.0, 0.5, 0.0]),
            gate_input=None,
            gate_logits=None,
            advanced_mask=torch.tensor([True, True, False]),
        )

        tracker.begin_forward()
        tracker.record_step(state)
        tracker.record_final(torch.tensor(0.4), state)

        torch.testing.assert_close(tracker.last_survival, torch.tensor([0.5]))
        torch.testing.assert_close(tracker.last_step_count, torch.tensor(1.0))
        torch.testing.assert_close(tracker.last_halted_fraction, torch.tensor(0.5))
        torch.testing.assert_close(tracker.last_ponder_cost_mean, torch.tensor(2.5))
        torch.testing.assert_close(
            tracker.last_ponder_cost,
            torch.tensor([2.0, 3.0]),
        )
        torch.testing.assert_close(tracker.last_ponder_loss, torch.tensor(0.4))

    def test_survival_commit_uses_the_tracker_buffer_dtype(self) -> None:
        tracker = HaltingUsageTracker().double()
        state = base_state(
            torch.tensor(
                [0.25, 0.75],
                dtype=torch.float32,
            )
        )

        tracker.begin_forward()
        tracker.record_step(state)
        tracker.record_final(None, state)

        self.assertEqual(tracker.last_survival.dtype, torch.float64)
        torch.testing.assert_close(
            tracker.last_survival,
            torch.tensor([1.0], dtype=torch.float64),
        )

    def test_base_state_uses_normalized_masks_and_empty_optional_metrics(
        self,
    ) -> None:
        tracker = HaltingUsageTracker()
        state = base_state(torch.ones(2))

        tracker.begin_forward()
        tracker.record_step(state)
        tracker.record_final(None, state)

        torch.testing.assert_close(tracker.last_survival, torch.ones(1))
        torch.testing.assert_close(tracker.last_step_count, torch.tensor(1.0))
        torch.testing.assert_close(tracker.last_ponder_cost_mean, torch.zeros(()))
        torch.testing.assert_close(tracker.last_halted_fraction, torch.zeros(()))
        torch.testing.assert_close(
            tracker.last_accumulated_halt_prob_mean,
            torch.zeros(()),
        )
        torch.testing.assert_close(tracker.last_remaining_mass_mean, torch.zeros(()))
        torch.testing.assert_close(tracker.last_ponder_loss, torch.zeros(()))

    def test_accumulated_probabilities_without_halt_mask_keep_all_remaining_mass(
        self,
    ) -> None:
        tracker = HaltingUsageTracker()
        state = base_state(torch.tensor([0.75, 0.25]))
        state.__dict__["accumulated_halt_probabilities"] = torch.tensor([0.25, 0.75])

        tracker.begin_forward()
        tracker.record_final(None, state)

        torch.testing.assert_close(
            tracker.last_accumulated_halt_prob_mean,
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            tracker.last_remaining_mass_mean,
            torch.tensor(0.5),
        )

    def test_all_invalid_state_records_empty_finite_metrics(self) -> None:
        tracker = HaltingUsageTracker()
        state = stick_state(
            halt_mask=torch.tensor([False, True]),
            accumulated_probabilities=torch.tensor([0.25, 0.75]),
            ponder_cost=torch.tensor([2.0, 4.0]),
        )
        state.valid_mask = torch.zeros(2, dtype=torch.bool)

        tracker.begin_forward()
        tracker.record_step(state)
        tracker.record_final(torch.tensor(0.0), state)

        torch.testing.assert_close(tracker.last_survival, torch.zeros(1))
        self.assertEqual(tuple(tracker.last_ponder_cost.shape), (0,))
        torch.testing.assert_close(tracker.last_ponder_cost_mean, torch.zeros(()))
        torch.testing.assert_close(tracker.last_ponder_cost_std, torch.zeros(()))
        torch.testing.assert_close(tracker.last_halted_fraction, torch.zeros(()))
        torch.testing.assert_close(
            tracker.last_accumulated_halt_prob_mean,
            torch.zeros(()),
        )
        torch.testing.assert_close(tracker.last_remaining_mass_mean, torch.zeros(()))

    def test_final_without_steps_commits_an_empty_survival_curve(self) -> None:
        tracker = HaltingUsageTracker()

        tracker.begin_forward()
        tracker.record_final(None, base_state(torch.ones(1)))

        self.assertEqual(tuple(tracker.last_survival.shape), (0,))
        torch.testing.assert_close(tracker.last_step_count, torch.zeros(()))

    def test_reset_clears_every_buffer_and_staged_value(self) -> None:
        tracker = HaltingUsageTracker()
        state = stick_state(
            halt_mask=torch.tensor([True, False]),
            accumulated_probabilities=torch.tensor([0.9, 0.25]),
            ponder_cost=torch.tensor([3.0, 5.0]),
        )
        tracker.begin_forward()
        tracker.record_step(state)
        tracker.record_final(torch.tensor(2.0), state)
        tracker.record_step(state)

        tracker.reset()

        self.assertEqual(tuple(tracker.last_survival.shape), (0,))
        self.assertEqual(tuple(tracker.last_ponder_cost.shape), (0,))
        for name in (
            "last_ponder_cost_mean",
            "last_ponder_cost_std",
            "last_step_count",
            "last_halted_fraction",
            "last_accumulated_halt_prob_mean",
            "last_remaining_mass_mean",
            "last_ponder_loss",
        ):
            torch.testing.assert_close(getattr(tracker, name), torch.zeros(()))
        self.assertEqual(tracker._survival_stage, [])

    def test_strict_state_roundtrip_resizes_both_dynamic_buffers(self) -> None:
        source = HaltingUsageTracker()
        state = stick_state(
            halt_mask=torch.tensor([[False, True], [False, False]]),
            accumulated_probabilities=torch.tensor([[0.4, 1.0], [0.2, 0.3]]),
            ponder_cost=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )
        source.begin_forward()
        source.record_step(state)
        source.record_step(state)
        source.record_final(torch.tensor(1.25), state)
        saved = {name: value.clone() for name, value in source.state_dict().items()}

        restored = HaltingUsageTracker()
        restored.load_state_dict(saved, strict=True)

        self.assertEqual(tuple(restored.last_survival.shape), (2,))
        self.assertEqual(tuple(restored.last_ponder_cost.shape), (4,))
        for name, value in saved.items():
            torch.testing.assert_close(restored.state_dict()[name], value)

        restored.load_state_dict(saved, strict=True)
        incomplete = dict(saved)
        del incomplete["last_survival"]
        result = restored.load_state_dict(incomplete, strict=False)
        self.assertEqual(result.missing_keys, ["last_survival"])

        unexpected = dict(saved)
        unexpected["unexpected_metric"] = torch.ones(())
        result = restored.load_state_dict(unexpected, strict=False)
        self.assertEqual(result.unexpected_keys, ["unexpected_metric"])

        malformed = dict(saved)
        malformed["last_step_count"] = torch.zeros(2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"size mismatch for last_step_count",
        ):
            restored.load_state_dict(malformed, strict=True)

    def test_diagnostics_detach_float_metrics_and_handle_empty_survival(self) -> None:
        tracker = HaltingUsageTracker().double()
        tracker.last_ponder_cost_mean.fill_(2.0)
        tracker.last_ponder_cost_std.fill_(0.5)
        tracker.last_ponder_cost = torch.tensor([1.0, 3.0], dtype=torch.float64)
        tracker.last_step_count.fill_(4.0)
        tracker.last_halted_fraction.fill_(0.75)
        tracker.last_accumulated_halt_prob_mean.fill_(0.8)
        tracker.last_remaining_mass_mean.fill_(0.1)
        tracker.last_ponder_loss.fill_(1.25)
        tracker.last_survival = torch.tensor([0.8, 0.3], dtype=torch.float64)

        metrics = _HaltingDiagnostics.calculate(tracker)

        torch.testing.assert_close(
            metrics.final_survival_fraction,
            torch.tensor(0.3),
        )
        for value in (
            metrics.ponder_cost_mean,
            metrics.ponder_cost_std,
            metrics.ponder_cost,
            metrics.step_count,
            metrics.halted_fraction,
            metrics.accumulated_halt_probability_mean,
            metrics.remaining_mass_mean,
            metrics.final_survival_fraction,
            metrics.ponder_loss,
            metrics.survival,
        ):
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)


class HaltingUsageTrackerManagerTests(unittest.TestCase):
    def test_real_stick_breaking_attachment_records_and_restores_methods(self) -> None:
        model = stick_model()
        manager = HaltingUsageTrackerManager()
        tracker = manager.attach(model)
        self.assertIs(manager.attach(model), tracker)
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        state, _ = model.update_halting_state(None, hidden)
        state, _ = model.update_halting_state(
            state,
            hidden * 2,
        )
        output, ponder_loss = model.finalize_weighted_accumulation(
            state,
            hidden * 2,
        )

        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(ponder_loss.shape, torch.Size([2]))
        torch.testing.assert_close(tracker.last_step_count, torch.tensor(2.0))
        self.assertEqual(tuple(tracker.last_survival.shape), (2,))
        self.assertIn("update_halting_state", model.__dict__)
        self.assertIn("finalize_weighted_accumulation", model.__dict__)

        manager.detach(model)

        self.assertNotIn("update_halting_state", model.__dict__)
        self.assertNotIn("finalize_weighted_accumulation", model.__dict__)
        self.assertNotIn("_usage_tracker", model._modules)
        manager.detach(model)

    def test_soft_halting_is_rejected_until_it_implements_the_supported_interface(
        self,
    ) -> None:
        model = soft_model()
        manager = HaltingUsageTrackerManager()

        with self.assertRaisesRegex(TypeError, "supported halting interface"):
            manager.attach(model)

    def test_attach_accepts_any_concrete_common_lifecycle(self) -> None:
        model = _LifecycleHalting()
        manager = HaltingUsageTrackerManager()

        tracker = manager.attach(model)

        self.assertIs(tracker, model._usage_tracker)
        manager.detach(model)

    def test_attach_rolls_back_wrappers_when_module_registration_fails(self) -> None:
        model = stick_model()
        model._usage_tracker = None

        with self.assertRaisesRegex(
            KeyError,
            r"attribute '_usage_tracker' already exists",
        ):
            HaltingUsageTrackerManager().attach(model)

        self.assertNotIn("update_halting_state", model.__dict__)
        self.assertNotIn("finalize_weighted_accumulation", model.__dict__)
        self.assertNotIn("_usage_tracker", model._modules)

    def test_detach_tolerates_an_externally_removed_tracker_module(self) -> None:
        model = stick_model()
        manager = HaltingUsageTrackerManager()
        manager.attach(model)
        del model._modules["_usage_tracker"]

        manager.detach(model)

        self.assertNotIn("update_halting_state", model.__dict__)
        self.assertNotIn("finalize_weighted_accumulation", model.__dict__)

    def test_base_requires_supported_strategy_methods(self) -> None:
        adapter = HaltingBase()

        with self.assertRaises(NotImplementedError):
            adapter.update_halting_state(
                None,
                torch.ones(1, 2),
            )
        with self.assertRaises(NotImplementedError):
            adapter.finalize_weighted_accumulation(
                base_state(torch.ones(1)),
                torch.ones(1, 2),
            )


class HaltingMonitorCallbackUnitTests(unittest.TestCase):
    def test_tracking_orchestration_lists_each_diagnostic(self) -> None:
        orchestration = (
            HaltingMonitorCallback._HaltingMonitorCallback__track_halting_diagnostics
        )
        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_ponder_cost_mean",
                "__track_ponder_cost_standard_deviation",
                "__track_step_count",
                "__track_halted_fraction",
                "__track_accumulated_halt_probability_mean",
                "__track_remaining_mass_mean",
                "__track_saturation_fraction",
                "__track_ponder_loss",
                "__track_survival_history",
                "__track_survival_histogram",
                "__track_ponder_cost_histogram",
                "__track_survival_heatmap",
            ),
        )

    def test_constructor_preserves_values_and_rejects_invalid_options(self) -> None:
        defaults = HaltingMonitorCallback()
        self.assertEqual(defaults.log_every_n_steps, 100)
        self.assertEqual(defaults.history_size, 128)

        callback = HaltingMonitorCallback(log_every_n_steps=3, history_size=7)
        self.assertEqual(callback.log_every_n_steps, 3)
        self.assertEqual(callback.history_size, 7)

        for option_name in ("log_every_n_steps", "history_size"):
            for value in (True, 1.5, "2"):
                with self.subTest(option_name=option_name, value=value):
                    with self.assertRaisesRegex(
                        TypeError,
                        rf"^{option_name} must be a positive integer, "
                        rf"received {type(value).__name__}\.$",
                    ):
                        HaltingMonitorCallback(**{option_name: value})
            for value in (0, -1):
                with self.subTest(option_name=option_name, value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"^{option_name} must be greater than 0\.$",
                    ):
                        HaltingMonitorCallback(**{option_name: value})

    def test_non_fit_setup_and_owner_without_halting_attach_nothing(self) -> None:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        callback = HaltingMonitorCallback()
        callback.setup(trainer, _HaltingOwner(), "validate")
        self.assertIsNone(callback._tracker_manager)

        callback.on_fit_start(trainer, _NoHaltingOwner())
        self.assertIsNotNone(callback._tracker_manager)
        self.assertEqual(callback._halting_layers, [])
        self.assertEqual(callback._survival_history, {})
        callback.on_fit_end(trainer, _NoHaltingOwner())

    def test_repeated_fit_start_reuses_one_real_tracker(self) -> None:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        owner = _HaltingOwner()
        callback = HaltingMonitorCallback()

        callback.on_fit_start(trainer, owner)
        tracker = owner.halting._usage_tracker
        callback.on_fit_start(trainer, owner)

        self.assertEqual(
            [(name, type(module)) for name, module in callback._halting_layers],
            [("halting", StickBreaking)],
        )
        self.assertIs(owner.halting._usage_tracker, tracker)
        self.assertEqual(
            callback._survival_history["halting"]._normalization,
            "unit_interval",
        )
        callback.on_fit_end(trainer, owner)

    def test_missing_first_tracker_does_not_prevent_second_module_logging(
        self,
    ) -> None:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        owner = _TwoHaltingOwner()
        callback = HaltingMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(trainer, owner)
        callback._tracker_manager.detach(owner.first)

        callback.on_train_batch_end(trainer, owner, None, None, 0)

        self.assertEqual(len(owner.logged_names), 8)
        self.assertTrue(all(name.startswith("second/") for name in owner.logged_names))
        callback.on_fit_end(trainer, owner)

    def test_missing_tracker_is_skipped_and_cleanup_without_setup_is_safe(
        self,
    ) -> None:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        owner = _HaltingOwner()
        callback = HaltingMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(trainer, owner)
        callback._tracker_manager.detach(owner.halting)

        callback.on_train_batch_end(trainer, owner, None, None, 0)
        callback.on_fit_end(trainer, owner)
        callback.on_fit_end(trainer, owner)

        self.assertIsNone(callback._tracker_manager)
        self.assertEqual(callback._halting_layers, [])
        self.assertEqual(callback._survival_history, {})

    def test_fit_setup_allows_strict_owner_state_roundtrip_with_tracker(self) -> None:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        source = _HaltingOwner()
        source_callback = HaltingMonitorCallback()
        source_callback.setup(trainer, source, "fit")
        hidden = torch.ones(3, 2)
        state, _ = source.halting.update_halting_state(None, hidden)
        state, _ = source.halting.update_halting_state(
            state,
            hidden,
        )
        source.halting.finalize_weighted_accumulation(
            state,
            hidden,
        )
        saved = {name: value.clone() for name, value in source.state_dict().items()}

        restored = _HaltingOwner()
        restored_callback = HaltingMonitorCallback()
        restored_callback.setup(trainer, restored, "fit")
        restored.load_state_dict(saved, strict=True)

        self.assertEqual(
            tuple(restored.halting._usage_tracker.last_survival.shape),
            (2,),
        )
        self.assertEqual(
            tuple(restored.halting._usage_tracker.last_ponder_cost.shape),
            (3,),
        )
        for name, value in saved.items():
            torch.testing.assert_close(restored.state_dict()[name], value)

        source_callback.on_fit_end(trainer, source)
        restored_callback.on_fit_end(trainer, restored)


if __name__ == "__main__":
    unittest.main()
