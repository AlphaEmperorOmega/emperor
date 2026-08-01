import copy
import unittest

import torch

from emperor.halting import (
    HaltingHiddenStateModeOptions,
    SoftHalting,
    SoftHaltingConfig,
    StickBreaking,
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
from support.sut_act_oracle import SutActOracle


def _configured_soft_halting(*, input_dim: int, threshold: float) -> SoftHalting:
    model = SoftHalting(
        SoftHaltingConfig(
            input_dim=input_dim,
            threshold=threshold,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        )
    ).double()
    model.eval()
    return model


class SoftHaltingSutParityTests(unittest.TestCase):
    def test_official_owner_interface_preserves_the_pinned_sut_recurrence(
        self,
    ) -> None:
        model = _configured_soft_halting(input_dim=2, threshold=0.9)
        with torch.no_grad():
            model._gate[0].weight.copy_(
                torch.tensor(((0.6, -0.3), (0.2, 0.7)), dtype=torch.float64)
            )
            model._gate[0].bias.copy_(torch.tensor((0.1, -0.2), dtype=torch.float64))
            model._gate[3].weight.copy_(
                torch.tensor(((0.8, -0.4), (-0.5, 0.9)), dtype=torch.float64)
            )

        oracle = SutActOracle(copy.deepcopy(model._gate), model.threshold)
        valid_mask = torch.ones(2, dtype=torch.bool)
        oracle_raw = torch.tensor(
            ((0.4, -0.7), (1.2, 0.3)),
            dtype=torch.float64,
        )
        candidates = (
            torch.tensor(((0.9, -0.2), (0.6, 1.1)), dtype=torch.float64),
            torch.tensor(((-0.3, 0.8), (1.4, -0.5)), dtype=torch.float64),
            torch.tensor(((0.2, 1.0), (-0.7, 0.4)), dtype=torch.float64),
            torch.tensor(((1.1, -0.6), (0.5, 0.9)), dtype=torch.float64),
        )
        local_state = None
        oracle_state = None

        for candidate in candidates:
            oracle_step = oracle.run_step(
                oracle_state,
                oracle_raw,
                valid_mask,
                lambda _raw, _context, _continuation, _mask, value=candidate: value,
            )
            local_state, local_output = model.update_halting_state(
                local_state,
                oracle_step.raw_hidden,
            )
            oracle_state = oracle_step.state
            oracle_raw = oracle_step.raw_hidden

            torch.testing.assert_close(local_state.raw_hidden, oracle_step.raw_hidden)
            torch.testing.assert_close(
                local_state.step_count,
                oracle_step.state.step_count,
            )
            torch.testing.assert_close(
                local_state.log_continuation,
                oracle_step.state.log_continuation,
            )
            torch.testing.assert_close(
                local_state.halt_probability,
                oracle_step.halt_mass,
            )
            torch.testing.assert_close(
                local_state.accumulated_hidden,
                oracle_step.state.accumulated_hidden,
            )
            torch.testing.assert_close(
                local_state.accumulated_ponder_cost,
                oracle_step.state.accumulated_expected_depth,
            )
            torch.testing.assert_close(
                local_state.continuation_probability,
                oracle_step.continuation,
            )
            torch.testing.assert_close(local_output, oracle_step.state.output_hidden)

        finalized_hidden, ponder_loss = model.finalize_weighted_accumulation(
            local_state,
            local_state.raw_hidden,
        )
        torch.testing.assert_close(finalized_hidden, oracle_state.output_hidden)
        torch.testing.assert_close(ponder_loss, oracle_step.loss)

        local_objective = finalized_hidden.square().sum() + ponder_loss
        oracle_objective = oracle_state.output_hidden.square().sum() + oracle_step.loss
        local_objective.backward()
        oracle_objective.backward()
        for (local_name, local_parameter), (
            oracle_name,
            oracle_parameter,
        ) in zip(
            model._gate.named_parameters(),
            oracle.gate.named_parameters(),
            strict=True,
        ):
            with self.subTest(gate_parameter=local_name):
                self.assertEqual(local_name, oracle_name)
                torch.testing.assert_close(
                    local_parameter.grad,
                    oracle_parameter.grad,
                )

    def test_official_owner_pipeline_matches_sut_for_computed_candidates(
        self,
    ) -> None:
        model = _configured_soft_halting(input_dim=3, threshold=0.82)
        with torch.no_grad():
            model._gate[0].weight.copy_(
                torch.tensor(
                    (
                        (0.7, -0.2, 0.4),
                        (-0.5, 0.9, 0.1),
                        (0.3, 0.2, -0.8),
                    ),
                    dtype=torch.float64,
                )
            )
            model._gate[0].bias.copy_(
                torch.tensor((0.15, -0.35, 0.25), dtype=torch.float64)
            )
            model._gate[3].weight.copy_(
                torch.tensor(
                    ((0.8, -0.6, 0.25), (-0.35, 0.75, -0.5)),
                    dtype=torch.float64,
                )
            )

        oracle = SutActOracle(copy.deepcopy(model._gate), model.threshold)
        valid_mask = torch.ones((2, 3), dtype=torch.bool)
        initial_hidden = torch.tensor(
            (
                ((0.4, -1.1, 0.7), (1.3, 0.2, -0.9), (3.0, -2.0, 1.0)),
                ((-0.8, 0.6, 1.4), (0.1, -0.5, 0.9), (1.7, 0.3, -1.2)),
            ),
            dtype=torch.float64,
        )
        step_biases = (
            torch.tensor((0.2, -0.1, 0.3), dtype=torch.float64),
            torch.tensor((-0.4, 0.5, 0.1), dtype=torch.float64),
            torch.tensor((0.6, 0.2, -0.3), dtype=torch.float64),
            torch.tensor((-0.2, -0.4, 0.7), dtype=torch.float64),
        )

        def compute_candidate(
            raw_hidden: torch.Tensor,
            context_hidden: torch.Tensor | None,
            step_index: int,
        ) -> torch.Tensor:
            context = raw_hidden if context_hidden is None else context_hidden
            mixed = (
                raw_hidden * (0.55 + 0.17 * step_index)
                + context.flip(-1) * (0.21 - 0.03 * step_index)
                + step_biases[step_index]
            )
            return torch.tanh(mixed)

        local_input = initial_hidden.clone().requires_grad_(True)
        oracle_input = initial_hidden.clone().requires_grad_(True)
        local_state = None
        oracle_state = None
        local_raw = local_input
        oracle_raw = oracle_input

        for step_index in range(4):
            local_context = None if local_state is None else local_state.output_hidden
            local_candidate = compute_candidate(
                local_raw,
                local_context,
                step_index,
            )
            local_state, local_output = model.update_halting_state(
                local_state,
                local_candidate,
            )
            oracle_step = oracle.run_step(
                oracle_state,
                oracle_raw,
                valid_mask,
                lambda raw, context, _continuation, _mask, index=step_index: (
                    compute_candidate(raw, context, index)
                ),
            )
            oracle_state = oracle_step.state
            local_raw = local_state.raw_hidden
            oracle_raw = oracle_step.raw_hidden

            torch.testing.assert_close(local_state.raw_hidden, oracle_step.raw_hidden)
            torch.testing.assert_close(
                local_state.log_continuation,
                oracle_step.state.log_continuation,
            )
            torch.testing.assert_close(
                local_state.accumulated_hidden,
                oracle_step.state.accumulated_hidden,
            )
            torch.testing.assert_close(
                local_state.accumulated_ponder_cost,
                oracle_step.state.accumulated_expected_depth,
            )
            torch.testing.assert_close(
                local_state.continuation_probability,
                oracle_step.continuation,
            )
            torch.testing.assert_close(local_output, oracle_step.state.output_hidden)

        local_output, local_loss = model.finalize_weighted_accumulation(
            local_state,
            local_raw,
        )
        torch.testing.assert_close(local_output, oracle_state.output_hidden)
        torch.testing.assert_close(local_loss, oracle_step.loss)

        local_objective = local_output.square().sum() + 0.37 * local_loss
        oracle_objective = (
            oracle_state.output_hidden.square().sum() + 0.37 * oracle_step.loss
        )
        local_objective.backward()
        oracle_objective.backward()
        torch.testing.assert_close(local_input.grad, oracle_input.grad)
        for name, local_parameter in model._gate.named_parameters():
            with self.subTest(gate_parameter=name):
                torch.testing.assert_close(
                    local_parameter.grad,
                    dict(oracle.gate.named_parameters())[name].grad,
                )


class StickBreakingOfficialLifecycleTests(unittest.TestCase):
    def test_official_lifecycle_preserves_the_stick_breaking_recurrence(self) -> None:
        gate_config = LayerStackConfig(
            input_dim=2,
            hidden_dim=2,
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
                layer_model_config=LinearLayerConfig(bias_flag=False),
            ),
        )
        model = StickBreaking(
            StickBreakingConfig(
                input_dim=2,
                threshold=0.99,
                dropout_probability=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=gate_config,
            )
        ).eval()
        first_candidate = torch.tensor(((2.0, 3.0), (4.0, 5.0)))
        second_candidate = first_candidate * 2.0

        state, _ = model.update_halting_state(None, first_candidate)
        state, _ = model.update_halting_state(state, second_candidate)
        output, loss = model.finalize_weighted_accumulation(
            state,
            second_candidate,
        )

        torch.testing.assert_close(
            state.accumulated_hidden,
            0.5 * first_candidate + 0.25 * second_candidate,
        )
        torch.testing.assert_close(
            output,
            0.5 * first_candidate + 0.5 * second_candidate,
        )
        torch.testing.assert_close(loss, torch.full((2,), 0.75))


if __name__ == "__main__":
    unittest.main()
