import copy
import unittest

import torch
from emperor.halting import (
    HaltingHiddenStateModeOptions,
    SoftHalting,
    SoftHaltingConfig,
)

from support.sut_act_oracle import SutActOracle


class SoftHaltingSutParityTests(unittest.TestCase):
    def test_soft_lifecycle_matches_pinned_sut_after_every_step(self) -> None:
        model = SoftHalting(
            SoftHaltingConfig(
                input_dim=3,
                threshold=0.82,
                dropout_probability=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            )
        ).double()
        model.eval()
        with torch.no_grad():
            model._gate[0].weight.copy_(
                torch.tensor(
                    [
                        [0.7, -0.2, 0.4],
                        [-0.5, 0.9, 0.1],
                        [0.3, 0.2, -0.8],
                    ],
                    dtype=torch.float64,
                )
            )
            model._gate[0].bias.copy_(
                torch.tensor([0.15, -0.35, 0.25], dtype=torch.float64)
            )
            model._gate[3].weight.copy_(
                torch.tensor(
                    [[0.8, -0.6, 0.25], [-0.35, 0.75, -0.5]],
                    dtype=torch.float64,
                )
            )

        oracle = SutActOracle(copy.deepcopy(model._gate), model.threshold)
        valid_mask = torch.tensor(
            [[True, True, False], [True, True, True]],
        )
        initial_hidden = torch.tensor(
            [
                [[0.4, -1.1, 0.7], [1.3, 0.2, -0.9], [3.0, -2.0, 1.0]],
                [[-0.8, 0.6, 1.4], [0.1, -0.5, 0.9], [1.7, 0.3, -1.2]],
            ],
            dtype=torch.float64,
        )
        step_biases = (
            torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64),
            torch.tensor([-0.4, 0.5, 0.1], dtype=torch.float64),
            torch.tensor([0.6, 0.2, -0.3], dtype=torch.float64),
            torch.tensor([-0.2, -0.4, 0.7], dtype=torch.float64),
        )

        def candidate(
            raw_hidden,
            context_hidden,
            continuation,
            computation_mask,
            step_index,
        ):
            context = raw_hidden if context_hidden is None else context_hidden
            mixed = (
                raw_hidden * (0.55 + 0.17 * step_index)
                + context.flip(-1) * (0.21 - 0.03 * step_index)
                + continuation.unsqueeze(-1) * step_biases[step_index]
            )
            computed = torch.tanh(mixed)
            return torch.where(
                computation_mask.unsqueeze(-1),
                computed,
                raw_hidden,
            )

        local_input = initial_hidden.clone().requires_grad_(True)
        oracle_input = initial_hidden.clone().requires_grad_(True)
        local_state = None
        oracle_state = None
        local_raw = local_input
        oracle_raw = oracle_input
        local_records = []
        computation_masks = []

        for step_index in range(4):

            def local_compute(record, index=step_index):
                local_records.append(record)
                return candidate(
                    record.raw_hidden,
                    record.context_hidden,
                    record.continuation_probability,
                    record.computation_mask,
                    index,
                )

            local_state = model.run_step(
                local_state,
                local_raw,
                local_compute,
                valid_mask=valid_mask,
            )
            oracle_step = oracle.run_step(
                oracle_state,
                oracle_raw,
                valid_mask,
                lambda raw, context, continuation, mask, index=step_index: candidate(
                    raw,
                    context,
                    continuation,
                    mask,
                    index,
                ),
            )
            oracle_state = oracle_step.state

            self.assertEqual(local_state.stop_requested, False)
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
            torch.testing.assert_close(
                local_state.output_hidden,
                oracle_step.state.output_hidden,
            )
            torch.testing.assert_close(local_state.raw_hidden, oracle_step.raw_hidden)
            self.assertEqual(
                local_state.gate_input is None,
                oracle_step.gate_input is None,
            )
            if oracle_step.gate_input is not None:
                torch.testing.assert_close(
                    local_state.gate_input,
                    oracle_step.gate_input,
                )
                torch.testing.assert_close(
                    local_state.gate_logits,
                    oracle_step.gate_logits,
                )
            self.assertTrue(
                torch.equal(
                    local_records[-1].computation_mask,
                    oracle_step.computation_mask,
                )
            )
            computation_masks.append(local_records[-1].computation_mask)

            local_raw = local_state.raw_hidden
            oracle_raw = oracle_step.raw_hidden

        local_output, local_loss = model.finalize(local_state, local_raw)
        torch.testing.assert_close(local_output, oracle_state.output_hidden)
        torch.testing.assert_close(local_loss, oracle_step.loss)

        all_computation_masks = torch.stack(computation_masks)
        expanded_valid_mask = valid_mask.unsqueeze(0).expand_as(all_computation_masks)
        self.assertTrue((all_computation_masks & expanded_valid_mask).any().item())
        self.assertTrue((~all_computation_masks & expanded_valid_mask).any().item())

        local_objective = local_output.square().sum() + 0.37 * local_loss
        oracle_objective = (
            oracle_state.output_hidden.square().sum() + 0.37 * oracle_step.loss
        )
        local_objective.backward()
        oracle_objective.backward()

        torch.testing.assert_close(local_input.grad, oracle_input.grad)
        local_gate_parameters = dict(model._gate.named_parameters())
        oracle_gate_parameters = dict(oracle.gate.named_parameters())
        self.assertEqual(local_gate_parameters.keys(), oracle_gate_parameters.keys())
        for name, local_parameter in local_gate_parameters.items():
            with self.subTest(gate_parameter=name):
                self.assertIsNotNone(local_parameter.grad)
                self.assertIsNotNone(oracle_gate_parameters[name].grad)
                torch.testing.assert_close(
                    local_parameter.grad,
                    oracle_gate_parameters[name].grad,
                )


if __name__ == "__main__":
    unittest.main()
