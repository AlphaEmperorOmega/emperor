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


def gate_config(input_dim: int = 2, *, num_layers: int = 2) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=input_dim,
        output_dim=2,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def strategies(input_dim: int = 2):
    common = {
        "input_dim": input_dim,
        "threshold": 0.9,
        "dropout_probability": None,
        "hidden_state_mode": HaltingHiddenStateModeOptions.RAW,
    }
    return (
        StickBreaking(
            StickBreakingConfig(
                **common,
                halting_gate_config=gate_config(input_dim, num_layers=1),
            )
        ).eval(),
        SoftHalting(
            SoftHaltingConfig(
                **common,
                halting_gate_config=None,
            )
        ).eval(),
    )


class HaltingRegressionTests(unittest.TestCase):
    def test_non_contiguous_hidden_matches_contiguous_hidden(self) -> None:
        non_contiguous = torch.arange(24, dtype=torch.float64).reshape(2, 2, 6)[
            ..., ::3
        ]
        self.assertFalse(non_contiguous.is_contiguous())
        contiguous = non_contiguous.contiguous()

        for model in strategies():
            model = model.double()
            with self.subTest(strategy=type(model).__name__):
                first = model.run_step(
                    None,
                    non_contiguous,
                    lambda computation: computation.raw_hidden,
                )
                second_model = (
                    StickBreaking(model.cfg).double().eval()
                    if isinstance(model, StickBreaking)
                    else SoftHalting(model.cfg).double().eval()
                )
                second_model.load_state_dict(model.state_dict(), strict=True)
                second = second_model.run_step(
                    None,
                    contiguous,
                    lambda computation: computation.raw_hidden,
                )
                torch.testing.assert_close(first.output_hidden, second.output_hidden)
                torch.testing.assert_close(
                    first.continuation_probability,
                    second.continuation_probability,
                )

    def test_omitted_later_valid_mask_restores_the_permanent_domain(self) -> None:
        hidden = torch.ones(2, 2)
        valid = torch.tensor([True, False])
        for model in strategies():
            with self.subTest(strategy=type(model).__name__):
                state = model.run_step(
                    None,
                    hidden,
                    lambda computation: computation.raw_hidden,
                    valid_mask=valid,
                )
                state = model.run_step(
                    state,
                    state.raw_hidden,
                    lambda computation: computation.raw_hidden + 1,
                )
                self.assertTrue(torch.equal(state.valid_mask, valid))
                torch.testing.assert_close(state.raw_hidden[1], hidden[1])

    def test_update_mask_uses_the_same_binary_validation_as_valid_mask(self) -> None:
        hidden = torch.ones(2, 2)
        for model in strategies():
            for update_mask, error_type in (
                ([True, False], TypeError),
                (torch.tensor([1.0, 0.25]), ValueError),
                (torch.tensor([1.0, float("inf")]), ValueError),
                (torch.ones(2, 1), ValueError),
            ):
                with self.subTest(
                    strategy=type(model).__name__,
                    update_mask=update_mask,
                ):
                    with self.assertRaises(error_type):
                        model.run_step(
                            None,
                            hidden,
                            lambda computation: computation.raw_hidden,
                            update_mask=update_mask,
                        )

    def test_compute_callback_must_return_a_tensor(self) -> None:
        for model in strategies():
            with self.subTest(strategy=type(model).__name__):
                with self.assertRaisesRegex(TypeError, "compute_step result"):
                    model.run_step(None, torch.ones(2, 2), lambda _computation: None)

    def test_none_soft_dropout_is_a_runtime_no_op_for_custom_gate(self) -> None:
        cfg = SoftHaltingConfig(
            input_dim=2,
            threshold=0.9,
            dropout_probability=None,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=gate_config(),
        )
        model = SoftHalting(cfg).train()
        hidden = torch.tensor([[1.0, -2.0], [0.5, 1.5]])
        with torch.no_grad():
            model._gate[0].model.weight_params.copy_(
                torch.tensor(((0.7, -0.2), (-0.4, 0.5)))
            )
            model._gate[0].model.bias_params.copy_(torch.tensor((0.1, -0.3)))
            model._gate[-1].model.weight_params.copy_(
                torch.tensor(((0.6, -0.1), (-0.8, 0.4)))
            )

        first = model._SoftHalting__compute_gate_logits(hidden)
        second = model._SoftHalting__compute_gate_logits(hidden)

        torch.testing.assert_close(first, second)
        projected_hidden = torch.nn.functional.relu(
            torch.nn.functional.linear(
                hidden,
                model._gate[0].model.weight_params.T,
                model._gate[0].model.bias_params,
            )
        )
        torch.testing.assert_close(
            first,
            torch.nn.functional.log_softmax(
                torch.nn.functional.linear(
                    projected_hidden,
                    model._gate[-1].model.weight_params.T,
                ),
                dim=-1,
            ),
        )

    def test_custom_soft_dropout_is_dedicated_before_the_output_projection(
        self,
    ) -> None:
        cfg = SoftHaltingConfig(
            input_dim=2,
            threshold=0.9,
            dropout_probability=1.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=gate_config(),
        )
        model = SoftHalting(cfg).train()
        with torch.no_grad():
            for parameter in model._gate.parameters():
                parameter.fill_(0.5)
        hidden = torch.tensor([[1.0, -2.0], [0.5, 1.5]])

        logits = model._SoftHalting__compute_gate_logits(hidden)

        torch.testing.assert_close(
            logits,
            torch.full_like(logits, -torch.log(torch.tensor(2.0))),
        )

    def test_finalize_rejects_wrong_feature_dimension_for_both_strategies(self) -> None:
        for model in strategies():
            state = model.run_step(
                None,
                torch.ones(2, 2),
                lambda computation: computation.raw_hidden,
            )
            with self.subTest(strategy=type(model).__name__):
                with self.assertRaisesRegex(ValueError, "final dimension"):
                    model.finalize(state, torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
